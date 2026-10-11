/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Static shared declarations, deduplicated inside the executable GPU module.
use dialect_mir::ops::MirSharedAllocOp;
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    r#type::Typed,
};
use pliron_mlir_export::{
    MlirAttribute, MlirBlock, MlirOperation, MlirRegion, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub(crate) fn register(registry: &mut TranslationRegistry) -> Result<(), TranslationError> {
    registry.register_operation::<MirSharedAllocOp>(SharedAllocation)
}
struct SharedAllocation;
impl OperationTranslation for SharedAllocation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        s: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        if !input.operands.is_empty()
            || input.results.len() != 1
            || !input.regions.is_empty()
            || !input.successors.is_empty()
        {
            return Err("invalid shared allocation shape".into());
        }
        for key in input.attributes.keys() {
            if !matches!(
                key.as_str(),
                "elem_type" | "size" | "alloc_key" | "source_key" | "source_name" | "mir_alignment"
            ) {
                return Err(format!("unsupported shared allocation attribute {key}"));
            }
        }
        let allocation = MirSharedAllocOp::new(source);
        let element = allocation
            .get_attr_elem_type(ctx)
            .ok_or("missing shared element type")?
            .get_type(ctx);
        let size = allocation
            .get_attr_size(ctx)
            .ok_or("missing shared size")?
            .value()
            .to_u64();
        if size == 0 {
            return Err("empty static shared allocation".into());
        }
        let alignment = allocation
            .get_alignment_value(ctx)
            .filter(|align| *align != 0)
            .or_else(|| crate::mir_memory::source_natural_alignment(ctx, element))
            .ok_or("unknown shared alignment")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err("invalid shared alignment".into());
        }
        let key = allocation
            .get_attr_alloc_key(ctx)
            .map(|key| String::from((*key).clone()))
            .unwrap_or_else(|| format!("unkeyed-{}", input.results[0].id.0));
        let symbol = format!("__cuda_oxide_shared_{:x}", Sha256::digest(key.as_bytes()));
        let ty = s.translate_type(element).map_err(|e| e.to_string())?;
        let global_type = MlirType::dialect(format!(
            "!llvm.array<{size} x {}>",
            crate::mir_memory::type_spelling(&ty)
        ))?;
        let mut global = MlirOperation::new("llvm.mlir.global")?;
        // Preserve source identity while deduplicating: signedness and enum
        // identity may disappear from the lowered storage type. Removed below.
        global.attributes.insert(
            "__source_element_identity".into(),
            MlirAttribute::String(format!("{element:?}")),
        );
        global
            .properties
            .insert("sym_name".into(), MlirAttribute::String(symbol.clone()));
        global.properties.insert(
            "global_type".into(),
            MlirAttribute::Type(global_type.clone()),
        );
        global.properties.insert(
            "linkage".into(),
            MlirAttribute::dialect("#llvm.linkage<internal>")?,
        );
        global.properties.insert(
            "addr_space".into(),
            MlirAttribute::Integer {
                value: 3,
                ty: MlirType::Integer(32),
            },
        );
        global.properties.insert(
            "alignment".into(),
            MlirAttribute::Integer {
                value: alignment.into(),
                ty: MlirType::Integer(64),
            },
        );
        let id = s.fresh_value();
        let mut undef = MlirOperation::new("llvm.mlir.undef")?;
        undef.results.push(MlirResult {
            id,
            ty: global_type.clone(),
        });
        let mut ret = MlirOperation::new("llvm.return")?;
        ret.operands.push(MlirValueUse {
            id,
            ty: global_type,
        });
        global.regions.push(MlirRegion {
            blocks: vec![MlirBlock {
                id: s.fresh_block(),
                arguments: vec![],
                operations: vec![undef, ret],
            }],
        });
        global.location = input.location.clone();
        let mut address = MlirOperation::new("llvm.mlir.addressof")?;
        address.results = input.results;
        address
            .properties
            .insert("global_name".into(), MlirAttribute::SymbolRef(symbol));
        address.location = input.location;
        // The profile hoists these declarations after translation; source IR
        // is immutable and its local allocation operation still yields an SSA address.
        Ok(vec![global, address])
    }
}

pub(crate) fn hoist_shared_globals(operations: &mut Vec<MlirOperation>) -> Result<(), String> {
    fn gather(
        operations: &mut Vec<MlirOperation>,
        globals: &mut BTreeMap<String, MlirOperation>,
    ) -> Result<(), String> {
        let mut remaining = vec![];
        for mut operation in std::mem::take(operations) {
            if operation.name == "llvm.mlir.global" {
                let Some(MlirAttribute::String(symbol)) = operation.properties.get("sym_name")
                else {
                    return Err("global without symbol".into());
                };
                if let Some(previous) = globals.get(symbol) {
                    if previous.properties != operation.properties
                        || previous.attributes != operation.attributes
                    {
                        return Err(format!(
                            "incompatible declarations for shared allocation {symbol}"
                        ));
                    }
                } else {
                    globals.insert(symbol.clone(), operation);
                }
            } else {
                for region in &mut operation.regions {
                    for block in &mut region.blocks {
                        gather(&mut block.operations, globals)?;
                    }
                }
                remaining.push(operation);
            }
        }
        *operations = remaining;
        Ok(())
    }
    let mut globals = BTreeMap::new();
    gather(operations, &mut globals)?;
    operations.splice(
        0..0,
        globals.into_values().map(|mut global| {
            global.attributes.remove("__source_element_identity");
            global
        }),
    );
    Ok(())
}
