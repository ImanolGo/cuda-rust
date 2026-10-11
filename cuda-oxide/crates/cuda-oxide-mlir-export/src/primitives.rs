/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pinned instruction-level consumer, independent of CuTe source semantics.

use crate::{CutlassFullCuteMlir22, ExportManifest, MlirConsumerProfile, ProfileError};
use dialect_mir::types::*;
use pliron::{
    attribute::{Attribute, attr_type},
    builtin::{
        attributes::{DictAttr, VecAttr},
        ops::ModuleOp,
        type_interfaces::FunctionTypeInterface,
    },
    context::Context,
    dialect::DialectName,
    linked_list::ContainsLinkedList,
    location::{Located, Location},
    op::Op,
    operation::{Operation, verify_operation},
    printable::Printable,
    r#type::{TypeHandle, Typed, type_cast},
};
use pliron_mlir_export::{MlirModule, TranslationRegistry, translate_module};
use std::collections::HashSet;

/// CUTLASS 4.7 / MLIR 22 primitives, initially validated for SM100a only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CutlassPrimitivesMlir22 {
    gpu_arch: String,
}

impl CutlassPrimitivesMlir22 {
    pub const NAME: &'static str = "cutlass-primitives-mlir22";

    pub fn new(gpu_arch: impl Into<String>) -> Result<Self, ProfileError> {
        let gpu_arch = gpu_arch.into();
        if gpu_arch != "sm_100a" {
            return Err(ProfileError::UnsupportedArchitecture(gpu_arch));
        }
        Ok(Self { gpu_arch })
    }
}

impl MlirConsumerProfile for CutlassPrimitivesMlir22 {
    fn name(&self) -> &'static str {
        Self::NAME
    }
    fn manifest(&self) -> ExportManifest {
        ExportManifest {
            profile: Self::NAME.into(),
            exporter_version: env!("CARGO_PKG_VERSION").into(),
            pliron_revision: CutlassFullCuteMlir22::PLIRON_REVISION.into(),
            consumer_revision: CutlassFullCuteMlir22::CONSUMER_REVISION.into(),
            llvm_major: 22,
            gpu_arch: self.gpu_arch.clone(),
            pointer_model: "opaque".into(),
            index_bits: 64,
        }
    }
    fn build_registry(&self) -> Result<TranslationRegistry, ProfileError> {
        let mut r = TranslationRegistry::new();
        crate::register_builtin_pack(&mut r)?;
        crate::register_mir_core_pack(&mut r)?;
        crate::register_nvvm_sreg_pack(&mut r)?;
        crate::register_nvvm_tcgen05_pack(&mut r)?;
        crate::register_nvvm_cluster_pack(&mut r)?;
        r.seal();
        Ok(r)
    }
    fn verify_source_module(&self, ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError> {
        verify_primitives_source(ctx, module)
    }
    fn translate_module(
        &self,
        ctx: &Context,
        module: &ModuleOp,
    ) -> Result<MlirModule, ProfileError> {
        self.verify_source_module(ctx, module)?;
        let mut target = translate_module(
            ctx,
            module,
            &self.build_registry()?,
            &self.translation_config(),
        )?;
        crate::profile::add_cutlass_executable_envelope(&mut target)
            .map_err(ProfileError::ExecutableEnvelope)?;
        Ok(target)
    }
}

/// Check before preparation can erase ghost types or opaque pointees.
pub fn verify_primitives_source(ctx: &Context, module: &ModuleOp) -> Result<(), ProfileError> {
    let mut scan = SourceScan {
        ctx,
        cute: DialectName::try_new("cute").unwrap(),
        seen: HashSet::new(),
    };
    let mut pending = vec![module.get_operation()];
    while let Some(ptr) = pending.pop() {
        let id = Operation::get_opid(ptr, ctx);
        if id.dialect == scan.cute {
            return Err(rejected(format!("CuTe operation `{id}`")));
        }
        let op = ptr.deref(ctx);
        for value in op.operands().chain(op.results()) {
            scan.ty(value.get_type(ctx))?;
        }
        for attr in op.attributes.0.values() {
            scan.attr(&**attr)?;
        }
        scan.location(&op.loc())?;
        for region in op.regions() {
            for block in region.deref(ctx).iter(ctx) {
                let block = block.deref(ctx);
                for arg in block.arguments() {
                    scan.ty(arg.get_type(ctx))?;
                }
                for attr in block.attributes.0.values() {
                    scan.attr(&**attr)?;
                }
                scan.location(&block.loc())?;
                pending.extend(block.iter(ctx));
            }
        }
    }
    dialect_mir::verification::verify_pointer_kind_producers(ctx, module.get_operation())
        .map_err(|e| ProfileError::SourceValidation(e.disp(ctx).to_string()))?;
    verify_operation(module.get_operation(), ctx)
        .map_err(|e| ProfileError::SourceValidation(e.disp(ctx).to_string()))
}

fn rejected(item: String) -> ProfileError {
    ProfileError::SourceValidation(format!("{item} is unsupported by the primitives profile"))
}

struct SourceScan<'a> {
    ctx: &'a Context,
    cute: DialectName,
    seen: HashSet<TypeHandle>,
}
impl SourceScan<'_> {
    fn ty(&mut self, handle: TypeHandle) -> Result<(), ProfileError> {
        if !self.seen.insert(handle) {
            return Ok(());
        }
        let ty = handle.deref(self.ctx);
        let id = ty.get_type_id();
        if id.dialect == self.cute {
            return Err(rejected(format!("CuTe type `{id}`")));
        }
        let children: Vec<TypeHandle> =
            if let Some(f) = type_cast::<dyn FunctionTypeInterface>(&*ty) {
                f.arg_types().into_iter().chain(f.res_types()).collect()
            } else if let Some(t) = ty.downcast_ref::<MirPtrType>() {
                vec![t.pointee]
            } else if let Some(t) = ty.downcast_ref::<MirTupleType>() {
                t.types.clone()
            } else if let Some(t) = ty.downcast_ref::<MirStructType>() {
                t.field_types.clone()
            } else if let Some(t) = ty.downcast_ref::<MirUnionType>() {
                t.field_types.clone()
            } else if let Some(t) = ty.downcast_ref::<MirArrayType>() {
                vec![t.element_ty]
            } else if let Some(t) = ty.downcast_ref::<MirSliceType>() {
                vec![t.element_ty]
            } else if let Some(t) = ty.downcast_ref::<MirDisjointSliceType>() {
                std::iter::once(t.element_ty)
                    .chain(t.space_tys.iter().copied())
                    .collect()
            } else if let Some(t) = ty.downcast_ref::<MirEnumType>() {
                std::iter::once(t.discriminant_ty)
                    .chain(t.all_field_types.iter().copied())
                    .collect()
            } else if let Some(t) = ty.downcast_ref::<llvm_export::types::ArrayType>() {
                vec![t.elem_type()]
            } else if let Some(t) = ty.downcast_ref::<llvm_export::types::VectorType>() {
                vec![t.elem_type()]
            } else if let Some(t) = ty.downcast_ref::<llvm_export::types::StructType>() {
                t.fields().collect()
            } else {
                vec![]
            };
        for child in children {
            self.ty(child)?;
        }
        Ok(())
    }
    fn attr(&mut self, attr: &dyn Attribute) -> Result<(), ProfileError> {
        let id = attr.get_attr_id();
        if id.dialect == self.cute {
            return Err(rejected(format!("CuTe attribute `{id}`")));
        }
        if let Some(ty) = attr_type(attr, self.ctx) {
            self.ty(ty)?;
        }
        if let Some(v) = attr.downcast_ref::<VecAttr>() {
            for a in &v.0 {
                self.attr(&**a)?;
            }
        }
        if let Some(d) = attr.downcast_ref::<DictAttr>() {
            for a in d.0.0.values() {
                self.attr(&**a)?;
            }
        }
        Ok(())
    }
    fn location(&mut self, loc: &Location) -> Result<(), ProfileError> {
        match loc {
            Location::Fused {
                metadata,
                locations,
            } => {
                if let Some(a) = metadata {
                    self.attr(&**a)?;
                }
                for loc in locations {
                    self.location(loc)?;
                }
            }
            Location::Named { child_loc, .. } => self.location(child_loc)?,
            Location::CallSite { callee, caller } => {
                self.location(callee)?;
                self.location(caller)?;
            }
            _ => {}
        }
        Ok(())
    }
}
