// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::{
    attributes::{FieldIndexAttr, VariantIndexAttr},
    ops::*,
    types::*,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use pliron_mlir_export::render_module;
fn append<O: Op>(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    types: Vec<TypeHandle>,
    values: Vec<Value>,
) -> Ptr<Operation> {
    let o = Operation::new(ctx, O::get_concrete_op_info(), types, values, vec![], 0);
    o.insert_at_back(b, ctx);
    o
}
fn function(
    ctx: &mut Context,
    args: Vec<TypeHandle>,
    returns: Vec<TypeHandle>,
) -> (ModuleOp, Ptr<BasicBlock>) {
    let m = ModuleOp::new(ctx, "storage".try_into().unwrap());
    let ft = FunctionType::get(ctx, args.clone(), returns);
    let f = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    MirFuncOp::new(ctx, f, TypeAttr::new(ft.into()))
        .set_symbol_name(ctx, "test".try_into().unwrap());
    m.append_operation(ctx, f, 0);
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(f.deref(ctx).get_region(0), ctx);
    (m, b)
}
fn render(ctx: &Context, m: &ModuleOp) -> Result<String, String> {
    CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(ctx, m)
        .map(|m| render_module(&m))
        .map_err(|e| e.to_string())
}
#[test]
fn checked_arithmetic_preserves_signedness_and_padded_overflow_tuple() {
    for signedness in [Signedness::Signed, Signedness::Unsigned] {
        for mul in [false, true] {
            let mut ctx = Context::new();
            dialect_mir::register(&mut ctx);
            let word: TypeHandle = IntegerType::get(&ctx, 64, signedness).into();
            let flag: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
            let tuple: TypeHandle = MirTupleType::get_with_layout(
                &mut ctx,
                vec![word, flag],
                vec![0, 1],
                vec![0, 8],
                16,
                8,
            )
            .into();
            let (m, b) = function(&mut ctx, vec![word, word], vec![tuple]);
            let args = b.deref(&ctx).arguments().collect();
            let op = if mul {
                append::<MirCheckedMulOp>(&mut ctx, b, vec![tuple], args)
            } else {
                append::<MirCheckedAddOp>(&mut ctx, b, vec![tuple], args)
            };
            let v = op.deref(&ctx).get_result(0);
            append::<MirReturnOp>(&mut ctx, b, vec![], vec![v]);
            let text = render(&ctx, &m).unwrap();
            let sign = if signedness == Signedness::Signed {
                "s"
            } else {
                "u"
            };
            let name = if mul { "mul" } else { "add" };
            assert!(
                text.contains(&format!("llvm.{sign}{name}.with.overflow.i64")),
                "{text}"
            );
            assert!(
                text.contains("!llvm.struct<(i64, i1, !llvm.array<7 x i8>)>"),
                "{text}"
            );
            assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 2);
            assert_eq!(text.matches("\"llvm.insertvalue\"").count(), 2);
        }
    }
}
#[test]
fn direct_enum_uses_physical_offsets_for_tag_and_active_payload() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let e: TypeHandle = MirEnumType::get_with_layout(
        &mut ctx,
        "Result".into(),
        tag,
        vec![7, 19],
        vec![
            EnumVariant::new_with_layout("Ok".into(), vec![word], vec![8], vec![8]),
            EnumVariant::new_with_layout("Err".into(), vec![tag], vec![1], vec![1]),
        ],
        0,
        16,
        8,
    )
    .into();
    let (m, b) = function(&mut ctx, vec![word], vec![word]);
    let arg = b.deref(&ctx).get_argument(0);
    let c = append::<MirConstructEnumOp>(&mut ctx, b, vec![e], vec![arg]);
    MirConstructEnumOp::new(c).set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    let v = c.deref(&ctx).get_result(0);
    append::<MirGetDiscriminantOp>(&mut ctx, b, vec![tag], vec![v]);
    let p = append::<MirEnumPayloadOp>(&mut ctx, b, vec![word], vec![v]);
    let p = MirEnumPayloadOp::new(p);
    p.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    p.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    let v = p.get_operation().deref(&ctx).get_result(0);
    append::<MirReturnOp>(&mut ctx, b, vec![], vec![v]);
    let text = render(&ctx, &m).unwrap();
    assert!(text.contains("value = 7 : i8"));
    assert_eq!(
        text.matches("rawConstantIndices = array<i32: 8>").count(),
        2
    );
    assert_eq!(
        text.matches("rawConstantIndices = array<i32: 0>").count(),
        2
    );
    assert!(text.contains("!llvm.struct<(array<16 x i8>, array<0 x i64>)>"));
}
fn shared_case(conflict: &str) -> Result<String, String> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let unsigned = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signed = IntegerType::get(&ctx, 32, Signedness::Signed);
    let size_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let (m, b) = function(&mut ctx, vec![], vec![]);
    for i in 0..2 {
        let element: TypeHandle = if conflict == "type" && i == 1 {
            signed.into()
        } else {
            unsigned.into()
        };
        let ptr: TypeHandle = MirPtrType::get(&mut ctx, element, true, 3).into();
        let a = append::<MirSharedAllocOp>(&mut ctx, b, vec![ptr], vec![]);
        let a = MirSharedAllocOp::new(a);
        a.set_attr_elem_type(&ctx, TypeAttr::new(element));
        a.set_attr_size(
            &ctx,
            IntegerAttr::new(
                size_ty,
                APInt::from_u64(
                    if conflict == "size" && i == 1 { 8 } else { 4 },
                    64.try_into().unwrap(),
                ),
            ),
        );
        a.set_attr_alloc_key(
            &ctx,
            StringAttr::new(if conflict == "distinct" {
                format!("key{i}")
            } else {
                "key".into()
            }),
        );
        a.set_alignment_value(
            &mut ctx,
            if conflict == "alignment" && i == 1 {
                16
            } else {
                0
            },
        );
    }
    append::<MirReturnOp>(&mut ctx, b, vec![], vec![]);
    render(&ctx, &m)
}
#[test]
fn shared_declarations_deduplicate_only_identical_storage_and_zero_means_natural() {
    let text = shared_case("same").unwrap();
    assert_eq!(text.matches("\"llvm.mlir.global\"").count(), 1);
    assert!(text.contains("alignment = 4 : i64"));
    assert!(!text.contains("__source_element_identity"));
    assert_eq!(
        shared_case("distinct")
            .unwrap()
            .matches("\"llvm.mlir.global\"")
            .count(),
        2
    );
    for conflict in ["type", "size", "alignment"] {
        assert!(
            shared_case(conflict)
                .unwrap_err()
                .contains("incompatible declarations"),
            "{conflict}"
        );
    }
}
