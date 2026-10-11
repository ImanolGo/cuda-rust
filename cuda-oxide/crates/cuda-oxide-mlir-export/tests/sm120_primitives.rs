// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_oxide_mlir_export::{CutlassPrimitivesMlir22, MlirConsumerProfile};
use dialect_mir::{
    ops::{MirConstantOp, MirFuncOp, MirReturnOp},
    types::MirPtrType,
};
use dialect_nvvm::ops::*;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};
use pliron_mlir_export::render_module;

struct Fixture {
    ctx: Context,
    module: ModuleOp,
    block: Ptr<BasicBlock>,
    types: Vec<TypeHandle>,
}
impl Fixture {
    fn new() -> Self {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        let i8 = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let i32 = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let i64 = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32 = FP32Type::get(&ctx).into();
        let types = vec![
            MirPtrType::get(&mut ctx, i32, true, 0).into(),
            MirPtrType::get(&mut ctx, i8, true, 3).into(),
            MirPtrType::get(&mut ctx, i64, true, 3).into(),
            MirPtrType::get(&mut ctx, i8, false, 0).into(),
            i32,
            i32,
            f32,
            f32,
            f32,
            f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
        ];
        let module = ModuleOp::new(&mut ctx, "sm120".try_into().unwrap());
        let ft = FunctionType::get(&ctx, types.clone(), vec![]);
        let raw = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        MirFuncOp::new(&mut ctx, raw, TypeAttr::new(ft.into()))
            .set_symbol_name(&mut ctx, "sm120_primitives".try_into().unwrap());
        raw.deref_mut(&ctx).attributes.set(
            "gpu_kernel".try_into().unwrap(),
            StringAttr::new("true".into()),
        );
        module.append_operation(&mut ctx, raw, 0);
        let block = BasicBlock::new(&mut ctx, None, types.clone());
        block.insert_at_back(raw.deref(&ctx).get_region(0), &ctx);
        Self {
            ctx,
            module,
            block,
            types,
        }
    }
    fn arg(&self, i: usize) -> Value {
        self.block.deref(&self.ctx).get_argument(i)
    }
    fn append<O: Op>(
        &mut self,
        marker: &str,
        operands: Vec<Value>,
        types: Vec<TypeHandle>,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            &mut self.ctx,
            O::get_concrete_op_info(),
            types,
            operands,
            vec![],
            0,
        );
        if !marker.is_empty() {
            op.deref_mut(&self.ctx).attributes.set(
                "cuda_oxide_intrinsic_marker".try_into().unwrap(),
                StringAttr::new(marker.into()),
            );
        }
        op.insert_at_back(self.block, &self.ctx);
        op
    }
    fn constant(&mut self, width: u32, value: u64) -> Value {
        let ty = IntegerType::get(&self.ctx, width, Signedness::Unsigned);
        let op = self.append::<MirConstantOp>("", vec![], vec![ty.into()]);
        MirConstantOp::new(op).set_attr_value(
            &self.ctx,
            IntegerAttr::new(
                ty,
                APInt::from_u64(value, (width as usize).try_into().unwrap()),
            ),
        );
        op.deref(&self.ctx).get_result(0)
    }
    fn finish(&mut self) {
        self.append::<MirReturnOp>("", vec![], vec![]);
    }
    fn translate(&self) -> Result<String, String> {
        CutlassPrimitivesMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&self.ctx, &self.module)
            .map(|m| render_module(&m))
            .map_err(|e| e.to_string())
    }
    fn load(&mut self) -> Ptr<Operation> {
        let address = self.arg(0);
        let load = LdmatrixOp::build(
            &mut self.ctx,
            address,
            LdmatrixShapeAttr::M8n8,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B16,
            LdmatrixStateSpaceAttr::Shared,
        );
        load.deref_mut(&self.ctx).attributes.set(
            "cuda_oxide_intrinsic_marker".try_into().unwrap(),
            StringAttr::new("v1:i0013".into()),
        );
        load.insert_at_back(self.block, &self.ctx);
        load
    }

    fn mma(&mut self) -> Ptr<Operation> {
        let mut operands = (6..16).map(|i| self.arg(i)).collect::<Vec<_>>();
        // Distinct definitions prove each selector and scale retains its slot.
        operands.push(self.arg(16));
        operands.push(self.constant(16, 0));
        operands.push(self.constant(16, 0));
        operands.push(self.arg(17));
        operands.push(self.constant(16, 0));
        operands.push(self.constant(16, 0));
        let op = self.append::<RegisterMmaOp>("v1:i1030", operands, vec![self.types[6]; 4]);
        let mma = RegisterMmaOp::new(op);
        mma.set_attr_nvvm_register_mma_shape(&self.ctx, RegisterMmaShapeAttr::M16n8k64);
        mma.set_attr_nvvm_register_mma_operation(&self.ctx, RegisterMmaOperationAttr::Multiply);
        mma.set_attr_nvvm_register_mma_kind(&self.ctx, RegisterMmaKindAttr::Mxf4);
        mma.set_attr_nvvm_register_mma_accumulator(&self.ctx, RegisterMmaAccumulatorAttr::F32);
        mma.set_attr_nvvm_register_mma_a_element(&self.ctx, RegisterMmaElementAttr::E2m1);
        mma.set_attr_nvvm_register_mma_b_element(&self.ctx, RegisterMmaElementAttr::E2m1);
        mma.set_attr_nvvm_register_mma_a_layout(&self.ctx, RegisterMmaLayoutAttr::Row);
        mma.set_attr_nvvm_register_mma_b_layout(&self.ctx, RegisterMmaLayoutAttr::Col);
        mma.set_attr_nvvm_register_mma_overflow(&self.ctx, RegisterMmaOverflowAttr::NotApplicable);
        op
    }
}

#[test]
fn sm120_mxf4_preserves_a_b_c_scale_order_and_extracts_all_results() {
    let mut f = Fixture::new();
    let mma = f.mma();
    let results = mma.deref(&f.ctx).results().collect::<Vec<_>>();
    let top = f.append::<CvtF16x2F32Op>("v1:i0081", results[..2].to_vec(), vec![f.types[4]]);
    let bottom = f.append::<CvtF16x2F32Op>("v1:i0081", results[2..].to_vec(), vec![f.types[4]]);
    let top = top.deref(&f.ctx).get_result(0);
    let bottom = bottom.deref(&f.ctx).get_result(0);
    f.append::<StmatrixM8n8X2Op>("v1:i0301", vec![f.arg(1), top, bottom], vec![]);
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("\"nvvm.mma.block_scale\"(%v10, %v11, %v12, %v13, %v14, %v15, %v6, %v7, %v8, %v9, %v16, %v18, %v19, %v17, %v20, %v21)"), "{text}");
    for expected in [
        "#nvvm.shape<m = 16, n = 8, k = 64>",
        "#nvvm.scale_vec_size<x2>",
        "#nvvm.block_scale_kind<mxf4>",
        "#nvvm.block_scale_format<ue8m0>",
        "operandSegmentSizes = array<i32: 4, 2, 4, 1, 1, 1, 1, 1, 1>",
    ] {
        assert!(text.contains(expected), "{text}");
    }
    assert_eq!(text.matches("#nvvm.mma_type<e2m1>").count(), 2);
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}

#[test]
fn sm120_mxf4_rejects_other_variants_and_forged_identity() {
    let mut f = Fixture::new();
    let op = f.mma();
    f.finish();
    let mma = RegisterMmaOp::new(op);
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k32);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Mxf8f6f4);
    let error = f.translate().unwrap_err();
    assert!(error.contains("supports m16n8k64 MXF4"), "{error}");
    mma.set_attr_nvvm_register_mma_shape(&f.ctx, RegisterMmaShapeAttr::M16n8k64);
    mma.set_attr_nvvm_register_mma_kind(&f.ctx, RegisterMmaKindAttr::Mxf4);
    op.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0858".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i1030")
    );
}

#[test]
fn sm100_rejects_sm120_mxf4() {
    let mut f = Fixture::new();
    f.mma();
    f.finish();
    let error = CutlassPrimitivesMlir22::new("sm_100a")
        .unwrap()
        .translate_module(&f.ctx, &f.module)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("MXF4 register MMA requires sm_120a"),
        "{error}"
    );
}

#[test]
fn sm120_load_pack_and_store_preserve_register_and_lane_order() {
    let mut f = Fixture::new();
    let load = f.load();
    let pack = f.append::<CvtF16x2F32Op>("v1:i0081", vec![f.arg(6), f.arg(7)], vec![f.types[4]]);
    let registers = load.deref(&f.ctx).results().collect::<Vec<_>>();
    let packed = pack.deref(&f.ctx).get_result(0);
    f.append::<StmatrixM8n8X2Op>("v1:i0301", vec![f.arg(1), packed, registers[3]], vec![]);
    f.finish();
    let text = f.translate().unwrap();
    assert!(text.contains("\"nvvm.ldmatrix\""), "{text}");
    assert!(text.contains("num = 4 : i32"), "{text}");
    assert_eq!(text.matches("\"llvm.extractvalue\"").count(), 4);
    assert!(text.contains("\"llvm.call_intrinsic\"(%v7, %v6)"), "{text}");
    assert!(text.contains("llvm.nvvm.ff2f16x2.rn"), "{text}");
    assert!(text.contains("\"nvvm.stmatrix\""), "{text}");
    assert!(text.contains("#nvvm.mma_layout<row>"), "{text}");
    assert!(!text.contains("cuda_oxide_intrinsic_marker"));
}

#[test]
fn sm120_rejects_other_matrix_load_selectors_and_forged_identity() {
    let mut f = Fixture::new();
    let load = f.load();
    f.finish();
    LdmatrixOp::new(load).set_attr_nvvm_ldmatrix_layout(&f.ctx, LdmatrixLayoutAttr::Transposed);
    assert!(
        f.translate()
            .unwrap_err()
            .contains("m8n8.x4.b16 non-transposed")
    );
    LdmatrixOp::new(load).set_attr_nvvm_ldmatrix_layout(&f.ctx, LdmatrixLayoutAttr::Normal);
    load.deref_mut(&f.ctx).attributes.set(
        "cuda_oxide_intrinsic_marker".try_into().unwrap(),
        StringAttr::new("v1:i0001".into()),
    );
    assert!(
        f.translate()
            .unwrap_err()
            .contains("expected intrinsic marker v1:i0013")
    );
}

#[test]
fn sm120_tma_uses_local_shared_barrier_and_no_multicast() {
    let mut f = Fixture::new();
    let mask = f.constant(16, 0);
    let hint = f.constant(64, 0);
    f.append::<CpAsyncBulkTensorG2sTile2dOp>(
        "v1:i0329",
        vec![f.arg(1), f.arg(2), f.arg(3), f.arg(4), f.arg(5), mask, hint],
        vec![],
    );
    f.finish();
    let text = f.translate().unwrap();
    assert!(
        text.contains("\"nvvm.cp.async.bulk.tensor.shared.cluster.global\""),
        "{text}"
    );
    assert!(
        text.contains("operandSegmentSizes = array<i32: 1, 1, 2, 1, 0, 0, 0, 0>"),
        "{text}"
    );
    assert!(text.contains("useIntrinsic = false"), "{text}");
    assert!(!text.contains("cta_2"));
}

#[test]
fn sm120_tma_rejects_nonzero_multicast_mask_or_cache_hint() {
    for (mask, hint) in [(1, 0), (0, 1)] {
        let mut f = Fixture::new();
        let mask = f.constant(16, mask);
        let hint = f.constant(64, hint);
        f.append::<CpAsyncBulkTensorG2sTile2dOp>(
            "v1:i0329",
            vec![f.arg(1), f.arg(2), f.arg(3), f.arg(4), f.arg(5), mask, hint],
            vec![],
        );
        f.finish();
        assert!(
            f.translate()
                .unwrap_err()
                .contains("zero multicast mask and cache hint")
        );
    }
}

#[test]
fn sm120_rejects_tcgen05_before_translation() {
    let mut f = Fixture::new();
    f.append::<Tcgen05LoadWaitOp>("v1:i0357", vec![], vec![]);
    f.finish();
    assert!(
        f.translate()
            .unwrap_err()
            .contains("TCGen05 is unsupported on sm_120a")
    );
}

#[test]
fn sm120_tma_requires_pointer_barrier_carrier() {
    let mut f = Fixture::new();
    let mask = f.constant(16, 0);
    let hint = f.constant(64, 0);
    f.append::<CpAsyncBulkTensorG2sTile2dOp>(
        "v1:i0329",
        vec![f.arg(1), f.arg(4), f.arg(3), f.arg(4), f.arg(5), mask, hint],
        vec![],
    );
    f.finish();
    assert!(
        f.translate()
            .unwrap_err()
            .contains("TMA requires pointer carriers")
    );
}
