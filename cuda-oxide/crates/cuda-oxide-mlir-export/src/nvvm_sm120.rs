/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Register-MMA and shared-memory primitives for SM120, using standard NVVM.
//! Source intrinsic identities and structural selectors are checked before any
//! source-only attributes are removed. No CuTe operations or types are involved.

use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::*;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirFloatType, MlirLocation, MlirOperation, MlirResult, MlirType,
    MlirValueUse, OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};

/// The exact instruction variants used by the SM120 block-scaled GEMM.
pub fn register_nvvm_sm120_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<LdmatrixShapeAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixMultiplicityAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixElementAttr>(DropAttribute)?;
    registry.register_attribute::<LdmatrixStateSpaceAttr>(DropAttribute)?;
    registry.register_operation::<LdmatrixOp>(LoadTranslation)?;
    registry.register_operation::<StmatrixM8n8X2Op>(StoreTranslation)?;
    registry.register_operation::<CvtF16x2F32Op>(PackTranslation)?;
    registry.register_operation::<CpAsyncBulkTensorG2sTile2dOp>(TmaTranslation)?;
    registry.register_attribute::<RegisterMmaShapeAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaOperationAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaKindAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaAccumulatorAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaElementAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<RegisterMmaOverflowAttr>(DropAttribute)?;
    registry.register_operation::<RegisterMmaOp>(MmaTranslation)?;
    Ok(())
}

struct MmaTranslation;
impl OperationTranslation for MmaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let mma = RegisterMmaOp::new(source);
        mma.verify(ctx).map_err(|e| e.to_string())?;
        if mma.get_attr_nvvm_register_mma_shape(ctx).as_deref()
            != Some(&RegisterMmaShapeAttr::M16n8k64)
            || mma.get_attr_nvvm_register_mma_kind(ctx).as_deref()
                != Some(&RegisterMmaKindAttr::Mxf4)
            || mma.get_attr_nvvm_register_mma_operation(ctx).as_deref()
                != Some(&RegisterMmaOperationAttr::Multiply)
            || mma.get_attr_nvvm_register_mma_accumulator(ctx).as_deref()
                != Some(&RegisterMmaAccumulatorAttr::F32)
            || mma.get_attr_nvvm_register_mma_a_element(ctx).as_deref()
                != Some(&RegisterMmaElementAttr::E2m1)
            || mma.get_attr_nvvm_register_mma_b_element(ctx).as_deref()
                != Some(&RegisterMmaElementAttr::E2m1)
            || mma.get_attr_nvvm_register_mma_a_layout(ctx).as_deref()
                != Some(&RegisterMmaLayoutAttr::Row)
            || mma.get_attr_nvvm_register_mma_b_layout(ctx).as_deref()
                != Some(&RegisterMmaLayoutAttr::Col)
            || mma.get_attr_nvvm_register_mma_overflow(ctx).as_deref()
                != Some(&RegisterMmaOverflowAttr::NotApplicable)
        {
            return Err("SM120 register MMA mapping supports m16n8k64 MXF4 row/col E2M1 with F32 accumulation and UE8M0 scale_vec::2X only".into());
        }
        require_input(&mut input, "v1:i1030", 16, 4)?;
        // Canonical source order is C, A, B, scales; NVVM uses A, B, C, scales.
        let operands = input.operands[4..10]
            .iter()
            .chain(&input.operands[..4])
            .chain(&input.operands[10..])
            .cloned()
            .collect();
        let (result, aggregate) = fresh(
            session,
            MlirType::dialect("!llvm.struct<(f32, f32, f32, f32)>")?,
        );
        let mut op = operation(
            "nvvm.mma.block_scale",
            operands,
            vec![result],
            &input.location,
        )?;
        property(&mut op, "shape", "#nvvm.shape<m = 16, n = 8, k = 64>")?;
        property(&mut op, "multiplicandAPtxType", "#nvvm.mma_type<e2m1>")?;
        property(&mut op, "multiplicandBPtxType", "#nvvm.mma_type<e2m1>")?;
        property(&mut op, "scaleVecSize", "#nvvm.scale_vec_size<x2>")?;
        property(
            &mut op,
            "blockScaleFormat",
            "#nvvm.block_scale_format<ue8m0>",
        )?;
        property(&mut op, "kind", "#nvvm.block_scale_kind<mxf4>")?;
        op.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![4, 2, 4, 1, 1, 1, 1, 1, 1]),
        );
        let mut operations = vec![op];
        extract_results(&mut operations, aggregate, input.results, &input.location)?;
        Ok(operations)
    }
}

struct LoadTranslation;
impl OperationTranslation for LoadTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let load = LdmatrixOp::new(source);
        load.verify(ctx).map_err(|e| e.to_string())?;
        if load.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() != Some(&LdmatrixShapeAttr::M8n8)
            || load.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref()
                != Some(&LdmatrixMultiplicityAttr::X4)
            || load.get_attr_nvvm_ldmatrix_layout(ctx).as_deref()
                != Some(&LdmatrixLayoutAttr::Normal)
            || load.get_attr_nvvm_ldmatrix_element(ctx).as_deref()
                != Some(&LdmatrixElementAttr::B16)
            || load.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref()
                != Some(&LdmatrixStateSpaceAttr::Shared)
        {
            return Err(
                "SM120 matrix load mapping supports m8n8.x4.b16 non-transposed shared loads only"
                    .into(),
            );
        }
        require_input(&mut input, "v1:i0013", 1, 4)?;
        let mut operations = vec![];
        let pointer = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            3,
            &input.location,
        )?;
        let (result, aggregate) = fresh(
            session,
            MlirType::dialect("!llvm.struct<(i32, i32, i32, i32)>")?,
        );
        let mut op = operation(
            "nvvm.ldmatrix",
            vec![pointer],
            vec![result],
            &input.location,
        )?;
        property(&mut op, "layout", "#nvvm.mma_layout<row>")?;
        op.properties.insert(
            "num".into(),
            MlirAttribute::Integer {
                value: 4,
                ty: MlirType::Integer(32),
            },
        );
        operations.push(op);
        extract_results(&mut operations, aggregate, input.results, &input.location)?;
        Ok(operations)
    }
}

struct StoreTranslation;
impl OperationTranslation for StoreTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        StmatrixM8n8X2Op::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0301", 3, 0)?;
        let mut operations = vec![];
        input.operands[0] = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            3,
            &input.location,
        )?;
        let mut op = operation("nvvm.stmatrix", input.operands, vec![], &input.location)?;
        property(&mut op, "layout", "#nvvm.mma_layout<row>")?;
        operations.push(op);
        Ok(operations)
    }
}

struct PackTranslation;
impl OperationTranslation for PackTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CvtF16x2F32Op::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0081", 2, 1)?;
        let (result, packed) = fresh(
            session,
            MlirType::Vector {
                shape: vec![2],
                element: Box::new(MlirType::Float(MlirFloatType::F16)),
            },
        );
        // Canonical Rust API is (low, high); LLVM's intrinsic is (high, low).
        input.operands.reverse();
        let mut call = operation(
            "llvm.call_intrinsic",
            input.operands,
            vec![result],
            &input.location,
        )?;
        call.properties.insert(
            "intrin".into(),
            MlirAttribute::String("llvm.nvvm.ff2f16x2.rn".into()),
        );
        call.properties.insert(
            "op_bundle_sizes".into(),
            MlirAttribute::DenseI32Array(vec![]),
        );
        call.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![2, 0]),
        );
        Ok(vec![
            call,
            operation("llvm.bitcast", vec![packed], input.results, &input.location)?,
        ])
    }
}

struct TmaTranslation;
impl OperationTranslation for TmaTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        CpAsyncBulkTensorG2sTile2dOp::new(source)
            .verify(ctx)
            .map_err(|e| e.to_string())?;
        require_input(&mut input, "v1:i0329", 7, 0)?;
        if input.operands[..3]
            .iter()
            .any(|operand| matches!(operand.ty, MlirType::Integer(_)))
        {
            return Err(
                "TMA requires pointer carriers for destination, barrier, and descriptor".into(),
            );
        }
        if constant_integer(ctx, source, 5)? != 0 || constant_integer(ctx, source, 6)? != 0 {
            return Err(
                "ordinary 2D TMA requires the importer's zero multicast mask and cache hint".into(),
            );
        }
        for index in [3, 4] {
            if input.operands[index].ty != MlirType::Integer(32) {
                return Err("2D TMA coordinates must be i32".into());
            }
        }
        if input.operands[5].ty != MlirType::Integer(16)
            || input.operands[6].ty != MlirType::Integer(64)
        {
            return Err("2D TMA mask/cache carriers must be i16/i64".into());
        }
        let mut operations = vec![];
        // The canonical LLVM g2s intrinsic encodes the destination in p7,
        // including for a non-multicast copy into the executing CTA's SMEM.
        let dst = pointer(
            session,
            &mut operations,
            input.operands[0].clone(),
            7,
            &input.location,
        )?;
        let barrier = pointer(
            session,
            &mut operations,
            input.operands[1].clone(),
            3,
            &input.location,
        )?;
        let descriptor = pointer(
            session,
            &mut operations,
            input.operands[2].clone(),
            0,
            &input.location,
        )?;
        let mut op = operation(
            "nvvm.cp.async.bulk.tensor.shared.cluster.global",
            vec![
                dst,
                descriptor,
                input.operands[3].clone(),
                input.operands[4].clone(),
                barrier,
            ],
            vec![],
            &input.location,
        )?;
        op.properties.insert(
            "operandSegmentSizes".into(),
            MlirAttribute::DenseI32Array(vec![1, 1, 2, 1, 0, 0, 0, 0]),
        );
        op.properties
            .insert("useIntrinsic".into(), MlirAttribute::Bool(false));
        operations.push(op);
        Ok(operations)
    }
}

fn require_input(
    input: &mut OperationInput,
    marker: &str,
    operands: usize,
    results: usize,
) -> Result<(), String> {
    if input.attributes.remove("cuda_oxide_intrinsic_marker")
        != Some(MlirAttribute::String(marker.into()))
    {
        return Err(format!(
            "SM120 operation expected intrinsic marker {marker}"
        ));
    }
    if input.operands.len() != operands
        || input.results.len() != results
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err(format!(
            "SM120 operation expected {operands} operands, {results} results, and no residual attributes or control flow"
        ));
    }
    Ok(())
}
fn constant_integer(ctx: &Context, source: Ptr<Operation>, index: usize) -> Result<u64, String> {
    let definition = source
        .deref(ctx)
        .get_operand(index)
        .defining_op()
        .ok_or("expected a constant integer operand")?;
    if Operation::get_opid(definition, ctx) != MirConstantOp::get_opid_static() {
        return Err("expected a constant integer operand".into());
    }
    MirConstantOp::new(definition)
        .get_attr_value(ctx)
        .ok_or("constant has no value")?
        .value()
        .to_string_decimal(false)
        .parse()
        .map_err(|_| "constant integer exceeds u64".into())
}
fn pointer(
    session: &mut TranslationSession<'_>,
    operations: &mut Vec<MlirOperation>,
    value: MlirValueUse,
    space: u32,
    location: &MlirLocation,
) -> Result<MlirValueUse, String> {
    let target = MlirType::dialect(if space == 0 {
        "!llvm.ptr".into()
    } else {
        format!("!llvm.ptr<{space}>")
    })?;
    if value.ty == target {
        return Ok(value);
    }
    let shared_source = value.ty == MlirType::dialect("!llvm.ptr")?
        || value.ty == MlirType::dialect("!llvm.ptr<3>")?;
    let descriptor_source = value.ty == MlirType::dialect("!llvm.ptr<1>")?
        || value.ty == MlirType::dialect("!llvm.ptr<4>")?;
    let cast = if space == 3 && value.ty == MlirType::Integer(32) {
        "llvm.inttoptr"
    } else if (space != 0 && shared_source) || (space == 0 && descriptor_source) {
        "llvm.addrspacecast"
    } else {
        return Err(format!(
            "unsupported pointer {:?} for SM120 address space {space}",
            value.ty
        ));
    };
    let (result, converted) = fresh(session, target);
    operations.push(operation(cast, vec![value], vec![result], location)?);
    Ok(converted)
}
fn property(op: &mut MlirOperation, name: &str, value: &str) -> Result<(), String> {
    op.properties
        .insert(name.into(), MlirAttribute::dialect(value)?);
    Ok(())
}
fn extract_results(
    operations: &mut Vec<MlirOperation>,
    aggregate: MlirValueUse,
    results: Vec<MlirResult>,
    location: &MlirLocation,
) -> Result<(), String> {
    for (index, result) in results.into_iter().enumerate() {
        let mut extract = operation(
            "llvm.extractvalue",
            vec![aggregate.clone()],
            vec![result],
            location,
        )?;
        extract.properties.insert(
            "position".into(),
            MlirAttribute::DenseI64Array(vec![index as i64]),
        );
        operations.push(extract);
    }
    Ok(())
}
fn fresh(session: &mut TranslationSession<'_>, ty: MlirType) -> (MlirResult, MlirValueUse) {
    let id = session.fresh_value();
    (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
}
fn operation(
    name: &str,
    operands: Vec<MlirValueUse>,
    results: Vec<MlirResult>,
    location: &MlirLocation,
) -> Result<MlirOperation, String> {
    let mut operation = MlirOperation::new(name)?;
    operation.operands = operands;
    operation.results = results;
    operation.location = location.clone();
    Ok(operation)
}
