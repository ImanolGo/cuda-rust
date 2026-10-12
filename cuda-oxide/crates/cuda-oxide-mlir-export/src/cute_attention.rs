/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Realize verified CuTe cooperative-copy and matrix-load plans for attention.
//!
//! The source copy ABI explicitly requires partial-atom source-size zero filling;
//! it is not an ordinary predicated CuTe copy (which can leave shared bytes
//! untouched). NVVM expresses that contract directly. Both operations retain the
//! verified CuTe thread/value assignment and composed shared layout, including
//! element-to-byte swizzle conversion. Commit, wait, and synchronization stay
//! caller-owned, as do the unsafe pointer, capacity, and participation promises.

use cute_layout::{
    ComposedLayout, IntTuple, Layout, OffsetUnit, Swizzle, validate_cooperative_copy_plan,
};
use dialect_cute::{
    attributes::{
        CuteComposedLayoutAttr, CuteCopyAtomAttr, CuteDivisibilityAttr, CuteLayoutAttr,
        CuteMatrixRoleAttr,
    },
    ops::{CuteCopyG2SOp, CuteLdmatrixOp},
};
use dialect_mir::types::MirFP16Type;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
    r#type::TypeHandle,
};
use pliron_mlir_export::{
    DropAttribute, MlirAttribute, MlirLocation, MlirOperation, MlirResult, MlirType, MlirValueUse,
    OperationInput, OperationTranslation, TranslationError, TranslationRegistry,
    TranslationSession,
};

pub(crate) fn register_cute_attention_pack(
    registry: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    registry.register_attribute::<CuteComposedLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<CuteCopyAtomAttr>(DropAttribute)?;
    registry.register_attribute::<CuteDivisibilityAttr>(DropAttribute)?;
    registry.register_attribute::<CuteLayoutAttr>(DropAttribute)?;
    registry.register_attribute::<CuteMatrixRoleAttr>(DropAttribute)?;
    registry.register_operation::<CuteCopyG2SOp>(CopyTranslation)?;
    registry.register_operation::<CuteLdmatrixOp>(MatrixTranslation)?;
    Ok(())
}

struct CopyTranslation;
impl OperationTranslation for CopyTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let copy = CuteCopyG2SOp::wrap(source);
        copy.verify(ctx).map_err(|e| e.to_string())?;
        let elem: TypeHandle = copy.get_attr_copy_g2s_elem(ctx).unwrap().clone().into();
        if !elem.deref(ctx).is::<MirFP16Type>() {
            return Err("CuTe attention copy supports f16 storage only".into());
        }
        input.attributes.remove("copy_g2s_elem");
        require_input(&input, 9)?;
        let atom = copy.get_attr_atom_bytes(ctx).unwrap().0;
        let plan = validate_cooperative_copy_plan(
            atom,
            &copy.get_attr_thread_layout(ctx).unwrap().0,
            &copy.get_attr_value_layout(ctx).unwrap().0,
            &copy.get_attr_tile_layout(ctx).unwrap().0,
            &copy.get_attr_smem_layout(ctx).unwrap().0,
            2,
        )
        .map_err(|e| e.to_string())?;
        let mut e = Emitter::new(session, input.location);
        let global = e.pointer(input.operands[0].clone(), 1)?;
        let shared = e.pointer(input.operands[6].clone(), 3)?;
        // The cooperative-copy contract supplies exactly the verified thread
        // IDs. Preserve that range in the scalar address calculation too.
        let thread_count = e.constant(plan.thread_count, 32)?;
        let thread = e.binary("arith.remui", input.operands[8].clone(), thread_count)?;
        let thread = e.value("arith.extui", MlirType::Integer(64), vec![thread])?;
        let tv_modes = plan.tv_layout.modes();
        let thread_cell = e.index(&tv_modes[0], thread)?;
        let tile_rows = e.c64(plan.tiler_shape[0])?;
        let tile_columns = e.c64(plan.tiler_shape[1])?;
        let row_base = e.binary("arith.muli", input.operands[4].clone(), tile_rows.clone())?;
        let column_base = e.binary(
            "arith.muli",
            input.operands[5].clone(),
            tile_columns.clone(),
        )?;
        // Reject overflowing tile origins through the bounds predicate before
        // choosing a source address. Out-of-range coordinates still zero-fill.
        let last_row_tile =
            e.binary("arith.divui", input.operands[1].clone(), tile_rows.clone())?;
        let last_column_tile = e.binary("arith.divui", input.operands[2].clone(), tile_columns)?;
        let row_tile_valid = e.compare(7, input.operands[4].clone(), last_row_tile)?; // ule
        let column_tile_valid = e.compare(7, input.operands[5].clone(), last_column_tile)?;
        let tile_valid = e.binary("arith.andi", row_tile_valid, column_tile_valid)?;
        let zero = e.c64(0)?;
        let two = e.c64(2)?;
        let values_per_atom = i64::from(atom) / 2;
        let atom_values = e.c64(values_per_atom)?;
        for value_index in (0..plan.values_per_thread).step_by(values_per_atom as usize) {
            let value_cell = tv_modes[1]
                .checked_call(&IntTuple::Leaf(value_index))
                .ok_or_else(|| "verified CuTe value assignment is out of range".to_owned())?;
            let value_cell = e.c64(value_cell)?;
            let cell = e.binary("arith.addi", thread_cell.clone(), value_cell)?;
            let row = e.binary("arith.remui", cell.clone(), tile_rows.clone())?;
            let column = e.binary("arith.divui", cell, tile_rows.clone())?;
            let shared_offset = e.composed(&plan.smem_byte_layout, row.clone(), column.clone())?;
            let destination = e.gep(shared.clone(), shared_offset, MlirType::Integer(8))?;
            let row = e.binary("arith.addi", row_base.clone(), row)?;
            let column = e.binary("arith.addi", column_base.clone(), column)?;
            let row_valid = e.compare(6, row.clone(), input.operands[1].clone())?; // ult
            let column_valid = e.compare(6, column.clone(), input.operands[2].clone())?;
            let valid = e.binary("arith.andi", row_valid, column_valid)?;
            let valid = e.binary("arith.andi", valid, tile_valid.clone())?;
            let remaining = e.binary("arith.subi", input.operands[2].clone(), column.clone())?;
            let remaining = e.binary("arith.minui", remaining, atom_values.clone())?;
            let valid_elements = e.select(valid.clone(), remaining, zero.clone())?;
            let valid_bytes = e.binary("arith.muli", valid_elements, two.clone())?;
            let valid_bytes = e.value("arith.trunci", MlirType::Integer(32), vec![valid_bytes])?;
            let row_offset = e.binary("arith.muli", row, input.operands[3].clone())?;
            let source_offset = e.binary("arith.addi", row_offset, column)?;
            let source_offset = e.select(valid, source_offset, zero.clone())?;
            let source = e.gep(
                global.clone(),
                source_offset,
                MlirType::Float(pliron_mlir_export::MlirFloatType::F16),
            )?;
            let mut op = e.operation(
                "nvvm.cp.async.shared.global",
                vec![destination, source, valid_bytes],
            )?;
            op.properties
                .insert("size".into(), integer(i128::from(atom), 32));
            op.properties.insert(
                "modifier".into(),
                MlirAttribute::dialect(if atom == 16 {
                    "#nvvm<load_cache_modifier cg>"
                } else {
                    "#nvvm<load_cache_modifier ca>"
                })?,
            );
            e.operations.push(op);
        }
        Ok(e.operations)
    }
}

struct MatrixTranslation;

/// Bound every intermediate in the byte-offset calculation, not just its final
/// value: negative strides/offsets could otherwise cancel an overflowing term.
/// The verified layout and the unsafe matrix-load API promise a hardware lane
/// (0..32) and an in-bounds selected matrix window. Thus the window coordinates,
/// their row/column expansions, and every nested-layout digit are bounded by
/// the positive shape products. This is not a check of dynamic caller inputs.
fn matrix_offset_fits_i32(layout: &ComposedLayout) -> bool {
    let limit = i64::from(i32::MAX);
    if layout.unit() != OffsetUnit::Bytes
        || layout.inner().modes().len() != 2
        || !layout
            .inner()
            .checked_size()
            .is_some_and(|size| size <= limit)
        || !(0..=limit).contains(&layout.offset())
    {
        return false;
    }
    let mut maximum = layout.offset();
    for (&extent, &stride) in layout
        .inner()
        .shape
        .leaves()
        .iter()
        .zip(&layout.inner().stride.leaves())
    {
        if !(0..=limit).contains(&stride) {
            return false;
        }
        let Some(next) = (extent - 1)
            .checked_mul(stride)
            .and_then(|term| maximum.checked_add(term))
        else {
            return false;
        };
        if next > limit {
            return false;
        }
        maximum = next;
    }
    // Keeping both fields below the sign bit bounds the mask, shifted bits,
    // and final XOR even when swizzling raises the maximum physical offset.
    let swizzle = layout.outer();
    swizzle.is_valid()
        && (0..=limit).contains(&swizzle.y_mask())
        && (0..=limit).contains(&swizzle.z_mask())
}

struct MatrixAddressPlan {
    low_mask: i64,
    high_mask: i64,
}

/// Prove a carry-free split between window-only and lane-only offsets. Nested
/// layout indexing need not distribute over coordinate addition, and a fixed
/// offset may overlap lane bits, so compare the factored result with the full
/// checked map for every legal window/lane. The existing static-validation cap
/// bounds this proof to at most one lane address per four tile elements.
fn matrix_address_plan(
    layout: &ComposedLayout,
    role: CuteMatrixRoleAttr,
) -> Option<MatrixAddressPlan> {
    if !matrix_offset_fits_i32(layout)
        || layout.inner().checked_size()? > cute_layout::validation::MAX_STATIC_VALIDATION_ELEMENTS
    {
        return None;
    }
    let modes = layout.inner().modes();
    let rows = modes[0].checked_size()?;
    let columns = modes[1].checked_size()?;
    let window_columns = if role == CuteMatrixRoleAttr::B { 8 } else { 16 };
    if rows < 16 || columns < window_columns {
        return None;
    }
    let index = |mode: usize, coordinate| modes[mode].checked_call(&IntTuple::Leaf(coordinate));
    let swizzle = layout.outer();
    let active = swizzle.y_mask() | swizzle.z_mask();
    let low_mask = (1i64 << (i64::BITS - active.leading_zeros())) - 1;
    let high_mask = i64::from(i32::MAX) ^ low_mask;
    for window_row in 0..rows / 16 {
        for window_column in 0..columns / window_columns {
            let tile_row = window_row * 16;
            let tile_column = window_column * window_columns;
            let uniform = layout.offset() + index(0, tile_row)? + index(1, tile_column)?;
            for lane in 0..32 {
                let (local_row, local_column) = if role == CuteMatrixRoleAttr::BPairKMajor {
                    (lane % 8 + lane / 16 * 8, lane / 8 % 2 * 8)
                } else {
                    (
                        lane % 16,
                        if role == CuteMatrixRoleAttr::B {
                            0
                        } else {
                            lane / 16 * 8
                        },
                    )
                };
                let local = index(0, local_row)? + index(1, local_column)?;
                if uniform & local != 0 {
                    return None;
                }
                let high = uniform & high_mask;
                let low = swizzle.apply(uniform & low_mask) ^ swizzle.apply(local);
                let coordinate = tile_row + local_row + rows * (tile_column + local_column);
                let original = layout.checked_call(&IntTuple::Leaf(coordinate))?;
                if high & low != 0 || high.checked_add(low)? != original {
                    return None;
                }
            }
        }
    }
    Some(MatrixAddressPlan {
        low_mask,
        high_mask,
    })
}

impl OperationTranslation for MatrixTranslation {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        mut input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let load = CuteLdmatrixOp::wrap(source);
        load.verify(ctx).map_err(|e| e.to_string())?;
        input.attributes.remove("ldmatrix_elem");
        require_input(&input, 5)?;
        let role = *load.get_attr_matrix_role(ctx).unwrap();
        let (count, columns, transpose) = match role {
            CuteMatrixRoleAttr::A | CuteMatrixRoleAttr::BPairKMajor => (4, 16, false),
            CuteMatrixRoleAttr::B => (2, 8, true),
            CuteMatrixRoleAttr::BPair => (4, 16, true),
        };
        let layout = load
            .get_attr_ldmatrix_smem_layout(ctx)
            .unwrap()
            .0
            .to_byte_offsets(2)
            .map_err(|e| e.to_string())?;
        let mut e = Emitter::new(session, input.location);
        let shared = e.pointer(input.operands[0].clone(), 3)?;
        let width = if matrix_offset_fits_i32(&layout) {
            32
        } else {
            64
        };
        let (window_row, window_column, lane) = if width == 32 {
            (
                e.value(
                    "arith.trunci",
                    MlirType::Integer(32),
                    vec![input.operands[1].clone()],
                )?,
                e.value(
                    "arith.trunci",
                    MlirType::Integer(32),
                    vec![input.operands[2].clone()],
                )?,
                input.operands[3].clone(),
            )
        } else {
            (
                input.operands[1].clone(),
                input.operands[2].clone(),
                e.value(
                    "arith.extui",
                    MlirType::Integer(64),
                    vec![input.operands[3].clone()],
                )?,
            )
        };
        let eight = e.constant(8, width)?;
        let two = e.constant(2, width)?;
        let sixteen = e.constant(16, width)?;
        let column_extent = e.constant(columns, width)?;
        let submatrix = e.binary("arith.divui", lane.clone(), eight.clone())?;
        let lane_row = e.binary("arith.remui", lane, eight.clone())?;
        // Each group of eight lanes supplies one 8x8 matrix to ldmatrix.x4.
        // K-major B pairs need [top-left, top-right, bottom-left, bottom-right]
        // so adjacent result registers form an MMA B operand. A fragments
        // and transposed B pairs retain their original column-first ordering.
        let (row_half_op, column_half_op) = if role == CuteMatrixRoleAttr::BPairKMajor {
            ("arith.divui", "arith.remui")
        } else {
            ("arith.remui", "arith.divui")
        };
        let row_half = e.binary(row_half_op, submatrix.clone(), two.clone())?;
        let row_half = e.binary("arith.muli", row_half, eight.clone())?;
        let local_row = e.binary("arith.addi", row_half, lane_row)?;
        let window_row = e.binary("arith.muli", window_row, sixteen)?;
        let window_column = e.binary("arith.muli", window_column, column_extent)?;
        let local_column = if count == 4 {
            let half = e.binary(column_half_op, submatrix, two)?;
            e.binary("arith.muli", half, eight)?
        } else {
            e.constant(0, width)?
        };
        let source = if let Some(plan) = matrix_address_plan(&layout, role) {
            let uniform = e.linear_pair(layout.inner(), window_row, window_column)?;
            let offset = e.constant(layout.offset(), 32)?;
            let uniform = e.binary("arith.addi", uniform, offset)?;
            let local = e.linear_pair(layout.inner(), local_row, local_column)?;
            let high_mask = e.constant(plan.high_mask, 32)?;
            let high = e.binary("arith.andi", uniform.clone(), high_mask)?;
            let low_mask = e.constant(plan.low_mask, 32)?;
            let low = e.binary("arith.andi", uniform, low_mask)?;
            let low = e.swizzle(layout.outer(), low)?;
            let local = e.swizzle(layout.outer(), local)?;
            let low = e.binary("arith.xori", low, local)?;
            // Keep the high window offset additive so constant matrix-row
            // displacements can remain GEP/immediate offsets after lowering.
            // The allocation pointer itself never enters the swizzle.
            let high = e.value("arith.extui", MlirType::Integer(64), vec![high])?;
            let low = e.value("arith.extui", MlirType::Integer(64), vec![low])?;
            let base = e.gep(shared, high, MlirType::Integer(8))?;
            e.gep(base, low, MlirType::Integer(8))?
        } else {
            let row = e.binary("arith.addi", window_row, local_row)?;
            let column = e.binary("arith.addi", window_column, local_column)?;
            let offset = e.composed(&layout, row, column)?;
            // Preserve 64-bit pointer arithmetic in both general paths.
            let offset = if width == 32 {
                e.value("arith.extui", MlirType::Integer(64), vec![offset])?
            } else {
                offset
            };
            e.gep(shared, offset, MlirType::Integer(8))?
        };
        let aggregate_type =
            MlirType::dialect(format!("!llvm.struct<({})>", vec!["i32"; count].join(", ")))?;
        let (result, aggregate) = e.fresh(aggregate_type);
        let mut load = e.operation("nvvm.ldmatrix", vec![source])?;
        load.results.push(result);
        load.properties
            .insert("num".into(), integer(count as i128, 32));
        load.properties.insert(
            "layout".into(),
            MlirAttribute::dialect(if transpose {
                "#nvvm.mma_layout<col>"
            } else {
                "#nvvm.mma_layout<row>"
            })?,
        );
        e.operations.push(load);
        for index in 0..count {
            let (result, register) = e.fresh(MlirType::Integer(32));
            let mut extract = e.operation("llvm.extractvalue", vec![aggregate.clone()])?;
            extract.results.push(result);
            extract.properties.insert(
                "position".into(),
                MlirAttribute::DenseI64Array(vec![index as i64]),
            );
            e.operations.push(extract);
            let index = e.c64(index as i64)?;
            let slot = e.gep(input.operands[4].clone(), index, MlirType::Integer(32))?;
            let mut store = e.operation("llvm.store", vec![register, slot])?;
            store.properties.insert("alignment".into(), integer(4, 64));
            e.operations.push(store);
        }
        Ok(e.operations)
    }
}

fn require_input(input: &OperationInput, operands: usize) -> Result<(), String> {
    if input.operands.len() != operands
        || !input.results.is_empty()
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err("CuTe attention operation retained unexpected structure".into());
    }
    Ok(())
}

fn integer(value: i128, width: u32) -> MlirAttribute {
    MlirAttribute::Integer {
        value,
        ty: MlirType::Integer(width),
    }
}

struct Emitter<'a, 'ctx> {
    session: &'a mut TranslationSession<'ctx>,
    location: MlirLocation,
    operations: Vec<MlirOperation>,
}
impl<'a, 'ctx> Emitter<'a, 'ctx> {
    fn new(session: &'a mut TranslationSession<'ctx>, location: MlirLocation) -> Self {
        Self {
            session,
            location,
            operations: vec![],
        }
    }
    fn fresh(&mut self, ty: MlirType) -> (MlirResult, MlirValueUse) {
        let id = self.session.fresh_value();
        (MlirResult { id, ty: ty.clone() }, MlirValueUse { id, ty })
    }
    fn operation(&self, name: &str, operands: Vec<MlirValueUse>) -> Result<MlirOperation, String> {
        let mut op = MlirOperation::new(name)?;
        op.location = self.location.clone();
        op.operands = operands;
        Ok(op)
    }
    fn value(
        &mut self,
        name: &str,
        ty: MlirType,
        operands: Vec<MlirValueUse>,
    ) -> Result<MlirValueUse, String> {
        let (result, use_) = self.fresh(ty);
        let mut op = self.operation(name, operands)?;
        op.results.push(result);
        self.operations.push(op);
        Ok(use_)
    }
    fn c64(&mut self, value: i64) -> Result<MlirValueUse, String> {
        self.constant(value, 64)
    }
    fn constant(&mut self, value: i64, width: u32) -> Result<MlirValueUse, String> {
        let (result, use_) = self.fresh(MlirType::Integer(width));
        let mut op = self.operation("arith.constant", vec![])?;
        op.results.push(result);
        op.properties
            .insert("value".into(), integer(i128::from(value), width));
        self.operations.push(op);
        Ok(use_)
    }
    fn binary(
        &mut self,
        name: &str,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        self.value(name, a.ty.clone(), vec![a, b])
    }
    fn compare(
        &mut self,
        predicate: i128,
        a: MlirValueUse,
        b: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let value = self.value("arith.cmpi", MlirType::Integer(1), vec![a, b])?;
        self.operations
            .last_mut()
            .unwrap()
            .properties
            .insert("predicate".into(), integer(predicate, 64));
        Ok(value)
    }
    fn select(
        &mut self,
        predicate: MlirValueUse,
        yes: MlirValueUse,
        no: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        self.value("arith.select", yes.ty.clone(), vec![predicate, yes, no])
    }
    fn pointer(&mut self, pointer: MlirValueUse, space: u32) -> Result<MlirValueUse, String> {
        let ty = MlirType::dialect(format!("!llvm.ptr<{space}>"))?;
        if pointer.ty == ty {
            Ok(pointer)
        } else {
            self.value("llvm.addrspacecast", ty, vec![pointer])
        }
    }
    fn gep(
        &mut self,
        base: MlirValueUse,
        index: MlirValueUse,
        element: MlirType,
    ) -> Result<MlirValueUse, String> {
        let value = self.value("llvm.getelementptr", base.ty.clone(), vec![base, index])?;
        let op = self.operations.last_mut().unwrap();
        op.properties
            .insert("elem_type".into(), MlirAttribute::Type(element));
        op.properties.insert(
            "rawConstantIndices".into(),
            MlirAttribute::DenseI32Array(vec![i32::MIN]),
        );
        op.properties.insert("noWrapFlags".into(), integer(0, 32));
        Ok(value)
    }
    /// A flat CuTe coordinate expands in column-major order through its modes.
    fn index(
        &mut self,
        layout: &Layout,
        mut coordinate: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let sizes = layout.shape.leaves();
        let strides = layout.stride.leaves();
        let MlirType::Integer(width) = coordinate.ty else {
            return Err("CuTe layout coordinate must be an integer".into());
        };
        let mut offset = self.constant(0, width)?;
        for (i, (&size, &stride)) in sizes.iter().zip(&strides).enumerate() {
            let size = self.constant(size, width)?;
            let digit = if i + 1 == sizes.len() {
                coordinate.clone()
            } else {
                self.binary("arith.remui", coordinate.clone(), size.clone())?
            };
            let stride = self.constant(stride, width)?;
            let term = self.binary("arith.muli", digit, stride)?;
            offset = self.binary("arith.addi", offset, term)?;
            if i + 1 != sizes.len() {
                coordinate = self.binary("arith.divui", coordinate, size)?;
            }
        }
        Ok(offset)
    }
    fn linear_pair(
        &mut self,
        layout: &Layout,
        row: MlirValueUse,
        column: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let modes = layout.modes();
        if modes.len() != 2 {
            return Err("CuTe attention shared layout requires two modes".into());
        }
        let row = self.index(&modes[0], row)?;
        let column = self.index(&modes[1], column)?;
        self.binary("arith.addi", row, column)
    }
    fn composed(
        &mut self,
        layout: &ComposedLayout,
        row: MlirValueUse,
        column: MlirValueUse,
    ) -> Result<MlirValueUse, String> {
        let MlirType::Integer(width) = row.ty else {
            return Err("CuTe layout coordinate must be an integer".into());
        };
        let plain = self.linear_pair(layout.inner(), row, column)?;
        let offset = self.constant(layout.offset(), width)?;
        let plain = self.binary("arith.addi", plain, offset)?;
        self.swizzle(layout.outer(), plain)
    }
    fn swizzle(&mut self, swizzle: Swizzle, plain: MlirValueUse) -> Result<MlirValueUse, String> {
        let MlirType::Integer(width) = plain.ty else {
            return Err("CuTe swizzle offset must be an integer".into());
        };
        if swizzle.bits == 0 {
            return Ok(plain);
        }
        let mask = self.constant(swizzle.y_mask(), width)?;
        let bits = self.binary("arith.andi", plain.clone(), mask)?;
        let shift = self.constant(i64::from(swizzle.shift.unsigned_abs()), width)?;
        let moved = self.binary(
            if swizzle.shift >= 0 {
                "arith.shrui"
            } else {
                "arith.shli"
            },
            bits,
            shift,
        )?;
        self.binary("arith.xori", plain, moved)
    }
}

#[cfg(test)]
mod tests {
    use crate::{CutlassFullCuteMlir22, MlirConsumerProfile};
    use cute_layout::{ComposedLayout, Layout, OffsetUnit, Swizzle};
    use dialect_cute::{
        attributes::CuteMatrixRoleAttr,
        ops::{CuteCopyG2SOp, CuteLdmatrixOp},
    };
    use dialect_mir::{
        ops::{MirFuncOp, MirReturnOp},
        types::{MirFP16Type, MirPtrType, address_space},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr,
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::TypeHandle,
    };
    use pliron_mlir_export::render_module;

    fn shared_layout() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::new(3, 3, 3),
            0,
            "(64,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    fn append<O: Op>(ctx: &Context, block: Ptr<BasicBlock>, op: O) {
        op.get_operation().insert_at_back(block, ctx);
    }

    fn module(ctx: &mut Context, copy: bool, role: CuteMatrixRoleAttr) -> ModuleOp {
        copy_module(ctx, copy, role, 64)
    }

    fn copy_module(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
    ) -> ModuleOp {
        module_with_layout(ctx, copy, role, rows, shared_layout())
    }

    fn module_with_layout(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
        layout: ComposedLayout,
    ) -> ModuleOp {
        module_with_copy_plan(ctx, copy, role, rows, layout, None)
    }

    type CopyPlanFixture = (u32, Layout, Layout, Layout, ComposedLayout);

    fn module_with_copy_plan(
        ctx: &mut Context,
        copy: bool,
        role: CuteMatrixRoleAttr,
        rows: usize,
        layout: ComposedLayout,
        copy_plan: Option<CopyPlanFixture>,
    ) -> ModuleOp {
        dialect_mir::register(ctx);
        dialect_cute::register(ctx);
        let module = ModuleOp::new(ctx, Identifier::try_from("attention_copy").unwrap());
        let f16_: TypeHandle = MirFP16Type::get(ctx).into();
        let i32_: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let i64_: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let gmem: TypeHandle = MirPtrType::get(ctx, f16_, false, address_space::GENERIC).into();
        let smem: TypeHandle = MirPtrType::get(ctx, f16_, true, address_space::GENERIC).into();
        let slot: TypeHandle = MirPtrType::get(ctx, i32_, true, address_space::GENERIC).into();
        let types = if copy {
            vec![gmem, i64_, i64_, i64_, i64_, i64_, smem, i64_, i32_]
        } else {
            vec![smem, i64_, i64_, i32_, slot]
        };
        let ty = FunctionType::get(ctx, types.clone(), vec![]);
        let op = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(ctx, op, TypeAttr::new(ty.into()));
        func.set_symbol_name(ctx, Identifier::try_from("attention_copy").unwrap());
        module.append_operation(ctx, op, 0);
        let block = BasicBlock::new(ctx, None, types);
        block.insert_at_back(op.deref(ctx).get_region(0), ctx);
        let args = block.deref(ctx).arguments().collect::<Vec<_>>();
        if copy {
            let (atom, thread, value, tile, shared) = copy_plan.unwrap_or_else(|| {
                (
                    16,
                    "(16,8):(8,1)".parse().unwrap(),
                    format!("({},8):(8,1)", rows / 16).parse().unwrap(),
                    format!("({rows},64):(64,1)").parse().unwrap(),
                    ComposedLayout::new(
                        Swizzle::new(3, 3, 3),
                        0,
                        format!("({rows},64):(64,1)").parse().unwrap(),
                        OffsetUnit::Elements,
                    )
                    .unwrap(),
                )
            });
            let copy = CuteCopyG2SOp::new(
                ctx,
                args.try_into().unwrap(),
                atom,
                thread,
                value,
                tile,
                shared,
                8,
                f16_,
            );
            append(ctx, block, copy);
        } else {
            let load = CuteLdmatrixOp::new(ctx, args.try_into().unwrap(), role, layout, f16_);
            append(ctx, block, load);
        }
        let ret = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        append(ctx, block, MirReturnOp::new(ret));
        module
    }

    #[test]
    fn attention_copy_repeats_four_atoms_with_dynamic_zero_fill() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, true, CuteMatrixRoleAttr::A);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let text = render_module(&target);
        assert_eq!(
            text.matches("\"nvvm.cp.async.shared.global\"").count(),
            4,
            "{text}"
        );
        assert!(text.contains("#nvvm<load_cache_modifier cg>"), "{text}");
        assert!(text.contains("arith.select"), "{text}");
        assert!(text.contains("arith.xori"), "{text}");
        assert!(!text.contains("cp.async.commit"), "{text}");
        assert!(!text.contains("cp.async.wait"), "{text}");
        assert!(!text.contains("cute.copy_g2s"), "{text}");
    }

    #[test]
    fn attention_matrix_roles_select_normal_and_transposed_loads() {
        for (role, number, layout) in [
            (CuteMatrixRoleAttr::A, 4, "row"),
            (CuteMatrixRoleAttr::B, 2, "col"),
            (CuteMatrixRoleAttr::BPair, 4, "col"),
            (CuteMatrixRoleAttr::BPairKMajor, 4, "row"),
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let text = render_module(&target);
            assert!(text.contains(&format!("num = {number} : i32")), "{text}");
            assert!(
                text.contains(&format!("#nvvm.mma_layout<{layout}>")),
                "{text}"
            );
            assert_eq!(text.matches("\"llvm.store\"").count(), number, "{text}");
            assert!(!text.contains("cute.ldmatrix"), "{text}");
        }
    }

    fn function_block(op: &pliron_mlir_export::MlirOperation) -> &pliron_mlir_export::MlirBlock {
        if op.name == "func.func" {
            return &op.regions[0].blocks[0];
        }
        op.regions
            .iter()
            .flat_map(|r| &r.blocks)
            .flat_map(|b| &b.operations)
            .find_map(|op| {
                fn has_function(op: &pliron_mlir_export::MlirOperation) -> bool {
                    op.name == "func.func"
                        || op
                            .regions
                            .iter()
                            .flat_map(|r| &r.blocks)
                            .flat_map(|b| &b.operations)
                            .any(has_function)
                }
                has_function(op).then(|| function_block(op))
            })
            .expect("translated function")
    }

    // Execute emitted scalar MLIR, observing the addresses and source byte
    // counts handed to hardware. This checks the actual lowering, not a second
    // copy of its address formula. No kernel or GPU runtime is involved.
    fn memory_arguments(block: &pliron_mlir_export::MlirBlock, args: &[u64]) -> Vec<Vec<u64>> {
        use pliron_mlir_export::{MlirAttribute as A, MlirFloatType, MlirType as T};
        let mut values = block
            .arguments
            .iter()
            .zip(args)
            .map(|(arg, value)| (arg.id, *value))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut memory = vec![];
        for op in &block.operations {
            let a = op
                .operands
                .iter()
                .map(|v| values[&v.id])
                .collect::<Vec<_>>();
            let property = |name: &str| match op.properties[name] {
                A::Integer { value, .. } => value as u64,
                _ => panic!("integer property {name}"),
            };
            let result = match op.name.as_str() {
                "arith.constant" => property("value"),
                "arith.extui" | "llvm.addrspacecast" => a[0],
                "arith.trunci" => u64::from(a[0] as u32),
                "arith.addi" => a[0].wrapping_add(a[1]),
                "arith.subi" => a[0].wrapping_sub(a[1]),
                "arith.muli" => a[0].wrapping_mul(a[1]),
                "arith.divui" => a[0] / a[1],
                "arith.remui" => a[0] % a[1],
                "arith.minui" => a[0].min(a[1]),
                "arith.andi" => a[0] & a[1],
                "arith.xori" => a[0] ^ a[1],
                "arith.shrui" => a[0] >> a[1],
                "arith.shli" => a[0] << a[1],
                "arith.cmpi" => match property("predicate") {
                    6 => u64::from(a[0] < a[1]),
                    7 => u64::from(a[0] <= a[1]),
                    p => panic!("unexpected predicate {p}"),
                },
                "arith.select" => {
                    if a[0] != 0 {
                        a[1]
                    } else {
                        a[2]
                    }
                }
                "llvm.getelementptr" => {
                    let width = match op.properties["elem_type"] {
                        A::Type(T::Integer(8)) => 1,
                        A::Type(T::Integer(32)) => 4,
                        A::Type(T::Float(MlirFloatType::F16)) => 2,
                        ref ty => panic!("unexpected GEP element {ty:?}"),
                    };
                    a[0].wrapping_add(a[1].wrapping_mul(width))
                }
                "nvvm.cp.async.shared.global" => {
                    memory.push(a);
                    continue;
                }
                "nvvm.ldmatrix" => {
                    memory.push(a);
                    break;
                }
                "func.return" => break,
                name => panic!("unexpected operation {name}"),
            };
            // Arithmetic has the result's bit width, including wraparound.
            // Pointers retain all 64 bits even when their offsets were i32.
            let result = match op.results[0].ty {
                T::Integer(width) if width < 64 => result & ((1u64 << width) - 1),
                _ => result,
            };
            values.insert(op.results[0].id, result);
        }
        memory
    }

    #[test]
    fn attention_copy_matches_thread_layout_and_zero_fills_edge_bytes() {
        let mut ctx = Context::new();
        let source = copy_module(&mut ctx, true, CuteMatrixRoleAttr::A, 16);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let block = function_block(&target.root);
        for (rows, columns, tile_row, tile_column) in [
            (64, 64, 0, 0),
            (19, 61, 1, 0),
            (19, 67, 1, 1),
            (0, 0, 0, 0),
            (19, 61, 2, 1),
            (64, 64, u64::MAX, u64::MAX),
        ] {
            for thread in 0..128 {
                let copies = memory_arguments(
                    block,
                    &[
                        0x10000,
                        rows,
                        columns,
                        128,
                        tile_row,
                        tile_column,
                        0x20000,
                        1024,
                        thread,
                    ],
                );
                assert_eq!(copies.len(), 1);
                let row = thread / 8;
                let column = thread % 8 * 8;
                let plain_bytes = (row * 64 + column) * 2;
                let expected_shared = 0x20000 + (plain_bytes ^ ((plain_bytes & 0x380) >> 3));
                let global_row = u128::from(tile_row) * 16 + u128::from(row);
                let global_column = u128::from(tile_column) * 64 + u128::from(column);
                let in_bounds =
                    global_row < u128::from(rows) && global_column < u128::from(columns);
                let expected_bytes = if in_bounds {
                    (columns - global_column as u64).min(8) * 2
                } else {
                    0
                };
                let expected_global = if in_bounds {
                    0x10000 + (global_row as u64 * 128 + global_column as u64) * 2
                } else {
                    0x10000
                };
                assert_eq!(
                    copies[0],
                    vec![expected_shared, expected_global, expected_bytes],
                    "thread={thread}, rows={rows}, columns={columns}, tile=({tile_row},{tile_column})"
                );
            }
        }
    }

    #[test]
    fn attention_copy_bounded_threads_preserve_all_atoms_and_u64_bounds() {
        // A mask instead of remainder corrupts the six-thread fixtures; a
        // warp-sized bound corrupts IDs32..1023. Each valid thread must keep
        // its three atoms, including partial source sizes and zero-fill.
        for (thread_rows, thread_columns, column_first) in [
            (1u64, 1u64, false),
            (6, 1, false),
            (8, 1, false),
            (16, 8, false),
            (1024, 1, false),
            (2, 3, true),
        ] {
            for atom_bytes in [4u32, 8, 16] {
                let values_per_atom = u64::from(atom_bytes) / 2;
                let values_per_thread = 3 * values_per_atom;
                let tile_columns = thread_columns * values_per_thread;
                let thread: Layout = if column_first {
                    format!("({thread_rows},{thread_columns}):(1,{thread_rows})")
                } else {
                    format!("({thread_rows},{thread_columns}):({thread_columns},1)")
                }
                .parse()
                .unwrap();
                let value: Layout = format!("(1,{values_per_thread}):(0,1)").parse().unwrap();
                let tile: Layout = format!("({thread_rows},{tile_columns}):({tile_columns},1)")
                    .parse()
                    .unwrap();
                let shared = ComposedLayout::new(
                    Swizzle::new(3, 3, 3),
                    8,
                    tile.clone(),
                    OffsetUnit::Elements,
                )
                .unwrap();
                let shared_bytes = shared.to_byte_offsets(2).unwrap();
                let mut ctx = Context::new();
                let source = module_with_copy_plan(
                    &mut ctx,
                    true,
                    CuteMatrixRoleAttr::A,
                    0,
                    shared_layout(),
                    Some((atom_bytes, thread, value, tile, shared)),
                );
                let target = CutlassFullCuteMlir22::new("sm_120a")
                    .unwrap()
                    .translate_module(&ctx, &source)
                    .unwrap();
                let block = function_block(&target.root);
                let cp_ops = block
                    .operations
                    .iter()
                    .filter(|op| op.name == "nvvm.cp.async.shared.global")
                    .collect::<Vec<_>>();
                assert_eq!(cp_ops.len(), 3);
                for op in cp_ops {
                    assert_eq!(
                        op.properties["size"],
                        super::integer(i128::from(atom_bytes), 32)
                    );
                }
                for (rows, columns, tile_row, tile_column) in [
                    (thread_rows, tile_columns, 0, 0),
                    (thread_rows - 1, tile_columns - 1, 0, 0),
                    (thread_rows + 1, tile_columns + values_per_atom - 1, 1, 1),
                    (0, 0, 0, 0),
                    (thread_rows, tile_columns, u64::MAX, u64::MAX),
                    (
                        (1u64 << 32) + thread_rows,
                        tile_columns,
                        (1u64 << 32) / thread_rows,
                        0,
                    ),
                    (
                        thread_rows,
                        (1u64 << 32) + tile_columns,
                        0,
                        (1u64 << 32) / tile_columns,
                    ),
                ] {
                    let leading_dim = columns.div_ceil(8) * 8;
                    for thread in 0..thread_rows * thread_columns {
                        // Independent inversion of the two fixture thread maps.
                        let (local_row, column_group) = if column_first {
                            (thread % thread_rows, thread / thread_rows)
                        } else {
                            (thread / thread_columns, thread % thread_columns)
                        };
                        let copies = memory_arguments(
                            block,
                            &[
                                0x1234_0001_0000,
                                rows,
                                columns,
                                leading_dim,
                                tile_row,
                                tile_column,
                                0x5678_0002_0000,
                                1 << 20,
                                thread,
                            ],
                        );
                        assert_eq!(copies.len(), 3);
                        for (atom, copy) in copies.iter().enumerate() {
                            let local_column =
                                column_group * values_per_thread + atom as u64 * values_per_atom;
                            let destination = shared_bytes
                                .checked_call(&cute_layout::IntTuple::Leaf(
                                    (local_row + thread_rows * local_column) as i64,
                                ))
                                .unwrap() as u64;
                            let global_row = u128::from(tile_row) * u128::from(thread_rows)
                                + u128::from(local_row);
                            let global_column = u128::from(tile_column) * u128::from(tile_columns)
                                + u128::from(local_column);
                            let valid = global_row < u128::from(rows)
                                && global_column < u128::from(columns);
                            let valid_bytes = if valid {
                                (columns - global_column as u64).min(values_per_atom) * 2
                            } else {
                                0
                            };
                            let source_offset = if valid {
                                (global_row as u64 * leading_dim + global_column as u64) * 2
                            } else {
                                0
                            };
                            assert_eq!(
                                copy,
                                &[
                                    0x5678_0002_0000 + destination,
                                    0x1234_0001_0000 + source_offset,
                                    valid_bytes
                                ],
                                "threads=({thread_rows},{thread_columns}), column_first={column_first}, atom_bytes={atom_bytes}, thread={thread}, atom={atom}, shape=({rows},{columns}), tile=({tile_row},{tile_column})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn attention_matrix_lane_addresses_follow_byte_swizzle() {
        for role in [
            CuteMatrixRoleAttr::A,
            CuteMatrixRoleAttr::B,
            CuteMatrixRoleAttr::BPair,
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let block = function_block(&target.root);
            for lane in 0..32 {
                let addresses = memory_arguments(block, &[0x20000, 1, 1, lane, 0x30000]);
                let row = 16 + lane % 16;
                let column = if role != CuteMatrixRoleAttr::B {
                    16 + lane / 16 * 8
                } else {
                    8
                };
                let plain_bytes = (row * 64 + column) * 2;
                assert_eq!(
                    addresses[0],
                    [0x20000 + (plain_bytes ^ ((plain_bytes & 0x380) >> 3))]
                );
                assert_eq!(addresses[0][0] % 16, 0);
            }
        }
    }

    #[test]
    fn attention_matrix_uses_i32_for_proven_shared_addresses() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, false, CuteMatrixRoleAttr::A);
        let target = CutlassFullCuteMlir22::new("sm_120a")
            .unwrap()
            .translate_module(&ctx, &source)
            .unwrap();
        let block = function_block(&target.root);
        for op in block
            .operations
            .iter()
            .take_while(|op| op.name != "nvvm.ldmatrix")
            .filter(|op| {
                matches!(
                    op.name.as_str(),
                    "arith.addi"
                        | "arith.muli"
                        | "arith.divui"
                        | "arith.remui"
                        | "arith.andi"
                        | "arith.xori"
                )
            })
        {
            assert_eq!(
                op.results[0].ty,
                pliron_mlir_export::MlirType::Integer(32),
                "{}",
                op.name
            );
        }
        let source_gep = block
            .operations
            .iter()
            .find(|op| op.name == "llvm.getelementptr")
            .unwrap();
        assert_eq!(
            source_gep.operands[1].ty,
            pliron_mlir_export::MlirType::Integer(64)
        );
    }

    #[test]
    fn attention_matrix_factors_window_and_lane_pointer_offsets() {
        for role in [
            CuteMatrixRoleAttr::A,
            CuteMatrixRoleAttr::B,
            CuteMatrixRoleAttr::BPair,
            CuteMatrixRoleAttr::BPairKMajor,
        ] {
            let mut ctx = Context::new();
            let source = module(&mut ctx, false, role);
            let target = CutlassFullCuteMlir22::new("sm_120a")
                .unwrap()
                .translate_module(&ctx, &source)
                .unwrap();
            let block = function_block(&target.root);
            let geps = block
                .operations
                .iter()
                .take_while(|op| op.name != "nvvm.ldmatrix")
                .filter(|op| op.name == "llvm.getelementptr")
                .collect::<Vec<_>>();
            assert_eq!(geps.len(), 2, "keep the uniform high offset additive");
            assert_eq!(geps[1].operands[0].id, geps[0].results[0].id);
            let mut lane_dependent = std::collections::BTreeSet::from([block.arguments[3].id]);
            for op in block
                .operations
                .iter()
                .take_while(|op| op.name != "nvvm.ldmatrix")
            {
                if op.operands.iter().any(|v| lane_dependent.contains(&v.id)) {
                    lane_dependent.extend(op.results.iter().map(|v| v.id));
                }
            }
            assert!(!lane_dependent.contains(&geps[0].operands[1].id));
            assert!(lane_dependent.contains(&geps[1].operands[1].id));
        }
    }

    #[test]
    fn attention_matrix_i32_proof_checks_each_bound_before_narrowing() {
        let max = i64::from(i32::MAX);
        for (inner, offset, expected) in [
            ("(16,16):(32,2)".to_owned(), max - 510, true),
            ("(16,16):(32,2)".to_owned(), max - 509, false),
            ("(16,16):(32,2)".to_owned(), -1, false),
            ("(16,16):(-32,2)".to_owned(), 480, false),
            (format!("(1,1):({max},0)"), 0, true),
            (format!("(1,1):({},0)", max + 1), 0, false),
            (format!("({max},1):(0,0)"), 0, true),
            (format!("({},1):(0,0)", max + 1), 0, false),
            ("(65536,65536):(0,0)".to_owned(), 0, false),
            (format!("({},2):(0,0)", i64::MAX), 0, false),
            ("(0,16):(32,2)".to_owned(), 0, false),
            ("(-16,16):(32,2)".to_owned(), 0, false),
            ("16:2".to_owned(), 0, false),
            ("((4,4),(8,2)):((32,128),(2,16))".to_owned(), 16, true),
        ] {
            let layout = ComposedLayout::new(
                Swizzle::IDENTITY,
                offset,
                inner.parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert_eq!(
                super::matrix_offset_fits_i32(&layout),
                expected,
                "{layout:?}"
            );
        }
        for (swizzle, expected) in [
            (Swizzle::new(1, 29, 1), true),
            (Swizzle::new(1, 29, -1), true),
            (Swizzle::new(1, 30, 1), false),
            (Swizzle::new(1, 30, -1), false),
            (Swizzle::new(0, 32, 0), true),
        ] {
            let layout = ComposedLayout::new(
                swizzle,
                0,
                "(16,16):(32,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert_eq!(
                super::matrix_offset_fits_i32(&layout),
                expected,
                "{layout:?}"
            );
        }
        assert!(!super::matrix_offset_fits_i32(&shared_layout()));
    }

    #[test]
    fn attention_matrix_addresses_match_checked_layout_for_every_window_and_lane() {
        use cute_layout::IntTuple;
        let layouts = [
            shared_layout(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                16,
                "(32,32):(32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                -16,
                "(32,32):(32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                1024,
                "(32,32):(-32,1)".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(2, 3, -2),
                32,
                "((4,8),(8,4)):((32,128),(1,8))".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                64,
                "((4,8),(8,4)):((32,128),(1,-8))".parse().unwrap(),
                OffsetUnit::Elements,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::IDENTITY,
                1i64 << 32,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(1, 30, 1),
                0,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(1, 30, -1),
                0,
                "(32,32):(64,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                8192,
                "((8,8),64):((128,1024),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                8192,
                "((8,8),64):((1024,128),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, -3),
                0,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                16,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                0,
                "((3,16),64):((128,512),2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
            ComposedLayout::new(
                Swizzle::new(3, 4, 3),
                i64::from(i32::MAX) - 8191,
                "(64,64):(128,2)".parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap(),
        ];
        for (layout_index, layout) in layouts.into_iter().enumerate() {
            let bytes = layout.to_byte_offsets(2).unwrap();
            let modes = layout.inner().modes();
            let rows = modes[0].checked_size().unwrap() as u64;
            let columns = modes[1].checked_size().unwrap() as u64;
            for role in [
                CuteMatrixRoleAttr::A,
                CuteMatrixRoleAttr::B,
                CuteMatrixRoleAttr::BPair,
                CuteMatrixRoleAttr::BPairKMajor,
            ] {
                let mut ctx = Context::new();
                let source = module_with_layout(&mut ctx, false, role, 0, layout.clone());
                let target = CutlassFullCuteMlir22::new("sm_120a")
                    .unwrap()
                    .translate_module(&ctx, &source)
                    .unwrap();
                let block = function_block(&target.root);
                let source_geps = block
                    .operations
                    .iter()
                    .take_while(|op| op.name != "nvvm.ldmatrix")
                    .filter(|op| op.name == "llvm.getelementptr")
                    .count();
                let expected_geps = if [0, 9, 10, 11, 14].contains(&layout_index) {
                    2
                } else {
                    1
                };
                assert_eq!(
                    source_geps, expected_geps,
                    "factoring selection: {layout:?}, {role:?}"
                );
                let expected_width = if [0, 1, 4].contains(&layout_index) || layout_index >= 9 {
                    32
                } else {
                    64
                };
                for op in block
                    .operations
                    .iter()
                    .take_while(|op| op.name != "nvvm.ldmatrix")
                    .filter(|op| {
                        matches!(
                            op.name.as_str(),
                            "arith.addi"
                                | "arith.muli"
                                | "arith.divui"
                                | "arith.remui"
                                | "arith.andi"
                                | "arith.xori"
                        )
                    })
                {
                    assert_eq!(
                        op.results[0].ty,
                        pliron_mlir_export::MlirType::Integer(expected_width),
                        "layout={layout:?}, op={}",
                        op.name
                    );
                }
                let tile_columns = if role == CuteMatrixRoleAttr::B { 8 } else { 16 };
                // Hardware matrix order, independent of the emitter's arithmetic.
                let quadrants = if role == CuteMatrixRoleAttr::BPairKMajor {
                    [(0, 0), (0, 8), (8, 0), (8, 8)]
                } else if role == CuteMatrixRoleAttr::B {
                    [(0, 0), (8, 0), (0, 0), (8, 0)]
                } else {
                    [(0, 0), (8, 0), (0, 8), (8, 8)]
                };
                for window_row in 0..rows / 16 {
                    for window_column in 0..columns / tile_columns {
                        for lane in 0..32 {
                            let (row_half, column_half) = quadrants[lane as usize / 8];
                            let row = window_row * 16 + row_half + lane % 8;
                            let column = window_column * tile_columns + column_half;
                            let offset = bytes
                                .checked_call(&IntTuple::Leaf((row + rows * column) as i64))
                                .unwrap();
                            let base = 0x1234_0002_0000u64;
                            let actual = memory_arguments(
                                block,
                                &[base, window_row, window_column, lane, 0x30000],
                            );
                            assert_eq!(
                                actual,
                                [vec![base.wrapping_add(offset as u64)]],
                                "layout={layout:?}, role={role:?}, window=({window_row},{window_column}), lane={lane}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn attention_matrix_factoring_caps_proof_and_requires_a_complete_window() {
        for inner in ["(1024,2048):(4096,2)", "(8,64):(128,2)", "(16,8):(16,2)"] {
            let layout = ComposedLayout::new(
                Swizzle::IDENTITY,
                0,
                inner.parse().unwrap(),
                OffsetUnit::Bytes,
            )
            .unwrap();
            assert!(super::matrix_offset_fits_i32(&layout));
            assert!(
                super::matrix_address_plan(&layout, CuteMatrixRoleAttr::A).is_none(),
                "{inner}"
            );
        }
    }

    #[test]
    fn attention_k_major_b_pair_loads_contiguous_mma_register_pairs() {
        let mut ctx = Context::new();
        let source = module(&mut ctx, false, CuteMatrixRoleAttr::BPairKMajor);
        let profile = CutlassFullCuteMlir22::new("sm_120a").unwrap();
        let target = profile.translate_module(&ctx, &source).unwrap();
        let new_block = function_block(&target.root);
        let old_source = module(&mut ctx, false, CuteMatrixRoleAttr::A);
        let old_target = profile.translate_module(&ctx, &old_source).unwrap();
        let old_block = function_block(&old_target.root);

        // ldmatrix.x4 gets one row address from each lane: groups of eight
        // lanes supply the four 8x8 submatrices. For K-major B, adjacent
        // registers must cover K=0..7 and K=8..15 at the same N coordinate.
        let quadrants = [(0, 0), (0, 8), (8, 0), (8, 8)];
        for window_row in 0..4 {
            for window_column in 0..4 {
                for (matrix, (row_half, column_half)) in quadrants.into_iter().enumerate() {
                    for matrix_row in 0..8 {
                        let lane = matrix as u64 * 8 + matrix_row;
                        let args = [0x20000, window_row, window_column, lane, 0x30000];
                        let addresses = memory_arguments(new_block, &args);
                        assert_eq!(addresses.len(), 1);
                        let row = window_row * 16 + row_half + matrix_row;
                        let column = window_column * 16 + column_half;
                        let bytes = (row * 64 + column) * 2;
                        let expected = 0x20000 + (bytes ^ ((bytes & 0x380) >> 3));
                        assert_eq!(
                            addresses[0],
                            [expected],
                            "window=({window_row},{window_column}), lane={lane}"
                        );
                        assert_eq!(expected % 16, 0);

                        // Independent compatibility check: the hardware's new
                        // register order equals the existing A-to-B permutation.
                        let old_matrix = [0, 2, 1, 3][matrix];
                        let mut old_args = args;
                        old_args[3] = old_matrix * 8 + matrix_row;
                        assert_eq!(addresses, memory_arguments(old_block, &old_args));
                    }
                }
            }
        }
    }
}
