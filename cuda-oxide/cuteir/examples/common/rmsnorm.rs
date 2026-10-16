/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared FP32/BF16 RMSNorm kernels with explicit BF16 rounding boundaries.

use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;
use cute_rs::tensor::partition::{StridedTensor, StridedTensorMut};

/// Four complete warps own four independent rows in every block.
pub(crate) const ROWS_PER_BLOCK: usize = 4;

/// Scalar row access supports arbitrary widths without alignment promises.
#[cuda_module]
pub mod kernels {
    use super::*;

    /// Normalize FP32 rows using one complete warp and FP32 arithmetic per row.
    #[kernel]
    pub fn rmsnorm_f32(
        input: &[f32],
        weight: &[f32],
        mut output: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        // This branch is uniform within each warp, including the final block.
        if row < rows {
            let lane = tid % 32;
            let base = row * cols;
            let values = StridedTensor::from_slice(input)
                .subview(base, cols)
                .partition::<32>(lane);
            let scales = StridedTensor::from_slice(weight)
                .subview(0, cols)
                .partition::<32>(lane);
            let squares = values.fold(0.0_f32, |sum, value| sum + value * value);
            // SAFETY: all 32 lanes of this row reach the collective. Lanes
            // with no valid column contribute zero, instead of returning.
            let total = unsafe { cute_rs::arch::warp_sum_f32(squares) };
            let inverse_rms = cute_rs::arch::rsqrt_f32(total / cols as f32 + epsilon);
            let mut destination = StridedTensorMut::from_disjoint_slice(&mut output)
                .subview(base, cols)
                .partition::<32>(lane);
            // SAFETY: the checked launch partitions rows by warp and each
            // row by lane; sources and destination own distinct allocations.
            unsafe {
                values.zip_map_to(scales, &mut destination, |value, scale| {
                    (value * inverse_rms) * scale
                });
            }
        }
    }

    /// Normalize BF16 storage, rounding normalized values before multiplying weights.
    #[kernel]
    pub fn rmsnorm_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        if row < rows {
            let lane = tid % 32;
            let base = row * cols;
            let values = StridedTensor::from_slice(input)
                .subview(base, cols)
                .partition::<32>(lane);
            let scales = StridedTensor::from_slice(weight)
                .subview(0, cols)
                .partition::<32>(lane);
            let squares = values.fold(0.0_f32, |sum, value| {
                let value = value.to_f32();
                sum + value * value
            });
            // SAFETY: each launched block contains four complete warps; the
            // row predicate is warp-uniform and column tails do not skip lanes.
            let total = unsafe { cute_rs::arch::warp_sum_f32(squares) };
            let inverse_rms = cute_rs::arch::rsqrt_f32(total / cols as f32 + epsilon);
            let mut destination = StridedTensorMut::from_disjoint_slice(&mut output)
                .subview(base, cols)
                .partition::<32>(lane);
            // SAFETY: complete warps own disjoint rows and cyclic lane
            // partitions own disjoint columns, with separate input storage.
            unsafe {
                values.zip_map_to(scales, &mut destination, |value, scale| {
                    let normalized = Bf16::from_f32(value.to_f32() * inverse_rms);
                    Bf16::from_f32(normalized.to_f32() * scale.to_f32())
                });
            }
        }
    }

    /// Keep each lane's values live across the reduction. Constant trip counts
    /// remove repeated address/control work without changing the serial FFMA
    /// order, warp XOR tree or either BF16 rounding boundary.
    #[inline(always)]
    fn rmsnorm_bf16_fixed<const ITEMS: usize>(
        input: &[Bf16],
        weight: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        divisor: usize,
        epsilon: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        if row < rows
            && rows <= input.len() / (ITEMS * 32)
            && rows <= output.len() / (ITEMS * 32)
            && weight.len() >= ITEMS * 32
        {
            let lane = tid % 32;
            let base = row * (ITEMS * 32);
            // SAFETY: the uniform extent proof bounds both rows. The cyclic
            // partition expresses the original lane + item * 32 mapping.
            let (values, scales) = unsafe {
                let values = StridedTensor::from_slice(input)
                    .subview_unchecked(base, ITEMS * 32)
                    .partition::<32>(lane)
                    .load_unchecked::<ITEMS>()
                    .map(Bf16::to_f32);
                let scales = StridedTensor::from_slice(weight)
                    .subview_unchecked(0, ITEMS * 32)
                    .partition::<32>(lane)
                    .load_unchecked::<ITEMS>()
                    .map(Bf16::to_f32);
                (values, scales)
            };
            let squares = values.fold(0.0_f32, |sum, value| sum + value * value);
            // SAFETY: full warps own rows, and the row predicate is uniform.
            // warp_sum_f32 retains XOR offsets 16, 8, 4, 2, 1.
            let total = unsafe { cute_rs::arch::warp_sum_f32(squares) };
            // Keep the divisor a runtime argument, as in the original kernel.
            // Constant power-of-two division may otherwise contract with the
            // epsilon addition and change subnormal rounding boundaries.
            let inverse_rms = cute_rs::arch::rsqrt_f32(total / divisor as f32 + epsilon);
            let normalized = values.zip_map(scales, |value, scale| {
                let normalized = Bf16::from_f32(value * inverse_rms);
                Bf16::from_f32(normalized.to_f32() * scale)
            });
            // SAFETY: the checked row extent and cyclic partition assign each
            // output one writer. The view consumes its parent writable borrow.
            unsafe {
                StridedTensorMut::from_disjoint_slice(&mut output)
                    .subview_unchecked(base, ITEMS * 32)
                    .partition::<32>(lane)
                    .store(normalized);
            }
        }
    }

    /// Head-width normalization with the original four values per lane.
    #[kernel]
    pub fn rmsnorm_bf16_128(
        input: &[Bf16],
        weight: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        rmsnorm_bf16_fixed::<4>(input, weight, output, rows, cols, epsilon);
    }

    /// Predictor-width normalization with the original 32 values per lane.
    #[kernel]
    pub fn rmsnorm_bf16_1024(
        input: &[Bf16],
        weight: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        rmsnorm_bf16_fixed::<32>(input, weight, output, rows, cols, epsilon);
    }

    /// Talker-width normalization with the original 64 values per lane.
    #[kernel]
    pub fn rmsnorm_bf16_2048(
        input: &[Bf16],
        weight: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        rmsnorm_bf16_fixed::<64>(input, weight, output, rows, cols, epsilon);
    }

    /// Copy BF16 bits through typed vector tiles, preserving even NaN payloads.
    #[kernel]
    pub fn copy_bf16(input: &[Bf16], mut output: DisjointSlice<Bf16>) {
        let tid = thread::index_1d().get();
        let input_tile = cute_rs::Tensor::from_slice(input)
            .zipped_divide::<8>()
            .slice(tid);
        let mut output_tile = cute_rs::TensorMut::from_disjoint_slice(&mut output)
            .zipped_divide::<8>()
            .slice(tid);
        if input_tile.is_full() {
            // SAFETY: device buffers are aligned, and each tile spans eight
            // adjacent two-byte elements with one writer and matching lengths.
            unsafe { output_tile.store(input_tile.load()) };
        } else {
            let mut index = input_tile.base();
            while index < input.len() {
                unsafe { output_tile.store_linear(index, input[index]) };
                index += 1;
            }
        }
    }
}
