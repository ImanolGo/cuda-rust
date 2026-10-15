/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared FP32/BF16 RMSNorm kernels with explicit BF16 rounding boundaries.

use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;

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
            let mut squares = 0.0_f32;
            let mut col = lane;
            while col < cols {
                let value = input[base + col];
                squares += value * value;
                col += 32;
            }
            // SAFETY: all 32 lanes of this row reach the collective. Lanes
            // with no valid column contribute zero, instead of returning.
            let total = unsafe { cute_rs::arch::warp_sum_f32(squares) };
            let inverse_rms = ctm_rs::float::rsqrt_approx_f32(total / cols as f32 + epsilon);
            col = lane;
            while col < cols {
                let value = (input[base + col] * inverse_rms) * weight[col];
                // SAFETY: the checked launch assigns each (row,col) to exactly
                // one lane and all matrix/weight allocations have exact sizes.
                unsafe { output.as_mut_ptr().add(base + col).write(value) };
                col += 32;
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
            let mut squares = 0.0_f32;
            let mut col = lane;
            while col < cols {
                let value = input[base + col].to_f32();
                squares += value * value;
                col += 32;
            }
            // SAFETY: each launched block contains four complete warps; the
            // row predicate is warp-uniform and column tails do not skip lanes.
            let total = unsafe { cute_rs::arch::warp_sum_f32(squares) };
            let inverse_rms = ctm_rs::float::rsqrt_approx_f32(total / cols as f32 + epsilon);
            col = lane;
            while col < cols {
                let normalized = Bf16::from_f32(input[base + col].to_f32() * inverse_rms);
                let value = Bf16::from_f32(normalized.to_f32() * weight[col].to_f32());
                // SAFETY: distinct lanes own disjoint output columns.
                unsafe { output.as_mut_ptr().add(base + col).write(value) };
                col += 32;
            }
        }
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
