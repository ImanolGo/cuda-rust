/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 element kernels for the checked Qwen decoder composition.

use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Gather dense embedding rows after the host has checked every token ID.
    #[kernel]
    pub fn embedding_bf16(
        table: &[Bf16],
        ids: &[u32],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
    ) {
        let index = thread::index_1d().get();
        if index < rows * width {
            let row = index / width;
            let column = index % width;
            let value = table[ids[row] as usize * width + column];
            // SAFETY: each thread owns one output element. The checked host
            // boundary validates the dense table extent and all token IDs.
            unsafe { output.as_mut_ptr().add(index).write(value) };
        }
    }

    /// Add two stored BF16 tensors and round their sum once to BF16.
    #[kernel]
    pub fn add_bf16(left: &[Bf16], right: &[Bf16], mut output: DisjointSlice<Bf16>, length: usize) {
        let index = thread::index_1d().get();
        if index < length {
            let value = Bf16::from_f32(left[index].to_f32() + right[index].to_f32());
            // SAFETY: each thread owns one element of the checked output range.
            unsafe { output.as_mut_ptr().add(index).write(value) };
        }
    }

    /// Evaluate SiLU in FP32 and store the rounded BF16 result.
    #[kernel]
    pub fn silu_bf16(input: &[Bf16], mut output: DisjointSlice<Bf16>, length: usize) {
        let index = thread::index_1d().get();
        if index < length {
            let value = input[index].to_f32();
            let sigmoid = if value >= 0.0 {
                let e = ctm_rs::float::ex2_approx_f32(-value * core::f32::consts::LOG2_E);
                ctm_rs::float::rcp_approx_ftz_f32(1.0 + e)
            } else {
                let e = ctm_rs::float::ex2_approx_f32(value * core::f32::consts::LOG2_E);
                e * ctm_rs::float::rcp_approx_ftz_f32(1.0 + e)
            };
            // SAFETY: the checked logical range assigns one output to each thread.
            unsafe {
                output
                    .as_mut_ptr()
                    .add(index)
                    .write(Bf16::from_f32(value * sigmoid));
            }
        }
    }

    /// Concatenate two dense storage ranges without changing their BF16 bits.
    #[kernel]
    pub fn concat_bf16(
        left: &[Bf16],
        right: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        left_length: usize,
        total_length: usize,
    ) {
        let index = thread::index_1d().get();
        if index < total_length {
            let value = if index < left_length {
                left[index]
            } else {
                right[index - left_length]
            };
            // SAFETY: the host checks both input ranges and their total length;
            // each thread writes one distinct element in the concatenated range.
            unsafe { output.as_mut_ptr().add(index).write(value) };
        }
    }

    /// Reduce codec embedding rows in FP32, rounding only the final sum to BF16.
    #[kernel]
    pub fn sum_rows_bf16(
        input: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
    ) {
        let column = thread::index_1d().get();
        if column < width {
            let mut sum = 0.0_f32;
            let mut row = 0;
            while row < rows {
                sum += input[row * width + column].to_f32();
                row += 1;
            }
            // SAFETY: each thread owns one output column. The host checks the
            // complete dense input extent; an empty reduction writes zero.
            unsafe {
                output.as_mut_ptr().add(column).write(Bf16::from_f32(sum));
            }
        }
    }
}
