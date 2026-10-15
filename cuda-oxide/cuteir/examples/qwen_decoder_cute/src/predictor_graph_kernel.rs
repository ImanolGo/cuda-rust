/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Scheduling helpers for the private, fixed-shape predictor region.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn reset_status(mut status: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            // SAFETY: the plan owns one status element and launches one writer.
            unsafe { status.as_mut_ptr().write(0) };
        }
    }

    /// Copy an explicitly checked dense subrange without changing BF16 bits.
    #[kernel]
    pub fn copy_range(
        input: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        input_start: usize,
        output_start: usize,
        count: usize,
    ) {
        let index = thread::index_1d().get();
        if index < count {
            // SAFETY: private host schedule checks both ranges and distinct
            // owners. Every logical output has one writer.
            unsafe {
                output
                    .as_mut_ptr()
                    .add(output_start + index)
                    .write(input[input_start + index]);
            }
        }
    }

    /// Full-warp argmax with first-index ties and sticky nonfinite rejection.
    #[kernel]
    pub fn greedy(
        logits: &[Bf16],
        mut codes: DisjointSlice<u32>,
        mut status: DisjointSlice<u32>,
        step: usize,
    ) {
        let lane = thread::threadIdx_x();
        let mut best = f32::NEG_INFINITY;
        let mut best_index = u32::MAX;
        let mut valid = true;
        let mut index = lane;
        while index < 2048 {
            let bits = logits[index as usize].to_bits();
            let value = logits[index as usize].to_f32();
            valid = valid && bits & 0x7f80 != 0x7f80;
            if value > best || (value == best && index < best_index) {
                best = value;
                best_index = index;
            }
            index += 32;
        }
        let mut offset = 16;
        while offset != 0 {
            let other = ctm_rs::warp::shuffle_xor_f32(best, offset);
            // SAFETY: the host launches exactly one full warp. All lanes run
            // every collective, including those that observed invalid logits.
            let other_index =
                unsafe { ctm_rs::warp::shuffle_sync(u32::MAX, best_index, lane ^ offset) };
            if other > best || (other == best && other_index < best_index) {
                best = other;
                best_index = other_index;
            }
            offset /= 2;
        }
        // SAFETY: every lane participates in the full-warp vote.
        let valid = unsafe { ctm_rs::warp::all(valid) };
        if lane == 0 {
            // Invalid frames still select a bounded fallback so their later
            // embedding reads remain safe; the host rejects the final status.
            unsafe {
                codes
                    .as_mut_ptr()
                    .add(step)
                    .write(if valid { best_index } else { 0 });
                let status = status.as_mut_ptr();
                status.write(status.read() | if valid { 0 } else { 1 });
            }
        }
    }

    /// Gather the selected embedding into both feedback and the ordered sum rows.
    #[kernel]
    pub fn embedding(
        table: &[Bf16],
        codes: &[u32],
        mut feedback: DisjointSlice<Bf16>,
        mut rows: DisjointSlice<Bf16>,
        step: usize,
        table_stride: usize,
    ) {
        let column = thread::index_1d().get();
        if column < 2048 {
            let value = table[codes[step] as usize * table_stride + column];
            // SAFETY: the preceding greedy kernel publishes an ID < 2048,
            // step < 15, checked table pitch and dense private output extents.
            unsafe {
                feedback.as_mut_ptr().add(column).write(value);
                rows.as_mut_ptr()
                    .add((step + 1) * 2048 + column)
                    .write(value);
            }
        }
    }

    /// Check active attention guard boundaries before scratch is reused.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn attention_guards(
        partials: &[f32],
        query: &[Bf16],
        key: &[Bf16],
        output: &[Bf16],
        cache_key: &[Bf16],
        cache_value: &[Bf16],
        mut status: DisjointSlice<u32>,
        rows: usize,
    ) {
        if thread::index_1d().get() == 0 {
            let mut valid = true;
            let mut i = 0;
            while i < 8 {
                valid = valid && partials[rows * 16 * 130 + i].to_bits() == 0x7fc1_0001;
                valid = valid && query[rows * 2048 + i].to_bits() == 0x7fc1;
                valid = valid && key[rows * 1024 + i].to_bits() == 0x7fc1;
                valid = valid && output[rows * 2048 + i].to_bits() == 0x7fc1;
                valid = valid && cache_key[16 * 1024 + i].to_bits() == 0x7fc1;
                valid = valid && cache_value[16 * 1024 + i].to_bits() == 0x7fc1;
                i += 1;
            }
            // Single writer, ordered after every merge on the plan's stream.
            unsafe {
                let status = status.as_mut_ptr();
                status.write(status.read() | if valid { 0 } else { 2 });
            }
        }
    }
}
