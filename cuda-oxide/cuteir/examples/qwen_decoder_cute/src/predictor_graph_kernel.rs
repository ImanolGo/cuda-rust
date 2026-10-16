/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Scheduling helpers for the private, fixed-shape predictor region.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;
use cute_rs::partition::{StridedTensor, StridedTensorMut};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn reset_status(mut status: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            // SAFETY: the plan owns one status element and launches one writer.
            unsafe { StridedTensorMut::from_disjoint_slice(&mut status).store_at(0, 0) };
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
                let source = StridedTensor::from_slice(input).subview_unchecked(input_start + index, 1);
                let mut destination = StridedTensorMut::from_disjoint_slice(&mut output).subview_unchecked(output_start + index, 1);
                source.map_to(&mut destination, |value| value);
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
        // SAFETY: the private plan launches a complete warp over 2048 logits;
        // all lanes execute the same partition/reduction, including invalid data.
        let selected = unsafe {
            cute_rs::reduction::argmax_bf16(StridedTensor::from_slice(logits).subview(0, 2048), lane)
        };
        let valid = selected.is_valid();
        if lane == 0 {
            // Invalid frames still select a bounded fallback so their later
            // embedding reads remain safe; the host rejects the final status.
            unsafe {
                StridedTensorMut::from_disjoint_slice(&mut codes)
                    .store_at(step, if valid { selected.index() } else { 0 });
                StridedTensorMut::from_disjoint_slice(&mut status)
                    .update_at(0, |value| value | if valid { 0 } else { 1 });
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
            let table = StridedTensor::from_slice(table);
            let codes = StridedTensor::from_slice(codes);
            let selected = table.subview(codes.get(step) as usize * table_stride, 2048);
            let value = selected.get(column);
            // SAFETY: the preceding greedy kernel publishes an ID < 2048,
            // step < 15, checked table pitch and dense private output extents.
            unsafe {
                StridedTensorMut::from_disjoint_slice(&mut feedback).store_at(column, value);
                StridedTensorMut::from_disjoint_slice(&mut rows)
                    .subview_unchecked((step + 1) * 2048, 2048).store_at(column, value);
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
            // Each guard is a register tile; ordered folds preserve sticky
            // failure without sharing writable scratch across iterations.
            let valid = StridedTensor::from_slice(partials).subview(rows * 16 * 130, 8).load::<8>().fold(true, |valid, value| valid && value.to_bits() == 0x7fc1_0001);
            let valid = StridedTensor::from_slice(query).subview(rows * 2048, 8).load::<8>().fold(valid, |valid, value| valid && value.to_bits() == 0x7fc1);
            let valid = StridedTensor::from_slice(key).subview(rows * 1024, 8).load::<8>().fold(valid, |valid, value| valid && value.to_bits() == 0x7fc1);
            let valid = StridedTensor::from_slice(output).subview(rows * 2048, 8).load::<8>().fold(valid, |valid, value| valid && value.to_bits() == 0x7fc1);
            let valid = StridedTensor::from_slice(cache_key).subview(16 * 1024, 8).load::<8>().fold(valid, |valid, value| valid && value.to_bits() == 0x7fc1);
            let valid = StridedTensor::from_slice(cache_value).subview(16 * 1024, 8).load::<8>().fold(valid, |valid, value| valid && value.to_bits() == 0x7fc1);
            // Single writer, ordered after every merge on the plan's stream.
            unsafe {
                StridedTensorMut::from_disjoint_slice(&mut status)
                    .update_at(0, |value| value | if valid { 0 } else { 2 });
            }
        }
    }
}
