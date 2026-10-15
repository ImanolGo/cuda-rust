/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Table-driven BF16 split-half rotation and contiguous cache append.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, thread};
use cute_rs::Bf16;

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rotate_bf16(
        input: &[Bf16],
        cos: &[Bf16],
        sin: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let pair = thread::index_1d().get();
        let half = dim / 2;
        if pair < tokens * heads * half {
            let row = pair / half;
            let frequency = pair % half;
            let table = (row / heads) * half + frequency;
            let left = row * dim + frequency;
            let right = left + half;
            let a = input[left].to_f32();
            let b = input[right].to_f32();
            let c = cos[table].to_f32();
            let s = sin[table].to_f32();
            // Eager BF16 RoPE rounds each multiplication before addition.
            // Keeping these explicit conversions also prevents FMA contraction
            // across the observable intermediate rounding boundaries.
            let ac = Bf16::from_f32(a * c);
            let neg_bs = Bf16::from_f32(-b * s);
            let bc = Bf16::from_f32(b * c);
            let a_s = Bf16::from_f32(a * s);
            let x = Bf16::from_f32(ac.to_f32() + neg_bs.to_f32());
            let y = Bf16::from_f32(bc.to_f32() + a_s.to_f32());
            // SAFETY: checked dense extents, even head width and one thread
            // per pair give two disjoint, in-bounds coordinates per thread.
            unsafe {
                output.as_mut_ptr().add(left).write(x);
                output.as_mut_ptr().add(right).write(y);
            }
        }
    }

    #[kernel]
    pub fn append_bf16(
        key: &[Bf16],
        value: &[Bf16],
        mut cache_key: DisjointSlice<Bf16>,
        mut cache_value: DisjointSlice<Bf16>,
        start: usize,
        count: usize,
    ) {
        let index = thread::index_1d().get();
        if index < count {
            // SAFETY: the host checked start+count against the cache capacity.
            // Each thread owns one new slot beyond the completed prefix.
            unsafe {
                cache_key.as_mut_ptr().add(start + index).write(key[index]);
                cache_value
                    .as_mut_ptr()
                    .add(start + index)
                    .write(value[index]);
            }
        }
    }
}
