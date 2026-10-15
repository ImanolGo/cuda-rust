/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 grouped-query attention using parallel, stable 64-key reductions.
use core::marker::PhantomData;
use ctm_rs::{
    async_copy, cuda_module, float, kernel, launch_bounds, launch_contract, mbarrier,
    memory::{DisjointSlice, DynamicSharedArray},
    thread, warp,
};
use cute_rs::markers::{C, Composed, CpAsync, L, LeadingDim, RowMajor, Swizzle, T2};
use cute_rs::{Bf16, GmemMatrix, SmemTile, copy_g2s};

/// Shared bytes for two 64-by-128 BF16 tiles and FP32 reduction state.
pub const PARTIAL_SHARED_BYTES: u32 = 35_104;
/// FP32 values per split: maximum, denominator, and 128 numerator coordinates.
pub const STATE_SIZE: usize = 130;

/// Shared-memory storage layout for one complete BF16 key or value tile.
type KvLayout = Composed<Swizzle<3, 3, 3>, 0, RowMajor<64, 128>>;
/// Eight rows per 128-thread copy; each thread transports eight BF16 values.
type CopyLayout = Composed<Swizzle<3, 3, 3>, 0, RowMajor<8, 128>>;
type CopyThreads = L<T2<C<8>, C<16>>, T2<C<16>, C<1>>>;
type CopyValues = L<T2<C<1>, C<8>>, T2<C<8>, C<1>>>;

/// BF16 storage carried unchanged by the existing typed integer copy primitive.
///
/// The private wrapper keeps the physical `u16` copy carrier separate from
/// numeric BF16 values. No FP16 reinterpretation or numeric cast occurs.
struct Bf16KvTile(SmemTile<u16, KvLayout>);

/// Attention scale `1 / sqrt(128)` in base two for the exponential instruction.
const SCALE_LOG2: f32 = 0.088_388_346 * core::f32::consts::LOG2_E;

/// Expand fixed iterations so per-lane numerator coordinates remain registers.
macro_rules! static_for {
    ($index:ident in [$($value:expr),*] $body:block) => {
        $({ const $index: usize = $value; $body })*
    };
}

#[inline(always)]
fn maximum(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// All lanes participate in the full-warp shuffle sequence.
#[inline(always)]
fn warp_max(mut value: f32) -> f32 {
    static_for!(MASK in [16, 8, 4, 2, 1] {
        value = maximum(value, warp::shuffle_xor_f32(value, MASK as u32));
    });
    value
}

/// All lanes participate in the full-warp shuffle sequence.
#[inline(always)]
fn warp_sum(mut value: f32) -> f32 {
    static_for!(MASK in [16, 8, 4, 2, 1] {
        value += warp::shuffle_xor_f32(value, MASK as u32);
    });
    value
}

#[inline(always)]
fn shared_offset(row: usize, column: usize) -> usize {
    let linear = row * 128 + column;
    linear ^ ((linear & (7 << 6)) >> 3)
}

impl Bf16KvTile {
    /// # Safety
    /// `base` supplies 8192 live, suitably aligned shared-memory `u16` elements.
    #[inline(always)]
    unsafe fn new(base: *mut u16) -> Self {
        Self(SmemTile {
            base,
            capacity: 8192,
            layout: PhantomData,
        })
    }

    /// # Safety
    /// The copy must be complete and CTA-synchronized; row < 64 and column < 128.
    #[inline(always)]
    unsafe fn value(&self, row: usize, column: usize) -> f32 {
        Bf16::from_bits(unsafe { *self.0.base.add(shared_offset(row, column)) }).to_f32()
    }
}

/// Stage only initialized and causally visible cache rows, zero-filling tails.
///
/// # Safety
/// All 128 threads participate with their unique thread IDs. `source` contains
/// at least `visible * kv_heads * 128` initialized BF16 values, `head < kv_heads`,
/// and destination storage remains live until the copy has been waited on.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn stage(
    source: &[Bf16],
    head: usize,
    kv_heads: usize,
    visible: usize,
    split: usize,
    destination: &Bf16KvTile,
    tid: u32,
) {
    // Bf16 is repr(transparent) over u16. This cast changes only the copy
    // carrier; loads later restore the BF16 type before numerical conversion.
    let source = GmemMatrix {
        base: unsafe { source.as_ptr().add(head * 128) }.cast::<u16>(),
        rows: visible,
        cols: 128,
        leading_dim: LeadingDim::<8> {
            elements: kv_heads * 128,
        },
    };
    static_for!(GROUP in [0, 1, 2, 3, 4, 5, 6, 7] {
        let mut part = SmemTile::<u16, CopyLayout> {
            base: unsafe { destination.0.base.add(GROUP * 1024) },
            capacity: 1024,
            layout: PhantomData,
        };
        unsafe {
            copy_g2s::<CpAsync<16>, CopyThreads, CopyValues, RowMajor<8, 128>, CopyLayout, u16>(
                &source, (split * 8 + GROUP, 0), &mut part, tid,
            );
        }
    });
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[launch_bounds(128)]
    #[launch_contract(domain=1, block=(128,1,1), dynamic_shared=35104,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn partial(
        q: &[Bf16],
        k: &[Bf16],
        v: &[Bf16],
        mut partials: DisjointSlice<f32>,
        query_tokens: u32,
        key_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        query_start: u32,
        causal: bool,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let splits = (key_tokens as usize).div_ceil(64);
        let work = thread::blockIdx_x() as usize;
        let split = work % splits;
        let query_head = (work / splits) % query_heads as usize;
        let query_row = (work / splits) / query_heads as usize;
        // This guard is CTA-uniform. The checked host grid supplies exactly
        // the query rows, heads and splits described by these dimensions.
        if query_row >= query_tokens as usize {
            return;
        }
        let visible = if causal {
            (query_start as usize + query_row + 1).min(key_tokens as usize)
        } else {
            key_tokens as usize
        };
        let kv_head = query_head / (query_heads as usize / kv_heads as usize);

        // SAFETY: the host checks head grouping, all tensor/scratch extents,
        // initialized KV prefix, context/readiness and the exact launch shape.
        // Each CTA owns one 130-value state; each lane owns four numerator
        // coordinates. All collectives below execute convergently.
        unsafe {
            let destination = partials.as_mut_ptr().add(work * STATE_SIZE);
            if split * 64 >= visible {
                // A causal query can have whole future splits. Publish the
                // neutral reduction state without touching any cache storage.
                if warp_id == 0 {
                    static_for!(D in [0, 1, 2, 3] {
                        *destination.add(2 + lane * 4 + D) = 0.0;
                    });
                    if lane == 0 {
                        *destination = f32::NEG_INFINITY;
                        *destination.add(1) = 0.0;
                    }
                }
                return;
            }

            let shared = DynamicSharedArray::<u16, 1024>::get_raw().cast::<u16>();
            let keys = Bf16KvTile::new(shared);
            let values = Bf16KvTile::new(shared.add(8192));
            let scores = shared.add(16384).cast::<f32>();
            let maxima = scores.add(64);
            let sums = maxima.add(4);
            let warp_outputs = sums.add(4);
            stage(
                k,
                kv_head,
                kv_heads as usize,
                visible,
                split,
                &keys,
                tid as u32,
            );
            stage(
                v,
                kv_head,
                kv_heads as usize,
                visible,
                split,
                &values,
                tid as u32,
            );
            async_copy::cp_async_commit_group();
            async_copy::cp_async_wait_group::<0>();
            mbarrier::barrier_cta_sync(1, 128);

            let query_base = (query_row * query_heads as usize + query_head) * 128;
            // Four neighboring lanes sum 32 dimensions apiece. Two passes
            // cover all 64 keys, including zero-filled tail rows.
            static_for!(PASS in [0, 1] {
                let row = tid / 4 + PASS * 32;
                let mut dot = 0.0f32;
                static_for!(D in [0, 1, 2, 3, 4, 5, 6, 7,
                                  8, 9, 10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23,
                                  24, 25, 26, 27, 28, 29, 30, 31] {
                    let column = (tid % 4) * 32 + D;
                    dot += (*q.as_ptr().add(query_base + column)).to_f32()
                        * keys.value(row, column);
                });
                dot += warp::shuffle_xor_f32(dot, 1);
                dot += warp::shuffle_xor_f32(dot, 2);
                if tid.is_multiple_of(4) {
                    *scores.add(row) = if split * 64 + row < visible {
                        dot
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            });
            mbarrier::barrier_cta_sync(1, 128);

            let score = if tid < 64 {
                *scores.add(tid)
            } else {
                f32::NEG_INFINITY
            };
            let local_maximum = warp_max(score);
            if lane == 0 {
                *maxima.add(warp_id) = local_maximum;
            }
            mbarrier::barrier_cta_sync(1, 128);
            let maximum = maximum(
                maximum(*maxima, *maxima.add(1)),
                maximum(*maxima.add(2), *maxima.add(3)),
            );
            // Nonempty splits have at least one visible score. Explicitly
            // zero masked rows before they contribute to normalization/PV.
            let probability = if tid < 64 && split * 64 + tid < visible {
                float::ex2_approx_f32((score - maximum) * SCALE_LOG2)
            } else {
                0.0
            };
            if tid < 64 {
                *scores.add(tid) = probability;
            }
            let local_sum = warp_sum(probability);
            if lane == 0 {
                *sums.add(warp_id) = local_sum;
            }
            mbarrier::barrier_cta_sync(1, 128);

            // A warp integrates 16 keys into four dimensions per lane.
            let mut numerator = [0.0f32; 4];
            static_for!(ROW in [0, 1, 2, 3, 4, 5, 6, 7,
                               8, 9, 10, 11, 12, 13, 14, 15] {
                let row = warp_id * 16 + ROW;
                let probability = *scores.add(row);
                static_for!(D in [0, 1, 2, 3] {
                    numerator[D] += probability * values.value(row, lane * 4 + D);
                });
            });
            static_for!(D in [0, 1, 2, 3] {
                *warp_outputs.add(warp_id * 128 + lane * 4 + D) = numerator[D];
            });
            mbarrier::barrier_cta_sync(1, 128);
            if warp_id == 0 {
                static_for!(D in [0, 1, 2, 3] {
                    let column = lane * 4 + D;
                    let mut sum = 0.0f32;
                    static_for!(WARP in [0, 1, 2, 3] {
                        sum += *warp_outputs.add(WARP * 128 + column);
                    });
                    *destination.add(2 + column) = sum;
                });
                if lane == 0 {
                    *destination = maximum;
                    *destination.add(1) = *sums + *sums.add(1) + *sums.add(2) + *sums.add(3);
                }
            }
        }
    }

    #[launch_bounds(32)]
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(12,0))]
    #[kernel]
    pub fn merge(
        partials: &[f32],
        mut out: DisjointSlice<Bf16>,
        query_tokens: u32,
        query_heads: u32,
        splits: u32,
    ) {
        let work = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        if work >= query_tokens as usize * query_heads as usize {
            return;
        }
        // SAFETY: one complete warp owns each query/head output. Host-checked
        // scratch extents include every split; the preceding launch completed
        // on this stream, and output coordinates are pairwise disjoint.
        unsafe {
            let base = partials.as_ptr().add(work * splits as usize * STATE_SIZE);
            let mut max_score = f32::NEG_INFINITY;
            let mut split = 0;
            while split < splits as usize {
                let state = base.add(split * STATE_SIZE);
                if *state.add(1) != 0.0 {
                    max_score = maximum(max_score, *state);
                }
                split += 1;
            }
            let mut denominator = 0.0f32;
            let mut numerator = [0.0f32; 4];
            split = 0;
            while split < splits as usize {
                let state = base.add(split * STATE_SIZE);
                let sum = *state.add(1);
                // A fully masked split is neutral. Skipping it also prevents
                // evaluating (-infinity)-(-infinity) for an empty state.
                if sum != 0.0 {
                    let scale = float::ex2_approx_f32((*state - max_score) * SCALE_LOG2);
                    denominator += sum * scale;
                    static_for!(D in [0, 1, 2, 3] {
                        numerator[D] += *state.add(2 + lane * 4 + D) * scale;
                    });
                }
                split += 1;
            }
            let reciprocal = if denominator == 0.0 {
                0.0
            } else {
                float::rcp_approx_ftz_f32(denominator)
            };
            static_for!(D in [0, 1, 2, 3] {
                *out.as_mut_ptr().add(work * 128 + lane * 4 + D) =
                    Bf16::from_f32(numerator[D] * reciprocal);
            });
        }
    }
}
