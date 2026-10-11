/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit three-stage SM120 MXFP4 GEMM. No CuTe layout or pipeline operations.
use crate::physical;
use ctm_rs::mbarrier::BarrierAddressExt;
use ctm_rs::{
    mbarrier::{self, Barrier, GeneralBarrier, ManagedBarrier, Ready, TmaBarrier, Uninit},
    memory::{F16Tile128, Mxf4ScaleTile, Mxf4Tile128, PhysicalFormat, SharedAddress, SharedRegion},
    mma::{self, Accumulator16x8, ScalePairA, ScalePairB},
    thread,
    tma::{self, Coordinates2D, TensorMap},
};
use cuda_device::{DynamicSharedArray, cuda_module, kernel, launch_bounds, launch_contract};
pub const M: usize = 2048;
pub const N: usize = 2048;
pub const K: usize = 1024;
pub const KB: usize = K / 32;
pub const K_STAGES: usize = K / 128;
pub const THREADS: usize = 288;
pub const TOTAL_TILES: usize = (M / 128) * (N / 128);
pub const DYNAMIC_SMEM_BYTES: usize = 86_016;
const STAGES: usize = 3;
const DATA_BYTES: usize = 8192;
const SCALE_BYTES: usize = 512;
const B_OFFSET: usize = 24576;
const SFA_OFFSET: usize = 49152;
const SFB_OFFSET: usize = 50688;
const BARRIERS: usize = 52224;
const OUTPUT: usize = 53248;
const TX_BYTES: u32 = 17408;
const _: () = assert!(OUTPUT + 2 * F16Tile128::FOOTPRINT_BYTES == DYNAMIC_SMEM_BYTES);

#[inline(always)]
unsafe fn region<'s, F: PhysicalFormat>(pointer: *mut u8) -> SharedRegion<'s, F> {
    unsafe { SharedRegion::from_raw_parts_or_trap(pointer.cast(), F::FOOTPRINT_BYTES) }
}
#[inline(always)]
unsafe fn wait<K>(bar: &ManagedBarrier<Ready, K>, phase: u32) {
    unsafe { while !mbarrier::try_wait_parity(bar, phase) {} }
}

#[cuda_module]
pub mod kernels {
    use super::*;
    #[launch_bounds(288)]
    #[launch_contract(domain=1, block=(288,1,1), dynamic_shared=86016,
                      dynamic_shared_alignment=1024, min_compute_capability=(12,0))]
    #[kernel]
    pub fn blockscale_gemm(
        #[grid_constant] a_map: &TensorMap<u8, Mxf4Tile128>,
        #[grid_constant] b_map: &TensorMap<u8, Mxf4Tile128>,
        #[grid_constant] sa_map: &TensorMap<u16, Mxf4ScaleTile>,
        #[grid_constant] sb_map: &TensorMap<u16, Mxf4ScaleTile>,
        #[grid_constant] c_map: &TensorMap<f16, F16Tile128>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let warp = tid / 32;
        let lane = tid % 32;
        let base = DynamicSharedArray::<u8, 1024>::get_raw();
        unsafe {
            let full = base.add(BARRIERS).cast::<Barrier>();
            let empty = full.add(STAGES);
            if tid == 0 {
                let mut s = 0;
                while s < STAGES {
                    ManagedBarrier::<Uninit, TmaBarrier>::from_static(full.add(s))
                        .init_instruction(1);
                    ManagedBarrier::<Uninit, GeneralBarrier>::from_static(empty.add(s))
                        .init_instruction(8);
                    s += 1;
                }
            }
            mbarrier::fence_mbarrier_init_release_cluster();
            mbarrier::barrier_cta_sync(0, THREADS as u32);
            let mut tile = thread::blockIdx_x() as usize;
            let stride = thread::gridDim_x() as usize;
            let mut slot = 0usize;
            if warp < 8 {
                let wm = warp % 4;
                let wn = warp / 4;
                let mut phase = 0u32;
                while tile < TOTAL_TILES {
                    let tile_m = tile % (M / 128);
                    let tile_n = tile / (M / 128);
                    let mut c00 = Accumulator16x8::zero();
                    let mut c01 = Accumulator16x8::zero();
                    let mut c02 = Accumulator16x8::zero();
                    let mut c03 = Accumulator16x8::zero();
                    let mut c04 = Accumulator16x8::zero();
                    let mut c05 = Accumulator16x8::zero();
                    let mut c06 = Accumulator16x8::zero();
                    let mut c07 = Accumulator16x8::zero();
                    let mut c10 = Accumulator16x8::zero();
                    let mut c11 = Accumulator16x8::zero();
                    let mut c12 = Accumulator16x8::zero();
                    let mut c13 = Accumulator16x8::zero();
                    let mut c14 = Accumulator16x8::zero();
                    let mut c15 = Accumulator16x8::zero();
                    let mut c16 = Accumulator16x8::zero();
                    let mut c17 = Accumulator16x8::zero();
                    let mut stage = 0;
                    while stage < K_STAGES {
                        let ready =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(slot));
                        wait(&ready, phase);
                        let a = base.add(slot * DATA_BYTES);
                        let b = base.add(B_OFFSET + slot * DATA_BYTES);
                        let sa = base.add(SFA_OFFSET + slot * SCALE_BYTES).cast::<u32>();
                        let sb = base.add(SFB_OFFSET + slot * SCALE_BYTES).cast::<u32>();
                        let a_provider = lane / 4 + 8 * (lane & 1);
                        let sa0 = *sa.add(physical::scale_word(16 * wm + a_provider));
                        let sa1 = *sa.add(physical::scale_word(16 * wm + a_provider + 64));
                        let sb0 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 0));
                        let sb1 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 8));
                        let sb2 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 32));
                        let sb3 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 40));
                        let sb4 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 64));
                        let sb5 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 72));
                        let sb6 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 96));
                        let sb7 = *sb.add(physical::scale_word(16 * wn + lane / 4 + 104));
                        let mut half = 0;
                        while half < 2 {
                            let a0 = mma::load_mxf4_a(SharedAddress::from_raw(
                                a.add(physical::matrix_row(wm, half, lane)).cast(),
                            ));
                            let a1 = mma::load_mxf4_a(SharedAddress::from_raw(
                                a.add(physical::matrix_row(wm + 4, half, lane)).cast(),
                            ));
                            let scale_a0 = ScalePairA::from_bits((sa0 >> (16 * half)) as u16);
                            let scale_a1 = ScalePairA::from_bits((sa1 >> (16 * half)) as u16);
                            let (b0, b1) = mma::load_mxf4_b_pair(SharedAddress::from_raw(
                                b.add(physical::matrix_row(wn + 0, half, lane)).cast(),
                            ));
                            c00 = mma::mma_mxf4(
                                c00,
                                a0,
                                b0,
                                scale_a0,
                                ScalePairB::from_bits((sb0 >> (16 * half)) as u16),
                            );
                            c01 = mma::mma_mxf4(
                                c01,
                                a0,
                                b1,
                                scale_a0,
                                ScalePairB::from_bits((sb1 >> (16 * half)) as u16),
                            );
                            c10 = mma::mma_mxf4(
                                c10,
                                a1,
                                b0,
                                scale_a1,
                                ScalePairB::from_bits((sb0 >> (16 * half)) as u16),
                            );
                            c11 = mma::mma_mxf4(
                                c11,
                                a1,
                                b1,
                                scale_a1,
                                ScalePairB::from_bits((sb1 >> (16 * half)) as u16),
                            );
                            let (b0, b1) = mma::load_mxf4_b_pair(SharedAddress::from_raw(
                                b.add(physical::matrix_row(wn + 2, half, lane)).cast(),
                            ));
                            c02 = mma::mma_mxf4(
                                c02,
                                a0,
                                b0,
                                scale_a0,
                                ScalePairB::from_bits((sb2 >> (16 * half)) as u16),
                            );
                            c03 = mma::mma_mxf4(
                                c03,
                                a0,
                                b1,
                                scale_a0,
                                ScalePairB::from_bits((sb3 >> (16 * half)) as u16),
                            );
                            c12 = mma::mma_mxf4(
                                c12,
                                a1,
                                b0,
                                scale_a1,
                                ScalePairB::from_bits((sb2 >> (16 * half)) as u16),
                            );
                            c13 = mma::mma_mxf4(
                                c13,
                                a1,
                                b1,
                                scale_a1,
                                ScalePairB::from_bits((sb3 >> (16 * half)) as u16),
                            );
                            let (b0, b1) = mma::load_mxf4_b_pair(SharedAddress::from_raw(
                                b.add(physical::matrix_row(wn + 4, half, lane)).cast(),
                            ));
                            c04 = mma::mma_mxf4(
                                c04,
                                a0,
                                b0,
                                scale_a0,
                                ScalePairB::from_bits((sb4 >> (16 * half)) as u16),
                            );
                            c05 = mma::mma_mxf4(
                                c05,
                                a0,
                                b1,
                                scale_a0,
                                ScalePairB::from_bits((sb5 >> (16 * half)) as u16),
                            );
                            c14 = mma::mma_mxf4(
                                c14,
                                a1,
                                b0,
                                scale_a1,
                                ScalePairB::from_bits((sb4 >> (16 * half)) as u16),
                            );
                            c15 = mma::mma_mxf4(
                                c15,
                                a1,
                                b1,
                                scale_a1,
                                ScalePairB::from_bits((sb5 >> (16 * half)) as u16),
                            );
                            let (b0, b1) = mma::load_mxf4_b_pair(SharedAddress::from_raw(
                                b.add(physical::matrix_row(wn + 6, half, lane)).cast(),
                            ));
                            c06 = mma::mma_mxf4(
                                c06,
                                a0,
                                b0,
                                scale_a0,
                                ScalePairB::from_bits((sb6 >> (16 * half)) as u16),
                            );
                            c07 = mma::mma_mxf4(
                                c07,
                                a0,
                                b1,
                                scale_a0,
                                ScalePairB::from_bits((sb7 >> (16 * half)) as u16),
                            );
                            c16 = mma::mma_mxf4(
                                c16,
                                a1,
                                b0,
                                scale_a1,
                                ScalePairB::from_bits((sb6 >> (16 * half)) as u16),
                            );
                            c17 = mma::mma_mxf4(
                                c17,
                                a1,
                                b1,
                                scale_a1,
                                ScalePairB::from_bits((sb7 >> (16 * half)) as u16),
                            );
                            half += 1;
                        }
                        if lane == 0 {
                            mbarrier::arrive(
                                &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                                    empty.add(slot),
                                ),
                            );
                        }
                        slot += 1;
                        if slot == STAGES {
                            slot = 0;
                            phase ^= 1;
                        }
                        stage += 1;
                    }
                    // The producer warp never joins the 256-thread output barriers.
                    if tid == 0 {
                        tma::bulk_wait_group_read::<0>();
                    }
                    mbarrier::barrier_cta_sync(2, 256);
                    let output = base.add(OUTPUT).cast::<f16>();
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 0)).cast(),
                        ),
                        c00,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 1)).cast(),
                        ),
                        c01,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 2)).cast(),
                        ),
                        c02,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 3)).cast(),
                        ),
                        c03,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 4)).cast(),
                        ),
                        c04,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 5)).cast(),
                        ),
                        c05,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 6)).cast(),
                        ),
                        c06,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 0, 7)).cast(),
                        ),
                        c07,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 0)).cast(),
                        ),
                        c10,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 1)).cast(),
                        ),
                        c11,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 2)).cast(),
                        ),
                        c12,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 3)).cast(),
                        ),
                        c13,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 4)).cast(),
                        ),
                        c14,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 5)).cast(),
                        ),
                        c15,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 6)).cast(),
                        ),
                        c16,
                    );
                    mma::store_f16(
                        SharedAddress::from_raw(
                            output.add(physical::output_row(warp, lane, 1, 7)).cast(),
                        ),
                        c17,
                    );
                    mbarrier::fence_proxy_async_shared_cta();
                    mbarrier::barrier_cta_sync(2, 256);
                    if tid == 0 {
                        let left = region::<F16Tile128>(base.add(OUTPUT));
                        let right = region::<F16Tile128>(base.add(OUTPUT + 16384));
                        tma::store_2d(
                            &left,
                            c_map,
                            Coordinates2D::new((tile_n * 128) as i32, (tile_m * 128) as i32),
                        );
                        tma::store_2d(
                            &right,
                            c_map,
                            Coordinates2D::new((tile_n * 128 + 64) as i32, (tile_m * 128) as i32),
                        );
                        tma::bulk_commit_group();
                    }
                    tile += stride;
                }
                if tid == 0 {
                    tma::bulk_wait_group_read::<0>();
                }
            } else if lane == 0 {
                let mut phase = 1u32;
                while tile < TOTAL_TILES {
                    let tile_m = tile % (M / 128);
                    let tile_n = tile / (M / 128);
                    let mut stage = 0;
                    while stage < K_STAGES {
                        wait(
                            &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(slot)),
                            phase,
                        );
                        let ready =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(slot));
                        mbarrier::arrive_expect_tx(&ready, TX_BYTES);
                        let a = region::<Mxf4Tile128>(base.add(slot * DATA_BYTES));
                        let b = region::<Mxf4Tile128>(base.add(B_OFFSET + slot * DATA_BYTES));
                        let sa = region::<Mxf4ScaleTile>(base.add(SFA_OFFSET + slot * SCALE_BYTES));
                        let sb = region::<Mxf4ScaleTile>(base.add(SFB_OFFSET + slot * SCALE_BYTES));
                        tma::load_2d(
                            &a,
                            a_map,
                            Coordinates2D::new((stage * 64) as i32, (tile_m * 128) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &b,
                            b_map,
                            Coordinates2D::new((stage * 64) as i32, (tile_n * 128) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &sa,
                            sa_map,
                            Coordinates2D::new(0, (tile_m * K_STAGES + stage) as i32),
                            ready.shared_address(),
                        );
                        tma::load_2d(
                            &sb,
                            sb_map,
                            Coordinates2D::new(0, (tile_n * K_STAGES + stage) as i32),
                            ready.shared_address(),
                        );
                        slot += 1;
                        if slot == STAGES {
                            slot = 0;
                            phase ^= 1;
                        }
                        stage += 1;
                    }
                    tile += stride;
                }
                let mut tail = 0;
                while tail < STAGES {
                    wait(
                        &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(slot)),
                        phase,
                    );
                    slot += 1;
                    if slot == STAGES {
                        slot = 0;
                        phase ^= 1;
                    }
                    tail += 1;
                }
            }
        }
    }
}
