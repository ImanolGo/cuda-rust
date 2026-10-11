/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Two-CTA persistent FP16 GEMM expressed entirely with instruction primitives.
//! The schedule matches fp16_gemm_256x352_cute; election, phases and completion
//! are visible here rather than inserted by a layout/pipeline dialect.
use ctm_rs::mbarrier::BarrierAddressExt;
use ctm_rs::tcgen05::tmem::{TmemGuard, TmemReady, TmemUninit};
use ctm_rs::{
    cluster::{self, CtaMask},
    groups::CtaGroup2,
    mbarrier::{self, Barrier, GeneralBarrier, ManagedBarrier, Ready, TmaBarrier, Uninit},
    memory::{F16Tile128, F16TileB80, F16TileB96, F16TileC, PhysicalFormat, SharedRegion},
    tcgen05::{
        self,
        descriptors::*,
        mma::{self, collector_a},
        tmem::{ColumnOffset, TmemF32},
    },
    thread,
    tma::{self, Coordinates2D, TensorMap},
    warp,
};
use cuda_device::{
    DynamicSharedArray, cluster_launch, cuda_module, kernel, launch_bounds, launch_contract,
};

pub type ASmem = F16Tile128;
pub type B0Smem = F16TileB96;
pub type B1Smem = F16TileB80;
pub type CSmem = F16TileC;
type Left = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
type Right = DenseSharedF16<CtaGroup2, 256, 160, KMajor, KMajor>;
type Accumulator = TmemGuard<TmemReady, 512, CtaGroup2>;

pub const THREADS: u32 = 192;
const STAGES: usize = 5;
const A_BYTES: usize = ASmem::FOOTPRINT_BYTES;
const B_BYTES: usize = B0Smem::FOOTPRINT_BYTES + B1Smem::FOOTPRINT_BYTES;
const B_OFFSET: usize = A_BYTES * STAGES;
const C_OFFSET: usize = B_OFFSET + B_BYTES * STAGES;
const C_STAGE_BYTES: usize = CSmem::FOOTPRINT_BYTES;
const BARRIER_OFFSET: usize = C_OFFSET + 2 * C_STAGE_BYTES;
pub const DYNAMIC_SMEM_BYTES: usize = BARRIER_OFFSET + 128;
const TX_BYTES: u32 = (2 * (A_BYTES + B_BYTES)) as u32;
const _: () = assert!(DYNAMIC_SMEM_BYTES == 211_072 && TX_BYTES == 77_824);

// A violated physical contract must abort the kernel, never let one lane leave
// a collective while its peers wait. All valid launch layouts satisfy these.
macro_rules! checked {
    ($value:expr) => {
        match $value {
            Ok(value) => value,
            Err(_) => cuda_device::debug::trap(),
        }
    };
}

#[inline(always)]
unsafe fn region<'s, F: PhysicalFormat<Element = f16>>(base: *mut u8) -> SharedRegion<'s, F> {
    unsafe {
        checked!(SharedRegion::<F>::from_raw_parts(
            base.cast(),
            F::FOOTPRINT_BYTES
        ))
    }
}
#[inline(always)]
unsafe fn init<K>(pointer: *mut Barrier, arrivals: u32) {
    unsafe {
        ManagedBarrier::<Uninit, K>::from_static(pointer).init_instruction(arrivals);
    }
}
#[inline(always)]
unsafe fn wait<K>(barrier: &ManagedBarrier<Ready, K>, phase: u32) {
    unsafe { while !mbarrier::try_wait_parity_cluster(barrier, phase) {} }
}
#[inline(always)]
fn both_ctas() -> CtaMask<CtaGroup2> {
    checked!(CtaMask::try_from_bits(3))
}

#[inline(always)]
unsafe fn issue_stage(acc: &Accumulator, base: *mut u8, stage: usize, accumulate: bool) {
    unsafe {
        let a = region::<ASmem>(base.add(stage * A_BYTES));
        let b0 = region::<B0Smem>(base.add(B_OFFSET + stage * B_BYTES));
        let b1 = region::<B1Smem>(base.add(B_OFFSET + stage * B_BYTES + B0Smem::FOOTPRINT_BYTES));
        let a = checked!(SmemOperand::<OperandA, ASmem>::from_region(&a));
        let b0 = checked!(SmemOperand::<OperandB, B0Smem>::from_region(&b0));
        let b1 = checked!(SmemOperand::<OperandB, B1Smem>::from_region(&b1));
        let left = checked!(acc.view::<TmemF32<256, 192>>(ColumnOffset::new(0)));
        let right = checked!(acc.view::<TmemF32<256, 160>>(ColumnOffset::new(192)));
        let left_desc = InstructionDescriptor::<Left>::new();
        let right_desc = InstructionDescriptor::<Right>::new();
        macro_rules! pair {
            ($k:literal, $enable:expr) => {{
                mma::mma::<Left, collector_a::Fill>(
                    &left,
                    &a.k16::<$k>(),
                    &b0.k16::<$k>(),
                    &left_desc,
                    $enable,
                );
                mma::mma::<Right, collector_a::LastUse>(
                    &right,
                    &a.k16::<$k>(),
                    &b1.k16::<$k>(),
                    &right_desc,
                    $enable,
                );
            }};
        }
        pair!(0, accumulate);
        pair!(1, true);
        pair!(2, true);
        pair!(3, true);
    }
}

#[inline(always)]
unsafe fn epilogue(
    acc: &Accumulator,
    dst: &SharedRegion<'_, CSmem>,
    tid: u32,
    subtile: u32,
    bias: f32,
) {
    unsafe {
        let read = checked!(acc.read_32x32((tid / 32) * 32, ColumnOffset::new(subtile * 32)));
        let pending = mma::load_32x32b_x32(&read);
        mma::load_wait();
        // By-value, constant-index extraction keeps the hardware result order.
        let values = pending.decode_f32().to_array();
        let output = dst.address().as_raw();
        macro_rules! stores { ($($col:literal),*) => {$(
            output.add(crate::physical::output_offset(tid, $col)).write((values[$col] + bias) as f16);
        )*}; }
        stores!(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31
        );
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Tensor allocations and all four map descriptors stay live through completion.
    #[launch_bounds(192)]
    #[cluster_launch(2, 1, 1)]
    #[launch_contract(domain = 1, block = (192, 1, 1), dynamic_shared = 211072,
                      dynamic_shared_alignment = 1024, min_compute_capability = (10, 0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn fp16_gemm_256x352(
        #[grid_constant] a: &TensorMap<f16, ASmem>,
        #[grid_constant] b0: &TensorMap<f16, B0Smem>,
        #[grid_constant] b1: &TensorMap<f16, B1Smem>,
        #[grid_constant] c: &TensorMap<f16, CSmem>,
        bias: *const f16,
        m: u64,
        n: u64,
        k: u64,
        has_bias: u64,
    ) {
        let (m, n, k) = (m as u32, n as u32, k as u32);
        let tid = thread::threadIdx_x();
        let warp_id = tid / 32;
        let rank = cluster::block_rank();
        let smem = DynamicSharedArray::<u8, 1024>::get_raw();
        unsafe {
            let full = smem.add(BARRIER_OFFSET).cast::<Barrier>();
            let empty = full.add(STAGES);
            let acc_empty_ptr = empty.add(STAGES);
            let acc_full_ptr = acc_empty_ptr.add(1);
            let dealloc_ptr = acc_full_ptr.add(1);
            let token = dealloc_ptr.add(1).cast::<u32>();
            if warp_id == 4 {
                tma::prefetch(a);
                tma::prefetch(b0);
                tma::prefetch(b1);
                tma::prefetch(c);
            }
            if warp_id == 0 && warp::is_elected_sync(u32::MAX) {
                init::<GeneralBarrier>(dealloc_ptr, 32);
                init::<GeneralBarrier>(acc_empty_ptr, 8);
                init::<GeneralBarrier>(acc_full_ptr, 1);
                let mut stage = 0;
                while stage < STAGES {
                    init::<TmaBarrier>(full.add(stage), 1);
                    init::<GeneralBarrier>(empty.add(stage), 1);
                    stage += 1;
                }
            }
            mbarrier::fence_mbarrier_init_release_cluster();
            cluster::barrier_cluster_arrive_relaxed();
            let mtiles = m / 256;
            let total_tiles = mtiles * n.div_ceil(352);
            let mut tile = thread::blockIdx_x() / 2;
            let tile_stride = thread::gridDim_x() / 2;
            let ktiles = k.div_ceil(64);
            cluster::barrier_cluster_wait();
            let acc_empty = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(acc_empty_ptr);
            let acc_full = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(acc_full_ptr);

            if warp_id == 5 {
                let mut stage = 0usize;
                let mut phase = 1u32;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let mut kt = 0;
                    while kt < ktiles {
                        let reusable =
                            ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(stage));
                        wait(&reusable, phase);
                        let loaded =
                            ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(stage));
                        if warp::is_elected_sync(u32::MAX) {
                            if rank == 0 {
                                mbarrier::arrive_expect_tx_cluster(&loaded, TX_BYTES);
                            }
                            let completion = cluster::map_shared(loaded.shared_address(), 0);
                            let mask = checked!(CtaMask::<CtaGroup2>::try_from_bits(1 << rank));
                            let a_tile = region::<ASmem>(smem.add(stage * A_BYTES));
                            let b0_tile = region::<B0Smem>(smem.add(B_OFFSET + stage * B_BYTES));
                            let b1_tile = region::<B1Smem>(
                                smem.add(B_OFFSET + stage * B_BYTES + B0Smem::FOOTPRINT_BYTES),
                            );
                            tma::load_2d_cg2(
                                &a_tile,
                                a,
                                Coordinates2D::new((kt * 64) as i32, row as i32),
                                completion,
                                mask,
                            );
                            tma::load_2d_cg2(
                                &b0_tile,
                                b0,
                                Coordinates2D::new((kt * 64) as i32, (col + rank * 96) as i32),
                                completion,
                                mask,
                            );
                            tma::load_2d_cg2(
                                &b1_tile,
                                b1,
                                Coordinates2D::new(
                                    (kt * 64) as i32,
                                    (col + 192 + rank * 80) as i32,
                                ),
                                completion,
                                mask,
                            );
                        }
                        stage += 1;
                        if stage == STAGES {
                            stage = 0;
                            phase ^= 1;
                        }
                        kt += 1;
                    }
                    tile += tile_stride;
                }
                // Reach the last producer slot/phase even for short K and ring wraps.
                let mut tail = 0;
                while tail < STAGES - 1 {
                    stage += 1;
                    if stage == STAGES {
                        stage = 0;
                        phase ^= 1;
                    }
                    tail += 1;
                }
                if warp::is_elected_sync(u32::MAX) {
                    wait(
                        &ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(empty.add(stage)),
                        phase,
                    );
                }
            } else if warp_id == 4 {
                mbarrier::barrier_cta_sync(2, 160);
                let acc = Accumulator::adopt_ready(token);
                let mut stage = 0usize;
                let mut phase = 0u32;
                let mut acc_phase = 1u32;
                while tile < total_tiles {
                    if rank == 0 {
                        wait(&acc_empty, acc_phase);
                        tcgen05::fence_after_thread_sync();
                        let mut accumulate = false;
                        let mut kt = 0;
                        while kt < ktiles {
                            let loaded =
                                ManagedBarrier::<Ready, TmaBarrier>::adopt_ready(full.add(stage));
                            wait(&loaded, phase);
                            tcgen05::fence_after_thread_sync();
                            if warp::is_elected_sync(u32::MAX) {
                                issue_stage(&acc, smem, stage, accumulate);
                                let reusable = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(
                                    empty.add(stage),
                                );
                                tcgen05::commit_multicast_cg2(
                                    reusable.shared_address(),
                                    both_ctas(),
                                );
                            }
                            accumulate = true;
                            stage += 1;
                            if stage == STAGES {
                                stage = 0;
                                phase ^= 1;
                            }
                            kt += 1;
                        }
                        if warp::is_elected_sync(u32::MAX) {
                            tcgen05::commit_multicast_cg2(acc_full.shared_address(), both_ctas());
                        }
                    }
                    acc_phase ^= 1;
                    tile += tile_stride;
                }
                if rank == 0 && warp::is_elected_sync(u32::MAX) {
                    wait(&acc_empty, acc_phase);
                    tcgen05::fence_after_thread_sync();
                }
            } else {
                if warp_id == 0 {
                    let owner = TmemGuard::<TmemUninit, 512, CtaGroup2>::from_raw_slot(token);
                    owner.alloc_collective();
                    tcgen05::relinquish_alloc_permit_cg2();
                }
                mbarrier::barrier_cta_sync(2, 160);
                let acc = Accumulator::adopt_ready(token);
                let mut phase = 0u32;
                let mut epi_stage = 0usize;
                while tile < total_tiles {
                    let row = (tile % mtiles) * 256 + rank * 128;
                    let col = (tile / mtiles) * 352;
                    let row_bias = if has_bias != 0 {
                        *bias.add((row + tid) as usize) as f32
                    } else {
                        0.0
                    };
                    wait(&acc_full, phase);
                    tcgen05::fence_after_thread_sync();
                    let mut subtile = 0;
                    while subtile < 11 {
                        epi_stage ^= 1;
                        let dst = region::<CSmem>(smem.add(C_OFFSET + epi_stage * C_STAGE_BYTES));
                        epilogue(&acc, &dst, tid, subtile, row_bias);
                        mbarrier::fence_proxy_async_shared_cta();
                        mbarrier::barrier_cta_sync(1, 128);
                        if tid == 0 {
                            tma::store_2d(
                                &dst,
                                c,
                                Coordinates2D::new((col + subtile * 32) as i32, row as i32),
                            );
                            tma::bulk_commit_group();
                            tma::bulk_wait_group_read::<1>();
                        }
                        mbarrier::barrier_cta_sync(1, 128);
                        subtile += 1;
                    }
                    tcgen05::fence_before_thread_sync();
                    if warp::is_elected_sync(u32::MAX) {
                        cluster::arrive_remote(cluster::map_shared(acc_empty.shared_address(), 0));
                    }
                    phase ^= 1;
                    tile += tile_stride;
                }
                if warp_id == 0 {
                    if tid == 0 {
                        tma::bulk_wait_group_read::<0>();
                    }
                    mbarrier::barrier_cta_sync(3, 32);
                    let dealloc = ManagedBarrier::<Ready, GeneralBarrier>::adopt_ready(dealloc_ptr);
                    cluster::arrive_remote(cluster::map_shared(dealloc.shared_address(), rank ^ 1));
                    wait(&dealloc, 0);
                    acc.dealloc_collective();
                }
            }
        }
    }
}
