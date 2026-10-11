/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Canonical barrier states with explicit instruction-only lifecycle methods.
pub use crate::barrier::{
    Barrier, GeneralBarrier, Invalidated, ManagedBarrier, Ready, TmaBarrier, Uninit,
};
pub use crate::barrier::{
    BarrierToken, barrier_cta_arrive, barrier_cta_arrive_aligned, barrier_cta_sync,
    barrier_cta_sync_aligned, fence_mbarrier_init_release_cluster, fence_proxy_async_shared_cta,
};

/// Issue one arrival, returning the hardware phase token.
///
/// # Safety
/// This participant must be expected in the current live barrier phase.
#[inline(always)]
pub unsafe fn arrive<K, const ID: usize>(bar: &ManagedBarrier<Ready, K, ID>) -> BarrierToken {
    bar.arrive()
}
/// Issue one arrival and add the expected TMA byte count. Arrival decrement is one.
///
/// # Safety
/// The issuing participant must be expected, and `bytes` must match the
/// asynchronous transfers completing against this barrier phase.
#[inline(always)]
pub unsafe fn arrive_expect_tx<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    bar.arrive_expect_tx(bytes)
}
/// Try the supplied phase parity once; no loop or phase update is inserted.
///
/// # Safety
/// Parity must be 0 or 1 and agree with the participant's intended barrier phase.
#[inline(always)]
pub unsafe fn try_wait_parity<K, const ID: usize>(
    bar: &ManagedBarrier<Ready, K, ID>,
    parity: u32,
) -> bool {
    bar.try_wait_parity(parity)
}

/// Arrive once and add expected TMA bytes with relaxed cluster scope.
///
/// # Safety
/// One expected participant issues this for the current phase. `bytes` must
/// cover the asynchronous transfers completing at this live local barrier;
/// initialization must already be published to every cluster participant.
#[inline(always)]
pub unsafe fn arrive_expect_tx_cluster<const ID: usize>(
    bar: &ManagedBarrier<Ready, TmaBarrier, ID>,
    bytes: u32,
) -> BarrierToken {
    BarrierToken::from_raw(unsafe {
        crate::barrier::mbarrier_arrive_expect_tx_cluster(bar.as_ptr(), 1, bytes)
    })
}

/// Test parity once with cluster-scope acquire ordering; no software wait loop.
///
/// # Safety
/// The live local barrier must track this participant's intended phase.
/// `parity` must be zero or one; coordinate remote participants explicitly.
#[inline(always)]
pub unsafe fn try_wait_parity_cluster<K, const ID: usize>(
    bar: &ManagedBarrier<Ready, K, ID>,
    parity: u32,
) -> bool {
    unsafe { crate::barrier::mbarrier_try_wait_parity_cluster(bar.as_ptr(), parity) }
}
