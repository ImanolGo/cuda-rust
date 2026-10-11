/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Canonical barrier states with explicit instruction-only lifecycle methods.
use crate::memory::SharedAddress;
pub use cuda_device::barrier::{
    Barrier, BarrierToken, GeneralBarrier, Invalidated, ManagedBarrier, Ready, TmaBarrier, Uninit,
};

/// Signal arrival at a numbered CTA barrier without waiting.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_cta_arrive(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_arrive(barrier_id, thread_count) }
}

/// Signal arrival at a numbered CTA barrier in aligned mode without waiting.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
/// Every non-exited thread in the warp must execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cta_arrive_aligned(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_arrive_aligned(barrier_id, thread_count) }
}

/// Synchronize the requested number of threads at a numbered CTA barrier.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
#[inline(always)]
pub unsafe fn barrier_cta_sync(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_sync(barrier_id, thread_count) }
}

/// Synchronize at a numbered CTA barrier in aligned mode.
///
/// # Safety
/// Participating threads must use a compatible barrier ID and expected count.
/// Every non-exited thread in the warp must execute with identical control flow.
#[inline(always)]
pub unsafe fn barrier_cta_sync_aligned(barrier_id: u32, thread_count: u32) {
    unsafe { cuda_device::barrier::barrier_cta_sync_aligned(barrier_id, thread_count) }
}

/// Publish preceding mbarrier initialization at cluster scope.
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`cuda_device::barrier::fence_mbarrier_init_release_cluster`].
#[inline(always)]
pub unsafe fn fence_mbarrier_init_release_cluster() {
    unsafe { cuda_device::barrier::fence_mbarrier_init_release_cluster() }
}

/// Make preceding CTA-shared generic-proxy writes visible to the async proxy.
///
/// # Safety
/// The caller must satisfy the barrier, scope, and memory-ordering contract of
/// [`cuda_device::barrier::fence_proxy_async_shared_cta`].
#[inline(always)]
pub unsafe fn fence_proxy_async_shared_cta() {
    unsafe { cuda_device::barrier::fence_proxy_async_shared_cta() }
}

/// Borrow typed shared addresses from initialized barrier handles.
pub trait BarrierAddressExt {
    /// Borrow a current-CTA shared address without extending this handle's lifetime.
    fn shared_address(&self) -> SharedAddress<'_, Barrier>;
}

impl<Kind, const ID: usize> BarrierAddressExt for ManagedBarrier<Ready, Kind, ID> {
    #[inline(always)]
    fn shared_address(&self) -> SharedAddress<'_, Barrier> {
        unsafe { SharedAddress::from_raw(self.as_ptr().cast_mut()) }
    }
}

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
        cuda_device::barrier::mbarrier_arrive_expect_tx_cluster(bar.as_ptr(), 1, bytes)
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
    unsafe { cuda_device::barrier::mbarrier_try_wait_parity_cluster(bar.as_ptr(), parity) }
}
