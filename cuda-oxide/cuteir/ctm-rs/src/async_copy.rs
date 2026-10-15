/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit completion controls for ordinary global-to-shared `cp.async`.
//! These operate on the executing thread's copy groups, separately from TMA
//! bulk-copy groups. Kernel authors supply CTA synchronization before sharing
//! completed destinations with other threads.

use crate::mbarrier::{GeneralBarrier, ManagedBarrier, Ready};
use crate::memory::SharedAddress;

/// Copy one aligned 32-bit global word into the calling CTA's shared memory.
///
/// # Safety
/// `source` must refer to global memory and remain unchanged until this copy
/// completes, even after its Rust borrow ends. Both operands must remain live
/// through completion. The destination is four-byte aligned shared storage
/// with no conflicting access. The issuing thread must supply matching copy
/// completion; synchronize threads before another thread reads or reuses the
/// destination. Custom completion assembly must include a compiler memory clobber.
///
/// ```compile_fail
/// # let source = 0u32;
/// unsafe { ctm_rs::async_copy::cp_async_ca_4(core::ptr::null_mut(), &source) };
/// ```
#[inline(always)]
pub unsafe fn cp_async_ca_4(destination: SharedAddress<'_, u32>, source: &u32) {
    unsafe { cuda_device::async_copy::cp_async_ca_4(destination.as_raw(), source) }
}

/// Signal this thread's outstanding copies through a ready local barrier.
/// The instruction consumes an already expected arrival without increasing it.
///
/// # Safety
/// This thread must be an expected participant in this live barrier phase.
/// Keep the barrier and all associated copy buffers live until completion;
/// callers coordinate every participant and phase explicitly.
///
/// ```compile_fail
/// use ctm_rs::mbarrier::{GeneralBarrier, ManagedBarrier, Uninit};
/// # fn rejected(barrier: &ManagedBarrier<Uninit, GeneralBarrier>) {
/// unsafe { ctm_rs::async_copy::cp_async_mbarrier_arrive_noinc(barrier) };
/// # }
/// ```
#[inline(always)]
pub unsafe fn cp_async_mbarrier_arrive_noinc<const ID: usize>(
    barrier: &ManagedBarrier<Ready, GeneralBarrier, ID>,
) {
    unsafe { cuda_device::async_copy::cp_async_mbarrier_arrive_noinc(barrier.as_ptr().cast_mut()) }
}

/// Commit this thread's uncommitted asynchronous copies as one group.
///
/// # Safety
/// Copies must satisfy their source/destination lifetime and alignment
/// contracts. Commit does not wait for completion or synchronize threads.
#[inline(always)]
pub unsafe fn cp_async_commit_group() {
    unsafe { cuda_device::async_copy::cp_async_commit_group() }
}

/// Wait until at most `N` of this thread's recent committed groups are pending.
///
/// `N` is an instruction immediate in `0..=7`. This inserts neither a CTA
/// barrier nor a wait for other threads' copies.
///
/// # Safety
/// Access only destinations whose copy groups this wait completes. Synchronize
/// threads before another thread accesses a completed destination. All copy
/// source and destination allocations must remain live until completion.
///
/// ```compile_fail
/// unsafe { ctm_rs::async_copy::cp_async_wait_group::<8>() };
/// ```
#[inline(always)]
pub unsafe fn cp_async_wait_group<const N: u32>() {
    const { assert!(N <= 7, "cp.async wait count must be in 0..=7") }
    unsafe { cuda_device::async_copy::cp_async_wait_group(N) }
}
