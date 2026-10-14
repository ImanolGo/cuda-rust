/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit completion controls for ordinary global-to-shared `cp.async`.
//! These operate on the executing thread's copy groups, separately from TMA
//! bulk-copy groups. Kernel authors supply CTA synchronization before sharing
//! completed destinations with other threads.

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

/// Issue one four-byte global-to-shared copy with the cache-all policy (sm_80+).
/// No commit, wait or thread synchronization is inserted.
///
/// # Safety
/// Both pointers must be four-byte aligned and cover four live bytes in their
/// respective address spaces. Keep both allocations live until completion;
/// coordinate destination access and finish the copy before reading it.
#[inline(always)]
pub unsafe fn cp_async_ca_4(shared_dst: *mut u32, global_src: *const u32) {
    unsafe { cuda_device::async_copy::cp_async_ca_4(shared_dst, global_src) }
}

/// Attach this thread's preceding `cp.async` completion to a barrier (sm_80+).
/// This does not increment the barrier's expected arrival count or wait.
///
/// # Safety
/// `barrier` must name a live, aligned, initialized CTA-shared mbarrier. Its
/// phase must already expect this completion arrival. Keep the barrier and
/// copy allocations live until completion and coordinate phase transitions.
#[inline(always)]
pub unsafe fn cp_async_mbarrier_arrive_noinc(barrier: *mut crate::mbarrier::Barrier) {
    unsafe { cuda_device::async_copy::cp_async_mbarrier_arrive_noinc(barrier) }
}
