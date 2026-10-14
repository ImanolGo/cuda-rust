/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! SM100 tensor-memory allocation, FP16 MMA and register readback.
//!
//! Start with [`descriptors::DenseSharedF16`] for supported shapes and
//! [`tmem::TmemGuard`] for allocation ownership. Issue, commit, wait and release
//! remain explicit. The crate's `docs/primitives-coverage.md` gives a complete
//! issue-site example and the supported compiler profiles.

pub mod descriptors;
pub mod mma;
pub mod tmem;

/// Order later tcgen05 accesses after thread synchronization.
/// All tcgen05 operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn fence_after_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_after_thread_sync()
}

/// Order prior tcgen05 accesses before thread synchronization.
/// All tcgen05 operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn fence_before_thread_sync() {
    cuda_device::tcgen05::tcgen05_fence_before_thread_sync()
}

/// Release the group-2 tensor-memory allocation permit.
/// One full warp in each peer CTA must execute uniformly. All tcgen05
/// operations in the kernel must use the same CTA-group mode.
#[inline(always)]
pub fn relinquish_alloc_permit_cg2() {
    cuda_device::tcgen05::tcgen05_relinquish_alloc_permit_cg2()
}

/// Commit group-2 MMA completion to the selected CTAs' local barrier offsets.
///
/// # Safety
/// The MMA issuer must commit its own operations. Every selected CTA must be
/// active with a matching initialized barrier expecting the completion arrival.
/// Keep all associated resources live until the barrier signals completion.
#[inline(always)]
pub unsafe fn commit_multicast_cg2(
    barrier: super::memory::SharedAddress<'_, cuda_device::barrier::Barrier>,
    mask: super::cluster::CtaMask<super::groups::CtaGroup2>,
) {
    unsafe {
        cuda_device::tcgen05::tcgen05_commit_multicast_cg2(barrier.as_raw().cast(), mask.bits())
    }
}
