/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit lane identity and election. No other primitive elects implicitly.

/// Elect one lane from `mask`, returning its lane ID and whether this lane won.
/// Every lane named in the mask must be converged at the call (sm_90+).
#[inline(always)]
pub fn elect_sync(mask: u32) -> (u32, bool) {
    cuda_device::warp::elect_sync(mask)
}

/// Return whether this lane is elected from `mask` (sm_90+).
/// Every lane named in the mask must be converged at the call.
#[inline(always)]
pub fn is_elected_sync(mask: u32) -> bool {
    cuda_device::warp::is_elected_sync(mask)
}

/// Return `threadIdx.x / 32`, the warp ID for a one-dimensional thread block.
#[inline(always)]
pub fn warp_id() -> u32 {
    cuda_device::warp::warp_id()
}
