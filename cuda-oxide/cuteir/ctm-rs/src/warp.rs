/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit lane identity, election and register exchange.
//! No other primitive elects implicitly.

/// Exchange an FP32 register with the lane selected by XORing `lane_mask`.
///
/// This forwards the full-warp shuffle with participation mask `u32::MAX`.
/// All 32 lanes must participate with the same lane mask; no reduction or
/// memory synchronization is inserted.
#[inline(always)]
pub fn shuffle_xor_f32(value: f32, lane_mask: u32) -> f32 {
    cuda_device::warp::shuffle_xor_f32(value, lane_mask)
}

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
