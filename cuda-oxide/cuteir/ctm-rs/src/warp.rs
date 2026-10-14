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

/// Return whether every active lane's predicate is true, using the active mask.
/// Participating lanes must execute the same vote instruction convergently.
#[inline(always)]
pub fn all(predicate: bool) -> bool {
    cuda_device::warp::all(predicate)
}

/// Synchronize the lanes named by `mask` and order their memory accesses.
/// Every non-exited lane named in the mask must execute with the same mask;
/// each calling lane must be included in it.
#[inline(always)]
pub fn sync_mask(mask: u32) {
    cuda_device::warp::sync_mask(mask)
}

/// Exchange a register with the lane selected by XORing `lane_mask`.
/// All 32 lanes must participate convergently with the same lane mask.
#[inline(always)]
pub fn shuffle_xor(value: u32, lane_mask: u32) -> u32 {
    cuda_device::warp::shuffle_xor(value, lane_mask)
}

/// Exchange a register with `src_lane` among lanes named by `mask`.
/// Every non-exited lane in the mask must execute with the same mask; the
/// source lane must participate. This inserts no memory synchronization.
#[inline(always)]
pub fn shuffle_sync(mask: u32, value: u32, src_lane: u32) -> u32 {
    cuda_device::warp::shuffle_sync(mask, value, src_lane)
}
