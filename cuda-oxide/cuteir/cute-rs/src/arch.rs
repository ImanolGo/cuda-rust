/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit warp-level compiler facts built from supported CUDA operations.

/// Return a warp-uniform value in a form that can favor uniform-register use.
///
/// This matches CuTe's `arch.make_warp_uniform`: a full-warp shuffle broadcasts
/// lane zero's value. Pass the returned value to subsequent calculations. The
/// compiler may then keep those calculations in uniform registers; this is an
/// optimization opportunity, not a register-allocation guarantee.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently with equal
/// `value` arguments. Do not call from an elected-lane or partially active
/// branch. This is an identity operation only under that uniformity promise;
/// it does not check or repair a divergent value.
#[inline(always)]
pub unsafe fn assume_warp_uniform(value: u32) -> u32 {
    cuda_device::warp::shuffle_sync(u32::MAX, value, 0)
}

/// Sum one FP32 contribution from each of the 32 lanes of a warp.
///
/// Every lane receives the sum. Uses the existing CUDA butterfly reduction,
/// with FP32 additions in XOR-offset order 16, 8, 4, 2, 1. The result need not
/// equal a sequential sum. No shared memory or block barrier is required.
/// Inactive logical elements must contribute zero while their lanes still
/// participate. This is a device operation, not a host reduction.
///
/// # Safety
/// All 32 lanes must be active and execute this call convergently. The block
/// must contain complete warps. Rust's type system cannot prove convergence;
/// do not invoke this in a lane-dependent branch or after a partial-warp exit.
#[must_use]
#[inline(always)]
pub unsafe fn warp_sum_f32(value: f32) -> f32 {
    cuda_device::warp::reduce_sum_f32(value)
}
