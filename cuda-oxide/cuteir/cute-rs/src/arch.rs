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
    ctm_rs::warp::shuffle_sync(u32::MAX, value, 0)
}
