/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-specific approximate FP32 math.
//! The flush-to-zero variants remain distinct so kernels retain their chosen
//! treatment of subnormal values and hardware approximation behavior.

/// Compute `2^value` with `ex2.approx.f32`, without flush-to-zero mode.
#[must_use]
#[inline(always)]
pub fn ex2_approx_f32(value: f32) -> f32 {
    cuda_device::float::ex2_approx_f32(value)
}

/// Compute `2^value` with `ex2.approx.ftz.f32`, flushing subnormal values to zero.
#[must_use]
#[inline(always)]
pub fn ex2_approx_ftz_f32(value: f32) -> f32 {
    cuda_device::float::ex2_approx_ftz_f32(value)
}

/// Compute a reciprocal with `rcp.approx.ftz.f32`, flushing subnormal values to zero.
#[must_use]
#[inline(always)]
pub fn rcp_approx_ftz_f32(value: f32) -> f32 {
    cuda_device::float::rcp_approx_ftz_f32(value)
}

/// Compute `log2(value)` with `lg2.approx.f32`, without flush-to-zero mode.
#[must_use]
#[inline(always)]
pub fn lg2_approx_f32(value: f32) -> f32 {
    cuda_device::float::lg2_approx_f32(value)
}
