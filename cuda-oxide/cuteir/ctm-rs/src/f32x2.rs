/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! SM100 packed two-lane FP32 register arithmetic.
//!
//! Values retain the canonical `u64` register carrier. Pack/unpack preserve
//! scalar bits; arithmetic requires an SM100 profile supporting native f32x2.

/// Pack two FP32 scalar bit patterns into one register pair.
#[must_use]
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> u64 {
    cuda_device::f32x2::pack_f32x2(lo, hi)
}

/// Unpack the low FP32 lane without changing its bits.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_lo(value: u64) -> f32 {
    cuda_device::f32x2::unpack_f32x2_lo(value)
}

/// Unpack the high FP32 lane without changing its bits.
#[must_use]
#[inline(always)]
pub fn unpack_f32x2_hi(value: u64) -> f32 {
    cuda_device::f32x2::unpack_f32x2_hi(value)
}

/// Add both FP32 lanes with round-to-nearest-even.
#[must_use]
#[inline(always)]
pub fn add_f32x2(a: u64, b: u64) -> u64 {
    cuda_device::f32x2::add_f32x2(a, b)
}

/// Multiply both FP32 lanes with round-to-nearest-even.
#[must_use]
#[inline(always)]
pub fn mul_f32x2(a: u64, b: u64) -> u64 {
    cuda_device::f32x2::mul_f32x2(a, b)
}

/// Compute fused `a * b + c` in both FP32 lanes with round-to-nearest-even.
#[must_use]
#[inline(always)]
pub fn fma_f32x2(a: u64, b: u64, c: u64) -> u64 {
    cuda_device::f32x2::fma_f32x2(a, b, c)
}
