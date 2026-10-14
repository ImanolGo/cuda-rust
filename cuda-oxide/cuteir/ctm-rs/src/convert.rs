/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit instruction packing and numeric conversion.

/// Convert two FP32 values to packed FP16 using nearest-even rounding.
///
/// `lo` occupies bits `0..16`, and `hi` occupies bits `16..32`. This preserves
/// the argument order of `cuda_device::convert::cvt_f16x2_f32` and does not
/// introduce flush-to-zero mode or an additional conversion.
#[must_use]
#[inline(always)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    cuda_device::convert::cvt_f16x2_f32(lo, hi)
}
