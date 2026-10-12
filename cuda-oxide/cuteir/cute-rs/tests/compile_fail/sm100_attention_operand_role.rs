// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::sm100_attention::{MmaSs, SharedLayout, SharedTile};
use cute_rs::OperandB;
type L = SharedLayout<128, 128>;
fn main() {
    let a = unsafe { SharedTile::<L, OperandB>::from_raw(core::ptr::null_mut()) };
    let b = unsafe { SharedTile::<L, OperandB>::from_raw(core::ptr::null_mut()) };
    let mma = MmaSs::<L, L, 128, 128, 128, 1>::new();
    unsafe { mma.gemm(0, &a, &b, false) };
}
