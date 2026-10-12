// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::OperandB;
use cute_rs::sm100_attention::{SharedLayout, SharedTileArray};
type L = SharedLayout<8, 32>;
const EMPTY: SharedTileArray<L, OperandB, 0> =
    unsafe { SharedTileArray::from_raw(core::ptr::null_mut()) };
const OVERFLOW: SharedTileArray<L, OperandB, { u32::MAX }> =
    unsafe { SharedTileArray::from_raw(core::ptr::null_mut()) };
fn main() {
    let _ = (EMPTY, OVERFLOW);
}
