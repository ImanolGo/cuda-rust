// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::OperandB;
use cute_rs::sm100_attention::{MmaSs, SharedLayout, SharedTileArray};
type L = SharedLayout<64, 64>;
fn main() {
    let tiles = unsafe { SharedTileArray::<L, OperandB, 15>::from_raw(core::ptr::null_mut()) };
    let tile = unsafe { tiles.tile_unchecked(1) };
    let mma = MmaSs::<L, L, 128, 128, 64, 2>::new();
    unsafe { mma.gemm(0, &tile, &tile, false) };
}
