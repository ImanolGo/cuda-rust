// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![feature(f16)]
use cute_rs::sm100_attention::{SharedLayout, SharedTile, TmaCopy};
use cute_rs::{OperandB, TmaDesc};
type Qk = SharedLayout<128, 128>;
type Pv = SharedLayout<128, 128, true>;
fn mismatch(desc: &TmaDesc<f16, Qk>, tile: &mut SharedTile<Pv, OperandB>) {
    unsafe { TmaCopy::<Qk, 1>::load(desc, tile, 0, 0, core::ptr::null_mut(), 0) };
}
fn main() {}
