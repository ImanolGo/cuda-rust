/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![feature(f16)]
use ctm_rs::{memory::F16Tile128,tma::TensorMap};
fn wrong(_: &TensorMap<f32,F16Tile128>) {}
fn main() {}
