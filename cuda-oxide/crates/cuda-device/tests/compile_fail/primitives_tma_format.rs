/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![feature(f16)]
use cuda_device::primitives::{memory::*,tma::*};
fn wrong(src:&SharedRegion<F16TileC>,map:&TensorMap<f16,F16Tile128>) { unsafe { store_2d(src,map,Coordinates2D::new(0,0)) }; }
fn main() {}
