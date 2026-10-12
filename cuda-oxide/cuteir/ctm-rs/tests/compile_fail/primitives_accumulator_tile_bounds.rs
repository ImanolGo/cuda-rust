// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use ctm_rs::mma::AccumulatorTile16x8;

const TILE: AccumulatorTile16x8<2, 8> = AccumulatorTile16x8::zero();
const OUTSIDE: ctm_rs::mma::Accumulator16x8 = TILE.fragment::<2, 0>();

fn main() {
    let _ = OUTSIDE;
}
