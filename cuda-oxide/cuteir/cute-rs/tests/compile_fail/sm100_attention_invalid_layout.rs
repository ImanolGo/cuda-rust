// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cute_rs::sm100_attention::{MmaSs, SharedLayout};
type UnsupportedK16 = SharedLayout<128, 16>;
const _: MmaSs<UnsupportedK16, UnsupportedK16, 128, 128, 16, 1> = MmaSs::new();
fn main() {}
