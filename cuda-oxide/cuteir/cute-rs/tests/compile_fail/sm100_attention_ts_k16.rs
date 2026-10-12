// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use cute_rs::sm100_attention::{MmaTs, SharedLayout};
// MN-major shared B alone permits K16, but the TMEM-source importer does not.
type Value = SharedLayout<128, 16, true>;
const _: MmaTs<Value, 128, 128, 16, 1> = MmaTs::new();
fn main() {}
