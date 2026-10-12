/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::barrier::Barrier;
use cute_rs::sm100_attention::pipeline::WaitPolicy;

#[derive(Clone, Copy)]
struct Unordered;

impl WaitPolicy for Unordered {
    unsafe fn wait(_: *const Barrier, _: u32) {}
}

fn main() {}
