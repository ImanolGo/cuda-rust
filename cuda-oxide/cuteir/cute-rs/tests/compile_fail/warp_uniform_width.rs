// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    let _ = unsafe { cute_rs::arch::assume_warp_uniform(0u64) };
}
