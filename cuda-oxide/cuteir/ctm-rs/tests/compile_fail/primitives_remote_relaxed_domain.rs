/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use ctm_rs::{cluster::arrive_remote_relaxed, memory::SharedAddress};
use cuda_device::barrier::Barrier;
fn wrong(local: SharedAddress<Barrier>) {
    unsafe { arrive_remote_relaxed(local) };
}
fn main() {}
