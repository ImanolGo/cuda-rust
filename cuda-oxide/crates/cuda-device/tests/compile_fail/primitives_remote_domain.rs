/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::{barrier::Barrier,primitives::{memory::SharedAddress,cluster::arrive_remote}};
fn wrong(local:SharedAddress<Barrier>) { unsafe { arrive_remote(local) }; }
fn main() {}
