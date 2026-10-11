/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::primitives::memory::{SharedAddress,ClusterSharedAddress};
fn cluster(_: ClusterSharedAddress<'_,u32>) {}
fn wrong(address: SharedAddress<'_,u32>) { cluster(address); }
fn main() {}
