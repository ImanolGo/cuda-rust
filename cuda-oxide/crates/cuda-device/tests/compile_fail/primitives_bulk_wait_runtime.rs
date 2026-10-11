/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::primitives::tma::bulk_wait_group_read;
fn wrong(n:u32) { bulk_wait_group_read(n); }
fn main() {}
