/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::barrier::{ManagedBarrier,Uninit,TmaBarrier};
fn wrong(b:ManagedBarrier<Uninit,TmaBarrier>) { let _ = b.shared_address(); }
fn main() {}
