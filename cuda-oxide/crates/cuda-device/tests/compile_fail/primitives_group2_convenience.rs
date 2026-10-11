/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::{tcgen05::{TmemGuard,TmemUninit},primitives::groups::CtaGroup2};
fn wrong(g:TmemGuard<TmemUninit,512,CtaGroup2>) { let _ = unsafe { g.alloc() }; }
fn main() {}
