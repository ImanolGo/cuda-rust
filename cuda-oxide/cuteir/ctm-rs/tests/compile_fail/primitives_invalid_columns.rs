/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemUninit},groups::CtaGroup2};
fn wrong(slot:*mut u32) { let _ = unsafe { TmemGuard::<TmemUninit,33,CtaGroup2>::from_raw_slot(slot) }; }
fn main() {}
