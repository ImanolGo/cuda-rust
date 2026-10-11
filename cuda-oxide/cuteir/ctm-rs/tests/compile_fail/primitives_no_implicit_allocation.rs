/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemUninit},groups::{CtaGroup1,CtaGroup2}};
fn group1(g:TmemGuard<TmemUninit,512,CtaGroup1>) { let _ = unsafe { g.alloc() }; }
fn group2(g:TmemGuard<TmemUninit,512,CtaGroup2>) { let _ = unsafe { g.alloc() }; }
fn main() {}
