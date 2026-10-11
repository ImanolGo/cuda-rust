/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use ctm_rs::{tcgen05::tmem::{TmemGuard,TmemReady},groups::CtaGroup2};
fn wrong(g:TmemGuard<TmemReady,512,CtaGroup2>) { let _ = unsafe { g.dealloc_collective() }; let _ = g.raw_address(); }
fn main() {}
