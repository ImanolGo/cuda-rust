/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use ctm_rs::{groups::CtaGroup2,tcgen05::tmem::{TmemGuard,TmemReady,TmemF32,ColumnOffset,TmemOperand}};
fn consume_view(_:impl TmemOperand) {}
fn wrong(g:TmemGuard<TmemReady,512,CtaGroup2>) {
    let view = g.view::<TmemF32<256,192>>(ColumnOffset::new(0)).unwrap();
    let _dead = unsafe { g.dealloc_collective() };
    consume_view(view);
}
fn main() {}
