/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::TmaDesc;

// Prefetching metadata preserves the existing element and layout types and
// requires no extra layout trait or conversion at the call site.
unsafe fn prefetch<T, Layout>(descriptor: &TmaDesc<T, Layout>) {
    unsafe { descriptor.prefetch() };
}

fn main() {
    struct Layout;
    let _: unsafe fn(&TmaDesc<u16, Layout>) = prefetch;
    let _: unsafe fn(&TmaDesc<f32, Layout>) = prefetch;
}
