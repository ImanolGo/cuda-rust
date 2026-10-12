/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, TwoCtaRing,
};

unsafe fn wrong_roles(ring: TwoCtaRing<15, AsyncCompletion>) {
    unsafe {
        ring.acquire_cursor(RingCursor::<15, Consumer>::new());
        ring.wait_cursor(RingCursor::<15, Producer>::new());
        ring.mma_full_cursor(RingCursor::<15, Consumer>::new());
        ring.mma_empty_cursor(RingCursor::<15, Producer>::new());
    }
}
fn main() {}
