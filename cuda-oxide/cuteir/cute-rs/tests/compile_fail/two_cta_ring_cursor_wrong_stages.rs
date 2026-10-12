/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, Consumer, Producer, RingCursor, TwoCtaRing,
};

unsafe fn wrong_stages(ring: TwoCtaRing<15, AsyncCompletion>) {
    unsafe {
        ring.acquire_cursor(RingCursor::<2, Producer>::new());
        ring.wait_cursor(RingCursor::<2, Consumer>::new());
    }
}
fn main() {}
