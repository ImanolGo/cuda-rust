/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, AsyncCopyReuse, Consumer, LocalShared, Producer, RingCursor, TwoCtaRing,
};

unsafe fn non_tc_receivers(
    copies: TwoCtaRing<15, AsyncCompletion, AsyncCopyReuse>,
    pages: TwoCtaRing<4, LocalShared>,
) {
    let producer = RingCursor::<15, Producer>::new();
    let consumer = RingCursor::<15, Consumer>::new();
    let page_producer = RingCursor::<4, Producer>::new();
    let page_consumer = RingCursor::<4, Consumer>::new();
    unsafe {
        copies.acquire(0);
        copies.expect(0, 16);
        copies.wait(0);
        copies.mma_empty(0);
        copies.acquire_cursor(producer);
        copies.expect_cursor(producer, 16);
        copies.wait_cursor(consumer);
        copies.mma_empty_cursor(consumer);
        pages.acquire(0);
        pages.wait(0);
        pages.acquire_cursor(page_producer);
        pages.wait_cursor(page_consumer);
        let _ = pages.full_at_cursor(page_producer);
        let _ = pages.empty_at_cursor(page_consumer);
    }
}

fn main() {}
