/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::pipeline::{
    AsyncCompletion, AsyncCopyReuse, Consumer, LocalShared, RingCursor, TwoCtaRing,
};

unsafe fn wrong_non_tc_signals(
    copies: TwoCtaRing<15, AsyncCompletion, AsyncCopyReuse>,
    reuse: TwoCtaRing<2, AsyncCopyReuse>,
    pages: TwoCtaRing<4, LocalShared>,
) {
    unsafe {
        copies.thread_empty(0);
        reuse.expect(0, 16);
        pages.expect(0, 16);
        pages.mma_full(0);
        pages.mma_empty_cursor(RingCursor::<4, Consumer>::new());
        pages.thread_full(0);
    }
}

fn main() {}
