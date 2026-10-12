/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::pipeline::{AsyncCompletion, TwoCtaRing};

const _: TwoCtaRing<0, AsyncCompletion> = unsafe { TwoCtaRing::from_raw(core::ptr::null_mut(), 0) };

fn main() {}
