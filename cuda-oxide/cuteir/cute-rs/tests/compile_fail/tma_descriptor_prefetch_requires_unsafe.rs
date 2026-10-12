/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::TmaDesc;

fn prefetch<T, Layout>(descriptor: &TmaDesc<T, Layout>) {
    descriptor.prefetch();
}

fn main() {}
