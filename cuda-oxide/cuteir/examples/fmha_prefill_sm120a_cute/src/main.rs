/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

mod fixture;
mod kernel;
mod runner;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}
