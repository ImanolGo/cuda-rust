/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Synthetic Talker/predictor composition; checkpoint comparison is a separate gate.
// Share the already validated kernels and owned-cache implementation. These
// source modules compile into this binary's own embedded device artifact.
#[path = "../../rope_cache_cute/src/attention.rs"]
mod attention;
#[path = "../../rope_cache_cute/src/attention_kernel.rs"]
mod attention_kernel;
#[path = "../../rope_cache_cute/src/attention_reference.rs"]
mod attention_reference;
mod element_kernel;
#[path = "../../ffn_cute/src/kernel.rs"]
mod ffn_kernel;
mod generation;
mod generation_validation;
#[path = "../../rope_cache_cute/src/kernel.rs"]
mod kernel;
mod layer_validation;
mod model;
#[path = "../../common/rmsnorm.rs"]
mod norm_kernel;
#[path = "../../rope_cache_cute/src/operator.rs"]
mod operator;
mod reference;
#[path = "../../rope_cache_cute/src/rotary.rs"]
mod rotary;
#[path = "../../rope_cache_cute/src/reference.rs"]
mod rotary_reference;
mod validation;

fn main() -> operator::Result<()> {
    validation::run()
}
