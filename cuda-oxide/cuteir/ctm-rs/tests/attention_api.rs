/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![doc = "Compile-pass coverage for attention primitives and canonical storage adapters."]
/// Compile the attention primitive probe with typed instruction operands and canonical storage.
#[test]
fn attention_primitives_accept_instruction_operands_and_canonical_storage() {
    trybuild::TestCases::new().pass("tests/compile_pass/attention_primitives.rs");
}
