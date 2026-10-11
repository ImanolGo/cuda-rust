/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#[test]
fn primitives_reject_invalid_type_contracts() {
    trybuild::TestCases::new().compile_fail("tests/compile_fail/primitives_*.rs");
}
