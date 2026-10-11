/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Explicit lane identity and election. No other primitive elects implicitly.
pub use crate::warp::{elect_sync, is_elected_sync, warp_id};
