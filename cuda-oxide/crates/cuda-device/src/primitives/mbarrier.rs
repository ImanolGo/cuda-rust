/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Canonical barrier states with explicit instruction-only lifecycle methods.
pub use crate::barrier::{
    Barrier, GeneralBarrier, Invalidated, ManagedBarrier, Ready, TmaBarrier, Uninit,
};
