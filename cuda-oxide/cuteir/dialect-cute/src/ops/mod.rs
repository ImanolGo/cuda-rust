/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cute dialect ops.
//!
//! The op set is intentionally small. `cute.copy` and `cute.copy_g2s` are
//! zero-result memory ops; `cute.assume_div` is an identity value that
//! preserves one compiler fact.

/// Typed copy widths, divisibility promises, and warp matrix operand roles.
mod copy;
/// SSA identity operations preserving explicit integer divisibility promises.
mod divisibility;
/// Warp-cooperative matrix loads with typed operand roles and shared placement.
mod ldmatrix;
/// Descriptor-backed raw transport views and their direct copy contracts.
mod tma;

pub use copy::{CuteCopyG2SOp, CuteCopyOp};
pub use divisibility::CuteAssumeDivOp;
pub use ldmatrix::CuteLdmatrixOp;
pub use tma::{CuteTmaLoad2dOp, CuteTmaStore2dOp};

/// Regression fixtures covering accepted contracts and rejected malformed IR.
#[cfg(test)]
mod tests;
