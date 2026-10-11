/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-level CUDA primitives with statically checked operands.
//!
//! Physical formats and instruction selectors are Rust types. Election,
//! synchronization, instruction order and asynchronous completion are explicit
//! responsibilities of the kernel author, documented on unsafe operations.

pub mod cluster;
pub mod groups;
pub mod mbarrier;
pub mod memory;
pub mod tcgen05;
pub mod tma;
pub mod warp;

/// Hardware thread/block coordinates, without inferred thread mapping.
pub mod thread {
    pub use crate::thread::{
        blockDim_x, blockDim_y, blockDim_z, blockIdx_x, blockIdx_y, blockIdx_z, gridDim_x,
        gridDim_y, gridDim_z, threadIdx_x, threadIdx_y, threadIdx_z,
    };
}
