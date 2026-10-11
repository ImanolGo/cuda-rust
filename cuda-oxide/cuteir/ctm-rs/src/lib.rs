/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-level CUDA primitives with statically checked operands.
//!
//! Physical formats and instruction selectors are Rust types. Election,
//! synchronization, instruction order and asynchronous completion are explicit
//! responsibilities of the kernel author, documented on unsafe operations.

#![feature(f16)]
#![no_std]

#[cfg(feature = "host")]
extern crate std;

#[cfg(feature = "host")]
pub mod host;

pub mod cluster;
pub mod groups;
pub mod mbarrier;
pub mod memory;
pub mod mma;
pub mod tcgen05;
pub mod tma;
pub mod warp;

// Keep callable APIs as forwarding functions instead of public function
// reexports. rustc may print a reexport's path as an intrinsic's name; the
// device compiler recognizes raw operations by their cuda-device paths.

/// Hardware thread/block coordinates, without inferred thread mapping.
pub mod thread {
    macro_rules! coordinate {
        ($($name:ident),* $(,)?) => {$(
            #[doc = concat!("Read the hardware coordinate through [`cuda_device::thread::", stringify!($name), "`].")]
            #[allow(non_snake_case)]
            #[inline(always)]
            pub fn $name() -> u32 {
                cuda_device::thread::$name()
            }
        )*};
    }
    coordinate!(
        blockDim_x,
        blockDim_y,
        blockDim_z,
        blockIdx_x,
        blockIdx_y,
        blockIdx_z,
        gridDim_x,
        gridDim_y,
        gridDim_z,
        threadIdx_x,
        threadIdx_y,
        threadIdx_z,
    );
}
