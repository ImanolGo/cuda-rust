/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-level CUDA primitives with statically checked operands.
//!
//! Physical formats and instruction selectors are Rust types. Election,
//! synchronization, instruction order and asynchronous completion are explicit
//! responsibilities of the kernel author, documented on unsafe operations.
//!
//! Register carriers can be constructed without issuing an instruction:
//!
//! ```
//! use ctm_rs::mma::AccumulatorTile16x8;
//! let tile = AccumulatorTile16x8::<2, 8>::zero(); // 2 × 8 fragments per lane.
//! assert_eq!(tile.fragment::<1, 7>().registers(), [0.0; 4]);
//! ```
//!
//! Setup and examples are in `cuda-oxide/cuteir/ctm-rs/README.md`; its
//! `docs/primitives-coverage.md` lists supported instruction contracts.
//! Enable `host` for `host::tma` descriptor encoding.
//! Users of the reexported kernel macros must also depend directly on
//! `cuda-device`: their expansions refer to `::cuda_device`.

#![feature(f16)]
#![no_std]

#[cfg(feature = "host")]
extern crate std;

#[cfg(feature = "host")]
pub mod host;

/// Kernel macros using cuda-device's launch ABI; require a direct `cuda-device` dependency.
pub use cuda_device::{cuda_module, kernel, launch_bounds, launch_contract};

pub mod async_copy;
pub mod cluster;
pub mod convert;
pub mod float;
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
    /// Generate named forwarding functions while preserving the intrinsic’s defining path.
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
