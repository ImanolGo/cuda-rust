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
pub use cuda_device::{cluster_launch, cuda_module, kernel, launch_bounds, launch_contract};

pub mod async_copy;
pub mod cluster;
pub mod convert;
pub mod f32x2;
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
// Compile-time markers are reexported to preserve their defining identity.

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
    /// Request unrolling by placing this marker at the start of a counted `while` body.
    /// `FACTOR` is zero for full unrolling, or `2..=1024` for partial unrolling.
    /// This reexport preserves the defining identity used by the compiler's unroll pass.
    /// Range-based `for` loops are not recognized by the current unroll pass.
    pub use cuda_device::thread::__unroll_config;

    /// Restricts register redistribution counts to instruction immediates.
    mod sealed {
        /// Implemented only for multiples of eight in 24..=256.
        pub trait RegisterCount<const N: u32> {}
    }
    /// Sealed proof that `N` is a multiple of eight in `24..=256`.
    pub trait ValidRegisterCount<const N: u32>: sealed::RegisterCount<N> {}
    /// Admit the register-budget immediates supported by setmaxnreg.
    macro_rules! register_counts {
        ($($n:literal),* $(,)?) => {$(
            impl sealed::RegisterCount<$n> for () {}
            impl ValidRegisterCount<$n> for () {}
        )*};
    }
    register_counts!(
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
        184, 192, 200, 208, 216, 224, 232, 240, 248, 256,
    );

    /// Increase the warpgroup's register budget to `N` (sm_90a or supported newer profile).
    /// `N` is an instruction immediate: a multiple of eight in `24..=256`.
    ///
    /// # Safety
    /// Every warpgroup thread must execute uniformly with the same count. The
    /// increase must obey the register redistribution protocol and launch budget.
    #[inline(always)]
    pub unsafe fn setmaxnreg_inc<const N: u32>()
    where
        (): ValidRegisterCount<N>,
    {
        unsafe { cuda_device::thread::setmaxnreg_inc::<N>() }
    }

    /// Decrease the warpgroup's register budget to `N` (sm_90a or supported newer profile).
    /// `N` is an instruction immediate: a multiple of eight in `24..=256`.
    ///
    /// # Safety
    /// Every warpgroup thread must execute uniformly with the same count. The
    /// decrease must obey the register redistribution protocol and launch budget.
    #[inline(always)]
    pub unsafe fn setmaxnreg_dec<const N: u32>()
    where
        (): ValidRegisterCount<N>,
    {
        unsafe { cuda_device::thread::setmaxnreg_dec::<N>() }
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
