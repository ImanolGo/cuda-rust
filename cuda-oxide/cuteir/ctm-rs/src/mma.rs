/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! SM120 warp-level MXFP4 instructions and role-specific register carriers.
//! Addresses, lane mapping, synchronization and issue order remain explicit.
use super::memory::SharedAddress;

macro_rules! registers {
    ($name:ident, $element:ty, $count:literal, $description:literal) => {
        #[doc = $description]
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct $name([$element; $count]);
        impl $name {
            #[inline(always)]
            pub const fn from_registers(registers: [$element; $count]) -> Self {
                Self(registers)
            }
            #[inline(always)]
            pub const fn registers(self) -> [$element; $count] {
                self.0
            }
        }
    };
}
registers!(
    Mxf4A,
    u32,
    4,
    "One lane's packed E2M1 A fragment for m16n8k64, in PTX register order."
);
registers!(
    Mxf4B,
    u32,
    2,
    "One lane's packed E2M1 B fragment for m16n8k64, in PTX register order."
);
registers!(
    Accumulator16x8,
    f32,
    4,
    "One lane's four FP32 accumulators for a warp-wide 16x8 output."
);
impl Accumulator16x8 {
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([0.0; 4])
    }
}
macro_rules! scales {
    ($name:ident, $role:literal) => {
        #[doc = $role]
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct $name(u16);
        impl $name {
            /// Low byte scales K[0..32], high byte scales K[32..64].
            #[inline(always)]
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }
            #[inline(always)]
            pub const fn bits(self) -> u16 {
                self.0
            }
        }
    };
}
scales!(ScalePairA, "Packed UE8M0 scales for operand A.");
scales!(ScalePairB, "Packed UE8M0 scales for operand B.");

/// Load four A registers using one row-layout ldmatrix.x4 instruction.
///
/// # Safety
/// All 32 lanes must participate with live, aligned shared row addresses in the
/// PTX m8n8.x4 mapping. Order writes/reads before and after this weak memory op.
#[inline(always)]
pub unsafe fn load_mxf4_a(address: SharedAddress<'_, u32>) -> Mxf4A {
    Mxf4A(unsafe { cuda_device::wmma::ldmatrix_x4(address.as_raw()) })
}
/// Load one B pair with ldmatrix.x4; separate the even/odd row submatrices.
///
/// # Safety
/// Same collective/address requirements as [`load_mxf4_a`]. Addresses must
/// select two neighboring N8 fragments of an E2M1 K64 tile.
#[inline(always)]
pub unsafe fn load_mxf4_b_pair(address: SharedAddress<'_, u32>) -> (Mxf4B, Mxf4B) {
    let r = unsafe { cuda_device::wmma::ldmatrix_x4(address.as_raw()) };
    (Mxf4B([r[0], r[2]]), Mxf4B([r[1], r[3]]))
}
/// Issue one SM120 m16n8k64 E2M1/UE8M0 block-scaled MMA with selectors zero.
/// No loads, synchronization, election or completion waits are inserted.
///
/// # Safety
/// All 32 lanes must execute uniformly on SM120a. Fragments and per-lane scale
/// providers must follow the PTX register mapping for the same K64 tile.
///
/// ```compile_fail
/// use ctm_rs::mma::*;
/// unsafe {
///     mma_mxf4(Accumulator16x8::zero(), Mxf4A::from_registers([0;4]),
///         Mxf4B::from_registers([0;2]), ScalePairB::from_bits(0), ScalePairA::from_bits(0));
/// }
/// ```
#[inline(always)]
pub unsafe fn mma_mxf4(
    c: Accumulator16x8,
    a: Mxf4A,
    b: Mxf4B,
    scale_a: ScalePairA,
    scale_b: ScalePairB,
) -> Accumulator16x8 {
    Accumulator16x8(unsafe {
        cuda_device::wmma::mma_m16n8k64_mxf4_f32(
            c.0,
            a.0,
            b.0,
            scale_a.0 as u32,
            0,
            0,
            scale_b.0 as u32,
            0,
            0,
        )
    })
}
/// Convert four accumulators to FP16 and issue one row-layout stmatrix.x2.
///
/// # Safety
/// All 32 lanes must participate. Shared row addresses must satisfy PTX x2
/// mapping and alignment, and refer to live output storage. Explicitly publish
/// the writes to the async proxy before any dependent TMA transfer.
#[inline(always)]
pub unsafe fn store_f16(address: SharedAddress<'_, u8>, c: Accumulator16x8) {
    let top = cuda_device::convert::cvt_f16x2_f32(c.0[0], c.0[1]);
    let bottom = cuda_device::convert::cvt_f16x2_f32(c.0[2], c.0[3]);
    unsafe { cuda_device::tcgen05::stmatrix_m8n8_x2(address.as_raw(), top, bottom) }
}
