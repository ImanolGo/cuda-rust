/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Composable FP16 CuTe operations for data-center Blackwell (SM100).
//!
//! QK and PV retain separate layout and MMA contracts. Control flow, online
//! softmax, buffer phases and participating warps remain explicit in Rust.
pub mod pipeline;

use crate::{OperandA, OperandB, TmaDesc};
use core::marker::PhantomData;
use cuda_device::{Barrier, CuSimd};

/// Physical layout of a shared FP16 operand, with CuTe's tiled byte swizzle.
mod sealed {
    pub trait Layout {}
}

/// Implemented only by the layouts whose structure the compiler recognizes.
pub trait SmemLayout: sealed::Layout {
    /// Evaluating this constant rejects an unsupported physical layout.
    const VALID: ();
    /// Logical MN extent.
    const ROWS: u32;
    /// Logical K extent.
    const COLS: u32;
    /// Whether the MN mode is contiguous.
    const MN_MAJOR: bool;
    /// Contiguous FP16 values in one swizzle atom.
    const SPAN: u32;
    /// Number of XOR bits in the shared-memory swizzle.
    const SWIZZLE_BITS: u32;
    /// Map a logical coordinate into shared storage in elements.
    fn offset(row: u32, col: u32) -> usize;
}

/// Logical `(MN,K)` layout. `MN_MAJOR` selects the contiguous logical mode.
/// Contiguous dimensions wider than 64 FP16 values are tiled in 128-byte chunks.
pub struct SharedLayout<const ROWS: u32, const COLS: u32, const MN_MAJOR: bool = false>;
impl<const R: u32, const C: u32, const MN: bool> SharedLayout<R, C, MN> {
    const VALID: () = {
        let contiguous = if MN { R } else { C };
        let outer = if MN { C } else { R };
        assert!(contiguous >= 32 && contiguous.is_multiple_of(32));
        assert!(outer > 0 && outer.is_multiple_of(8));
        assert!(contiguous == 32 || contiguous.is_multiple_of(64));
    };
    /// Offset in FP16 elements from the start of this shared tile.
    #[inline(always)]
    pub fn offset(row: u32, col: u32) -> usize {
        let () = Self::VALID;
        let (inner, outer, outer_extent) = if MN { (row, col, C) } else { (col, row, R) };
        let span = Self::SPAN;
        let linear = (inner / span) * (span * outer_extent) + outer * span + inner % span;
        let mask = (1u32 << Self::SWIZZLE_BITS) - 1;
        (linear ^ (((linear >> 6) & mask) << 3)) as usize
    }
}
impl<const R: u32, const C: u32, const MN: bool> sealed::Layout for SharedLayout<R, C, MN> {}
impl<const R: u32, const C: u32, const MN: bool> SmemLayout for SharedLayout<R, C, MN> {
    const VALID: () = SharedLayout::<R, C, MN>::VALID;
    const ROWS: u32 = R;
    const COLS: u32 = C;
    const MN_MAJOR: bool = MN;
    const SPAN: u32 = if (if MN { R } else { C }) == 32 {
        32
    } else {
        64
    };
    const SWIZZLE_BITS: u32 = if Self::SPAN == 32 { 2 } else { 3 };
    #[inline(always)]
    fn offset(row: u32, col: u32) -> usize {
        Self::offset(row, col)
    }
}

/// Non-owning shared tile, typed by its physical layout and MMA operand role.
pub struct SharedTile<L, Role> {
    base: *mut f16,
    byte_offset: u32,
    marker: PhantomData<(L, Role)>,
}
impl<L: SmemLayout, Role> SharedTile<L, Role> {
    /// # Safety
    /// Storage must cover the layout and remain live through asynchronous use.
    /// Its base must be aligned to the layout's complete swizzle period (1024
    /// bytes for B128, 512 bytes for B64).
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        let () = L::VALID;
        Self {
            base,
            byte_offset: 0,
            marker: PhantomData,
        }
    }
    /// Borrow this tile's shared storage address.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut f16 {
        // Forming a raw address does not access storage or require it to remain
        // live. Dereferencing it retains the caller's lifetime obligations.
        self.base.wrapping_add((self.byte_offset / 2) as usize)
    }
}

/// Contiguous array of complete shared tiles with one layout and operand role.
///
/// Views retain the allocation base separately from their tile offset. MMA can
/// reuse the base descriptor; tensor copies address the selected physical tile.
/// Complete tiles preserve the XOR swizzle's phase at every array element.
pub struct SharedTileArray<L, Role, const TILES: u32> {
    base: *mut f16,
    marker: PhantomData<(L, Role)>,
}
impl<L: SmemLayout, Role, const TILES: u32> SharedTileArray<L, Role, TILES> {
    const TILE_BYTES: u32 = {
        let () = L::VALID;
        let elements = (L::ROWS as u64) * (L::COLS as u64);
        assert!(elements <= u32::MAX as u64 / 2);
        let bytes = elements as u32 * 2;
        assert!(bytes.is_multiple_of(1 << (L::SWIZZLE_BITS + 7)));
        bytes
    };
    const VALID: () = {
        assert!(TILES > 0);
        assert!(TILES <= u32::MAX / Self::TILE_BYTES);
    };
    /// # Safety
    /// `base` must point to live shared storage covering `TILES` consecutive
    /// complete tiles. It must be aligned to the complete swizzle period (1024
    /// bytes for B128, 512 bytes for B64). The allocation and each accessed tile
    /// must remain live through all asynchronous operations using their views.
    #[inline(always)]
    pub const unsafe fn from_raw(base: *mut f16) -> Self {
        let () = Self::VALID;
        Self {
            base,
            marker: PhantomData,
        }
    }
    /// Select a complete tile while retaining the allocation's descriptor base.
    /// # Safety
    /// `index` must be less than `TILES`. All asynchronous accesses through this
    /// view must obey the allocation lifetime and buffer-ownership protocol.
    #[inline(always)]
    pub const unsafe fn tile_unchecked(&self, index: u32) -> SharedTile<L, Role> {
        SharedTile {
            base: self.base,
            byte_offset: index * Self::TILE_BYTES,
            marker: PhantomData,
        }
    }
}

/// Independent shared/shared FP16 MMA with FP32 accumulation.
pub struct MmaSs<AL, BL, const M: u32, const N: u32, const K: u32, const GROUP: u32>(
    PhantomData<(AL, BL)>,
);
impl<AL: SmemLayout, BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32>
    MmaSs<AL, BL, M, N, K, G>
{
    const VALID: () = {
        let () = AL::VALID;
        let () = BL::VALID;
        assert!(G == 1 || G == 2);
        assert!(M == 128 && N >= 16 && N <= 256 && N.is_multiple_of(16));
        assert!(K >= 16 && K.is_multiple_of(16));
        assert!(AL::ROWS == M / G && BL::ROWS == N / G);
        assert!(AL::COLS == K && BL::COLS == K);
    };
    /// Validate the static MMA dimensions and operand layouts.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }
    /// Issue the complete K tile, with the first instruction honoring `accumulate`.
    /// # Safety
    /// One elected leader lane issues MMA after input completion. Both shared
    /// views and destination TMEM must remain live until its completion barrier.
    /// Every operation using the allocation must use the same CTA group.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        dst: u32,
        a: &SharedTile<AL, OperandA>,
        b: &SharedTile<BL, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::mma_ss::<AL, BL, M, N, K, G>(
                dst,
                a.base,
                b.base,
                accumulate,
                a.byte_offset,
                b.byte_offset,
            )
        }
    }
}
impl<AL: SmemLayout, BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> Default
    for MmaSs<AL, BL, M, N, K, G>
{
    fn default() -> Self {
        Self::new()
    }
}

/// TMEM/shared FP16 MMA, used for the probability/value product in FMHA.
/// The importer currently supports K=32 or a positive multiple of 64.
pub struct MmaTs<BL, const M: u32, const N: u32, const K: u32, const GROUP: u32>(PhantomData<BL>);
impl<BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> MmaTs<BL, M, N, K, G> {
    const VALID: () = {
        let () = BL::VALID;
        assert!(G == 1 && M == 128);
        assert!(N >= 16 && N <= 256 && N.is_multiple_of(16));
        assert!(K == 32 || (K > 0 && K.is_multiple_of(64)));
        assert!(BL::ROWS == N && BL::COLS == K);
    };
    /// Validate the static MMA dimensions and operand layouts.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID;
        Self(PhantomData)
    }
    /// # Safety
    /// An elected lane issues this after packed FP16 probability stores and
    /// shared input copies are visible to MMA. Source and destination TMEM
    /// regions must not overlap and must remain live until completion.
    #[inline(always)]
    pub unsafe fn gemm(
        &self,
        dst: u32,
        a_tmem: u32,
        b: &SharedTile<BL, OperandB>,
        accumulate: bool,
    ) {
        unsafe {
            __compiler::mma_ts::<BL, M, N, K, G>(dst, a_tmem, b.base, accumulate, b.byte_offset)
        }
    }
}
impl<BL: SmemLayout, const M: u32, const N: u32, const K: u32, const G: u32> Default
    for MmaTs<BL, M, N, K, G>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Warp-collective 32-data-path TMEM register fragment, one row per lane.
pub struct TmemFragment<const REGS: usize> {
    /// This lane's FP32 register values.
    pub values: [f32; REGS],
}
macro_rules! fragment {
    ($n:literal,$load:ident,$store:ident) => {
        impl TmemFragment<$n> {
            /// Load a fragment and wait for the warp's preceding TMEM loads.
            /// # Safety
            /// All 32 lanes must execute with the same live, row-group-aligned
            /// TMEM address after the producer's completion and thread fence.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                cuda_device::tcgen05::tcgen05_load_wait();
                result
            }
            /// Load without a completion wait, retaining register dependencies.
            /// # Safety
            /// Same participation and producer obligations as `load`. Only true
            /// register dependencies order subsequent operations. The caller
            /// must wait before memory anti-dependencies or a consumer-release
            /// handoff, and apply the required before-thread-sync fence.
            /// See the [PTX register-dependency rule](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread).
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                let raw = unsafe { cuda_device::tcgen05::$load(address) }.to_array();
                let mut values = [0.0; $n];
                let mut i = 0;
                while i < $n {
                    cuda_device::thread::__unroll_config::<0>();
                    values[i] = f32::from_bits(raw[i]);
                    i += 1;
                }
                Self { values }
            }
            /// Store a fragment and wait for the warp's preceding TMEM stores.
            /// # Safety
            /// All 32 lanes must participate with the same live TMEM address.
            /// Complete consumer handoffs only after the returned store wait.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                cuda_device::tcgen05::tcgen05_store_wait();
            }
            /// Store without waiting, allowing disjoint TMEM chunks to overlap.
            /// # Safety
            /// Same participation and lifetime requirements as `store`. The
            /// caller must wait for stores before conflicting memory operations
            /// or completion handoffs, with the required thread-sync fence.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let mut raw = [0u32; $n];
                let mut i = 0;
                while i < $n {
                    cuda_device::thread::__unroll_config::<0>();
                    raw[i] = self.values[i].to_bits();
                    i += 1;
                }
                unsafe { Self::store_packed_deferred(address, raw) }
            }
            /// Store packed words unchanged (low FP16 lane in the low 16 bits).
            /// # Safety
            /// Same collective and lifetime requirements as `store`.
            #[inline(always)]
            pub unsafe fn store_packed(address: u32, raw: [u32; $n]) {
                unsafe { Self::store_packed_deferred(address, raw) };
                cuda_device::tcgen05::tcgen05_store_wait();
            }
            /// Store packed words without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as
            /// `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                unsafe { cuda_device::tcgen05::$store(address, CuSimd::new(raw)) };
            }
        }
    };
}
fragment!(2, tcgen05_ld_32x32b_x2_raw, tcgen05_st_32x32b_x2_raw);
fragment!(4, tcgen05_ld_32x32b_x4_raw, tcgen05_st_32x32b_x4_raw);
fragment!(8, tcgen05_ld_32x32b_x8_raw, tcgen05_st_32x32b_x8_raw);
fragment!(16, tcgen05_ld_32x32b_x16_raw, tcgen05_st_32x32b_x16_raw);
fragment!(32, tcgen05_ld_32x32b_x32_raw, tcgen05_st_32x32b_x32_raw);
// The attention references transfer larger logical fragments as repeated x32
// instructions. Wide native transfers pass standalone probes but fail in the
// full attention compilation context with the pinned compiler.
macro_rules! chunked_fragment {
    ($n:literal; $($offset:literal => $chunk:ident),+) => {
        impl TmemFragment<$n> {
            /// Load the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live row-group-aligned TMEM
            /// address after producer completion and the thread-sync fence.
            #[inline(always)]
            pub unsafe fn load(address: u32) -> Self {
                let result = unsafe { Self::load_deferred(address) };
                cuda_device::tcgen05::tcgen05_load_wait();
                result
            }
            /// Load x32 chunks without a completion wait.
            /// # Safety
            /// Same participation and producer requirements as `load`. Only true
            /// register dependencies provide ordering. Wait before memory
            /// anti-dependencies or a consumer-release handoff, and apply the
            /// required before-thread-sync fence.
            #[inline(always)]
            pub unsafe fn load_deferred(address: u32) -> Self {
                $(let $chunk = unsafe { cuda_device::tcgen05::tcgen05_ld_32x32b_x32_raw(address + $offset) };)+
                let mut values = [0.0; $n];
                $(
                    let raw = $chunk.to_array();
                    let mut i = 0;
                    while i < 32 {
                        cuda_device::thread::__unroll_config::<0>();
                        values[$offset + i] = f32::from_bits(raw[i]);
                        i += 1;
                    }
                )+
                Self { values }
            }
            /// Store the logical fragment in x32 chunks with one completion wait.
            /// # Safety
            /// All lanes participate with the same live TMEM address. Publish
            /// completion only after a before-thread-sync fence by every lane.
            #[inline(always)]
            pub unsafe fn store(&self, address: u32) {
                unsafe { self.store_deferred(address) };
                cuda_device::tcgen05::tcgen05_store_wait();
            }
            /// Store x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and lifetime requirements as `store`. Wait
            /// before conflicting memory operations or completion handoffs,
            /// with the required thread-sync fence.
            #[inline(always)]
            pub unsafe fn store_deferred(&self, address: u32) {
                let mut raw = [0; $n];
                let mut i = 0;
                while i < $n {
                    cuda_device::thread::__unroll_config::<0>();
                    raw[i] = self.values[i].to_bits();
                    i += 1;
                }
                unsafe { Self::store_packed_deferred(address, raw) }
            }
            /// Store packed FP16 words in x32 chunks, preserving every bit.
            /// # Safety
            /// Same participation, lifetime, and fence obligations as `store`.
            #[inline(always)]
            pub unsafe fn store_packed(address: u32, raw: [u32; $n]) {
                unsafe { Self::store_packed_deferred(address, raw) };
                cuda_device::tcgen05::tcgen05_store_wait();
            }
            /// Store packed x32 chunks without waiting for completion.
            /// # Safety
            /// Same participation and explicit completion obligations as
            /// `store_deferred`.
            #[inline(always)]
            pub unsafe fn store_packed_deferred(address: u32, raw: [u32; $n]) {
                $(
                    let mut chunk = [0; 32];
                    let mut i = 0;
                    while i < 32 {
                        cuda_device::thread::__unroll_config::<0>();
                        chunk[i] = raw[$offset + i];
                        i += 1;
                    }
                    unsafe { cuda_device::tcgen05::tcgen05_st_32x32b_x32_raw(address + $offset, CuSimd::new(chunk)) };
                )+
            }
        }
    }
}
chunked_fragment!(64; 0 => a, 32 => b);
chunked_fragment!(128; 0 => a, 32 => b, 64 => c, 96 => d);

/// A rank-three TMA copy into the same physical layout used by MMA.
pub struct TmaCopy<L, const GROUP: u32>(PhantomData<L>);
impl<L: SmemLayout, const G: u32> TmaCopy<L, G> {
    /// Copy a physical row-major global tile into a typed shared tile.
    /// # Safety
    /// All producer-warp lanes participate. Coordinates are in physical
    /// global rows/columns; columns must be a multiple of `L::SPAN`. The leader
    /// barrier must expect copies from the complete group. For group two,
    /// rank is the local CTA rank and completion routes to CTA zero.
    #[inline(always)]
    pub unsafe fn load<R>(
        desc: &TmaDesc<f16, L>,
        dst: &mut SharedTile<L, R>,
        row: i32,
        col: i32,
        barrier: *mut Barrier,
        rank: u32,
    ) {
        unsafe {
            __compiler::tma_load::<L, G>(
                desc as *const _,
                dst.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
                barrier,
                rank,
            )
        }
    }
    /// # Safety
    /// One elected lane issues after stores to shared memory are visible to
    /// the async proxy. Storage stays live through the bulk-copy wait.
    #[inline(always)]
    pub unsafe fn store<R>(desc: &TmaDesc<f16, L>, src: &SharedTile<L, R>, row: i32, col: i32) {
        unsafe {
            __compiler::tma_store::<L, G>(
                desc as *const _,
                src.as_ptr(),
                col % L::SPAN as i32,
                row,
                col / L::SPAN as i32,
            )
        }
    }
}

#[doc(hidden)]
pub mod __compiler {
    use super::*;
    #[inline(never)]
    pub unsafe fn mma_ss<AL, BL, const M: u32, const N: u32, const K: u32, const G: u32>(
        dst: u32,
        a: *mut f16,
        b: *mut f16,
        accumulate: bool,
        a_byte_offset: u32,
        b_byte_offset: u32,
    ) {
        let _ = (dst, a, b, accumulate, a_byte_offset, b_byte_offset);
        unreachable!("CuTe compiler marker")
    }
    #[inline(never)]
    pub unsafe fn mma_ts<BL, const M: u32, const N: u32, const K: u32, const G: u32>(
        dst: u32,
        a: u32,
        b: *mut f16,
        accumulate: bool,
        b_byte_offset: u32,
    ) {
        let _ = (dst, a, b, accumulate, b_byte_offset);
        unreachable!("CuTe compiler marker")
    }
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn tma_load<L, const G: u32>(
        desc: *const TmaDesc<f16, L>,
        dst: *mut f16,
        c0: i32,
        c1: i32,
        c2: i32,
        bar: *mut Barrier,
        rank: u32,
    ) {
        let _ = (desc, dst, c0, c1, c2, bar, rank);
        unreachable!("CuTe compiler marker")
    }
    #[inline(never)]
    pub unsafe fn tma_store<L, const G: u32>(
        desc: *const TmaDesc<f16, L>,
        src: *mut f16,
        c0: i32,
        c1: i32,
        c2: i32,
    ) {
        let _ = (desc, src, c0, c1, c2);
        unreachable!("CuTe compiler marker")
    }
}

#[cfg(feature = "host")]
mod host;
#[cfg(feature = "host")]
pub use host::make_tma_desc;
