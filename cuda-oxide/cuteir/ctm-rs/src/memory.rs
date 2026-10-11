/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Instruction-visible storage formats, without tensor layout algebra.

use core::{cell::UnsafeCell, marker::PhantomData};
use cuda_device::SharedArray;

mod sealed {
    pub trait Format {}
}

/// Physical storage contract implemented only for reviewed instruction formats.
pub trait PhysicalFormat: sealed::Format {
    type Element;
    const ROWS: usize;
    const COLS: usize;
    const FOOTPRINT_BYTES: usize;
    const BASE_ALIGNMENT: usize;
    /// Swizzle span in bytes, or zero for unswizzled storage.
    const SWIZZLE_BYTES: usize;
}

pub enum Swizzle128 {}
pub enum Swizzle64 {}
pub struct Swizzled2D<const ROWS: usize, const COLS: usize, Swizzle>(PhantomData<Swizzle>);
pub type F16Tile128 = Swizzled2D<128, 64, Swizzle128>;
pub type F16TileB96 = Swizzled2D<96, 64, Swizzle128>;
pub type F16TileB80 = Swizzled2D<80, 64, Swizzle128>;
pub type F16TileC = Swizzled2D<128, 32, Swizzle64>;

macro_rules! format {
    ($rows:literal, $cols:literal, $swizzle:ty, $bytes:literal, $alignment:literal) => {
        impl sealed::Format for Swizzled2D<$rows, $cols, $swizzle> {}
        impl PhysicalFormat for Swizzled2D<$rows, $cols, $swizzle> {
            type Element = f16;
            const ROWS: usize = $rows;
            const COLS: usize = $cols;
            const FOOTPRINT_BYTES: usize = $rows * $cols * 2;
            const BASE_ALIGNMENT: usize = $alignment;
            const SWIZZLE_BYTES: usize = $bytes;
        }
    };
}
format!(128, 64, Swizzle128, 128, 1024);
format!(96, 64, Swizzle128, 128, 1024);
format!(80, 64, Swizzle128, 128, 1024);
format!(128, 32, Swizzle64, 64, 512);

/// A 128-by-64 byte tile holding two packed FP4 values per byte.
///
/// The inner TMA coordinate and row width count bytes, not FP4 values.
pub enum Mxf4Tile128 {}
impl sealed::Format for Mxf4Tile128 {}
impl PhysicalFormat for Mxf4Tile128 {
    type Element = u8;
    const ROWS: usize = 128;
    const COLS: usize = 64;
    const FOOTPRINT_BYTES: usize = 8192;
    const BASE_ALIGNMENT: usize = 512;
    const SWIZZLE_BYTES: usize = 64;
}

/// One unswizzled 512-byte scale tile, transported as 256 packed `u16` elements.
///
/// The kernel is responsible for packing scale values in instruction order.
pub enum Mxf4ScaleTile {}
impl sealed::Format for Mxf4ScaleTile {}
impl PhysicalFormat for Mxf4ScaleTile {
    type Element = u16;
    const ROWS: usize = 1;
    const COLS: usize = 256;
    const FOOTPRINT_BYTES: usize = 512;
    const BASE_ALIGNMENT: usize = 128;
    const SWIZZLE_BYTES: usize = 0;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteOffset(usize);
impl ByteOffset {
    pub const fn new(bytes: usize) -> Self {
        Self(bytes)
    }
    pub const fn bytes(self) -> usize {
        self.0
    }
}

macro_rules! address {
    ($name:ident, $domain:literal) => {
        #[doc = $domain]
        #[repr(transparent)]
        pub struct $name<'s, T> {
            pointer: *mut T,
            storage: PhantomData<&'s UnsafeCell<T>>,
        }
        impl<T> Copy for $name<'_, T> {}
        impl<T> Clone for $name<'_, T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<'s, T> $name<'s, T> {
            /// Adopt an address without issuing a hardware operation.
            ///
            /// # Safety
            /// The pointer must belong to this address domain and refer to live,
            /// suitably aligned storage for `'s`. This does not establish
            /// initialization, exclusive access or synchronization.
            #[inline(always)]
            pub const unsafe fn from_raw(pointer: *mut T) -> Self {
                Self {
                    pointer,
                    storage: PhantomData,
                }
            }
            #[inline(always)]
            pub const fn as_raw(self) -> *mut T {
                self.pointer
            }
        }
    };
}
address!(
    SharedAddress,
    "An address in the current CTA's shared memory."
);
address!(
    ClusterSharedAddress,
    "A shared address mapped to a CTA within the cluster."
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryError {
    NullAddress,
    Misaligned,
    InsufficientCapacity,
    SizeOverflow,
}

/// Non-owning physical shared storage. No exclusive slice is manufactured.
pub struct SharedRegion<'s, F: PhysicalFormat> {
    address: SharedAddress<'s, F::Element>,
    format: PhantomData<F>,
}
impl<'s, F: PhysicalFormat> SharedRegion<'s, F> {
    /// Validate a region's numeric layout and adopt its storage.
    ///
    /// # Safety
    /// If validation succeeds, `base` must point to live shared memory for `'s`
    /// and cover `capacity_bytes`. Concurrent accesses and completion of
    /// asynchronous operations must be coordinated by the kernel author.
    #[inline(always)]
    pub unsafe fn from_raw_parts(
        base: *mut F::Element,
        capacity_bytes: usize,
    ) -> Result<Self, MemoryError> {
        if base.is_null() {
            return Err(MemoryError::NullAddress);
        }
        if !(base as usize).is_multiple_of(F::BASE_ALIGNMENT) {
            return Err(MemoryError::Misaligned);
        }
        if capacity_bytes < F::FOOTPRINT_BYTES {
            return Err(MemoryError::InsufficientCapacity);
        }
        Ok(Self {
            address: unsafe { SharedAddress::from_raw(base) },
            format: PhantomData,
        })
    }

    /// Validate shared storage and abort the kernel if validation fails.
    ///
    /// Checks the same null address, alignment and capacity conditions as
    /// [`Self::from_raw_parts`]. Use this when the kernel cannot recover from
    /// invalid storage. The pointer remains a pointer throughout construction,
    /// without an intermediate pointer-bearing `Result` storage carrier.
    ///
    /// # Safety
    /// If validation succeeds, `base` must point to live shared memory for `'s`
    /// and cover `capacity_bytes`. Concurrent accesses and completion of
    /// asynchronous operations must be coordinated by the kernel author.
    #[inline(always)]
    pub unsafe fn from_raw_parts_or_trap(base: *mut F::Element, capacity_bytes: usize) -> Self {
        if base.is_null()
            || !(base as usize).is_multiple_of(F::BASE_ALIGNMENT)
            || capacity_bytes < F::FOOTPRINT_BYTES
        {
            cuda_device::debug::trap();
        }
        Self {
            address: unsafe { SharedAddress::from_raw(base) },
            format: PhantomData,
        }
    }

    /// Bind a format to a canonical static shared-storage declaration.
    ///
    /// # Safety
    /// `storage` must name the actual static shared declaration, not a copied
    /// marker. The raw-storage and asynchronous-access contracts still apply.
    #[inline(always)]
    pub unsafe fn from_array<const N: usize, const A: usize>(
        storage: &'s SharedArray<F::Element, N, A>,
    ) -> Result<Self, MemoryError> {
        let capacity = N
            .checked_mul(size_of::<F::Element>())
            .ok_or(MemoryError::SizeOverflow)?;
        let pointer =
            unsafe { SharedArray::as_raw_mut_ptr(core::ptr::from_ref(storage).cast_mut()) };
        unsafe { Self::from_raw_parts(pointer, capacity) }
    }
    #[inline(always)]
    pub fn address(&self) -> SharedAddress<'s, F::Element> {
        self.address
    }
}
