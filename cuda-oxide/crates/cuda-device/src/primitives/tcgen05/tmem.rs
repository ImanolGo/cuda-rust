/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Borrowed TMEM views and explicit collective allocation instructions.
use super::super::groups::CtaGroup;
use crate::tcgen05::{TmemDeallocated, TmemGuard, TmemReady, TmemUninit};
use core::marker::PhantomData;

mod sealed {
    pub trait Columns<const N: u32> {}
    pub trait Format {}
    pub trait Operand {}
}
pub trait ValidColumns<const N: u32>: sealed::Columns<N> {}
macro_rules! columns { ($($n:literal),*) => { $(impl sealed::Columns<$n> for () {} impl ValidColumns<$n> for () {})* }; }
columns!(32, 64, 128, 256, 512);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnOffset(u32);
impl ColumnOffset {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}
pub struct TmemF32<const M: u32, const N: u32>;
pub struct TmemReadF32<const ROWS: u32, const COLS: u32>;
pub trait TmemFormat: sealed::Format {
    const COLUMNS: u32;
    const LOCAL_ROWS: u32;
}
pub trait MmaTmemFormat: TmemFormat {}
macro_rules! mma_format {
    ($n:literal) => {
        impl sealed::Format for TmemF32<256, $n> {}
        impl TmemFormat for TmemF32<256, $n> {
            const COLUMNS: u32 = $n;
            const LOCAL_ROWS: u32 = 128;
        }
        impl MmaTmemFormat for TmemF32<256, $n> {}
    };
}
mma_format!(192);
mma_format!(160);
impl sealed::Format for TmemReadF32<32, 32> {}
impl TmemFormat for TmemReadF32<32, 32> {
    const COLUMNS: u32 = 32;
    const LOCAL_ROWS: u32 = 32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TmemRangeError {
    Columns,
    Rows,
    AddressOverflow,
}

#[inline(always)]
fn pack_address<const N: u32>(
    base: u32,
    row: u32,
    column: ColumnOffset,
    width: u32,
    rows: u32,
) -> Result<u32, TmemRangeError> {
    let end = column.0.checked_add(width).ok_or(TmemRangeError::Columns)?;
    if end > N {
        return Err(TmemRangeError::Columns);
    }
    if row % 32 != 0 || row.checked_add(rows).ok_or(TmemRangeError::Rows)? > 128 {
        return Err(TmemRangeError::Rows);
    }
    if (base & 0xffff)
        .checked_add(end)
        .ok_or(TmemRangeError::AddressOverflow)?
        > 0x10000
    {
        return Err(TmemRangeError::AddressOverflow);
    }
    base.checked_add((row << 16) | column.0)
        .ok_or(TmemRangeError::AddressOverflow)
}

/// A local borrow of a live allocation; other GPU participants still require coordination.
pub struct TmemView<'g, F: TmemFormat, const N: u32, G: CtaGroup> {
    address: u32,
    guard: PhantomData<&'g TmemGuard<TmemReady, N, G>>,
    format: PhantomData<F>,
}
pub trait TmemOperand: sealed::Operand {
    type Group: CtaGroup;
    type Format: TmemFormat;
    /// Escape to the packed instruction address.
    ///
    /// # Safety
    /// The address may only be used while this view and its guard remain live;
    /// callers must also enforce cross-thread lifetime and completion rules.
    unsafe fn raw_address(&self) -> u32;
}
impl<F: TmemFormat, const N: u32, G: CtaGroup> sealed::Operand for TmemView<'_, F, N, G> {}
impl<F: TmemFormat, const N: u32, G: CtaGroup> TmemOperand for TmemView<'_, F, N, G> {
    type Group = G;
    type Format = F;
    #[inline(always)]
    unsafe fn raw_address(&self) -> u32 {
        self.address
    }
}

impl<const N: u32, G: CtaGroup> TmemGuard<TmemUninit, N, G>
where
    (): ValidColumns<N>,
{
    /// Adopt the shared-memory address slot without allocating or synchronizing.
    ///
    /// # Safety
    /// `slot` must be aligned live shared storage through allocation and release.
    #[inline(always)]
    pub unsafe fn from_raw_slot(slot: *mut u32) -> Self {
        Self {
            smem_ptr: slot,
            _state: PhantomData,
        }
    }
    /// Issue allocation without election, fences or barriers.
    ///
    /// # Safety
    /// One complete warp per required CTA must issue uniformly. All tcgen05
    /// operations must use the same group mode. Synchronize before other users
    /// adopt the resulting allocation or read its shared address slot.
    #[inline(always)]
    pub unsafe fn alloc_collective(self) -> TmemGuard<TmemReady, N, G> {
        unsafe { G::alloc(self.smem_ptr, N) };
        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
}
impl<const N: u32, G: CtaGroup> TmemGuard<TmemReady, N, G>
where
    (): ValidColumns<N>,
{
    /// Adopt another participant's allocation after caller-provided synchronization.
    ///
    /// # Safety
    /// The shared slot and its group-`G` allocation of `N` columns must remain
    /// valid until all views and asynchronous accesses complete. This handle
    /// does not establish global ownership or permission to deallocate.
    #[inline(always)]
    pub unsafe fn adopt_ready(slot: *mut u32) -> Self {
        Self {
            smem_ptr: slot,
            _state: PhantomData,
        }
    }
    /// Issue deallocation without election, fences or barriers.
    ///
    /// # Safety
    /// All users and asynchronous operations must be finished. The required
    /// complete warps must participate uniformly, exactly once per allocation.
    #[inline(always)]
    pub unsafe fn dealloc_collective(self) -> TmemGuard<TmemDeallocated, N, G> {
        unsafe { G::dealloc(self.raw_address(), N) };
        TmemGuard {
            smem_ptr: self.smem_ptr,
            _state: PhantomData,
        }
    }
    #[inline(always)]
    pub fn view<F: MmaTmemFormat>(
        &self,
        column: ColumnOffset,
    ) -> Result<TmemView<'_, F, N, G>, TmemRangeError> {
        Ok(TmemView {
            address: pack_address::<N>(self.raw_address(), 0, column, F::COLUMNS, F::LOCAL_ROWS)?,
            guard: PhantomData,
            format: PhantomData,
        })
    }
    #[inline(always)]
    pub fn read_32x32(
        &self,
        local_row: u32,
        column: ColumnOffset,
    ) -> Result<TmemView<'_, TmemReadF32<32, 32>, N, G>, TmemRangeError> {
        Ok(TmemView {
            address: pack_address::<N>(self.raw_address(), local_row, column, 32, 32)?,
            guard: PhantomData,
            format: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_tmem_ranges_preserve_allocation_bits() {
        assert_eq!(
            pack_address::<512>(0x0200_0000, 0, ColumnOffset::new(0), 192, 128),
            Ok(0x0200_0000)
        );
        assert_eq!(
            pack_address::<512>(0x0200_0000, 0, ColumnOffset::new(192), 160, 128),
            Ok(0x0200_00c0)
        );
        for row in [0, 32, 64, 96] {
            assert_eq!(
                pack_address::<512>(0x0200_0000, row, ColumnOffset::new(192), 32, 32),
                Ok(0x0200_00c0 | row << 16)
            );
        }
        assert!(pack_address::<512>(0, 128, ColumnOffset::new(0), 32, 32).is_err());
        assert!(pack_address::<512>(0, 1, ColumnOffset::new(0), 32, 32).is_err());
        assert!(pack_address::<256>(0, 0, ColumnOffset::new(192), 160, 128).is_err());
        assert!(pack_address::<512>(0, 0, ColumnOffset::new(u32::MAX), 32, 32).is_err());
        assert!(pack_address::<512>(0x0000_fff0, 0, ColumnOffset::new(32), 32, 32).is_err());
        assert!(pack_address::<512>(0xffff_0000, 32, ColumnOffset::new(0), 32, 32).is_err());
    }
}
