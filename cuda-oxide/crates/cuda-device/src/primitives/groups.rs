/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Static CTA-group selectors. They do not elect issuing threads.
mod sealed {
    pub trait Group {}
}
pub enum CtaGroup1 {}
pub enum CtaGroup2 {}
/// Reviewed hardware CTA-group modes.
pub trait CtaGroup: sealed::Group {
    #[doc(hidden)]
    /// # Safety
    /// A complete issuing warp in each required CTA must participate uniformly.
    unsafe fn alloc(slot: *mut u32, columns: u32);
    #[doc(hidden)]
    /// # Safety
    /// All accesses must be complete and the required warps must participate.
    unsafe fn dealloc(address: u32, columns: u32);
}
macro_rules! group {
    ($g:ty,$alloc:ident,$dealloc:ident) => {
        impl sealed::Group for $g {}
        impl CtaGroup for $g {
            #[inline(always)]
            unsafe fn alloc(slot: *mut u32, columns: u32) {
                unsafe { crate::tcgen05::$alloc(slot, columns) }
            }
            #[inline(always)]
            unsafe fn dealloc(address: u32, columns: u32) {
                unsafe { crate::tcgen05::$dealloc(address, columns) }
            }
        }
    };
}
group!(CtaGroup1, tcgen05_alloc, tcgen05_dealloc);
group!(CtaGroup2, tcgen05_alloc_cg2, tcgen05_dealloc_cg2);
