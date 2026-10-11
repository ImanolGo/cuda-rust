/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed tensor maps with canonical opaque descriptor storage.
use super::memory::PhysicalFormat;
use crate::TmaDescriptor;
use core::marker::PhantomData;

/// Host-encoded tensor map matching one reviewed physical shared format.
///
/// A copy of this descriptor does not extend the global allocation's lifetime.
#[repr(transparent)]
pub struct TensorMap<T, F: PhysicalFormat<Element = T>> {
    raw: TmaDescriptor,
    marker: PhantomData<fn() -> (T, F)>,
}
impl<T, F: PhysicalFormat<Element = T>> Copy for TensorMap<T, F> {}
impl<T, F: PhysicalFormat<Element = T>> Clone for TensorMap<T, F> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, F: PhysicalFormat<Element = T>> TensorMap<T, F> {
    /// Adopt a descriptor encoded outside the typed host encoder.
    ///
    /// # Safety
    /// Bytes must describe element type `T`, rank two and the tile/swizzle
    /// specified by `F`. Underlying global storage must remain live through
    /// every asynchronous operation using this map.
    #[inline(always)]
    pub const unsafe fn from_encoded(raw: TmaDescriptor) -> Self {
        Self {
            raw,
            marker: PhantomData,
        }
    }
    #[inline(always)]
    pub const fn as_descriptor(&self) -> &TmaDescriptor {
        &self.raw
    }
}
