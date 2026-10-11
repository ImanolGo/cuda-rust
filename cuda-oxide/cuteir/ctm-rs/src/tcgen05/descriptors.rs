/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Typed dense FP16 descriptors using the canonical instruction encoders.
use super::tmem::{TmemF32, TmemFormat};
use crate::{
    groups::{CtaGroup, CtaGroup2},
    memory::*,
};
use core::{cell::UnsafeCell, marker::PhantomData};
use cuda_device::tcgen05::{
    Tcgen05InstructionDescriptor, Tcgen05SmemDescriptor, Tcgen05SwizzleMode,
};

mod sealed {
    pub trait Spec {}
    pub trait Input {}
    pub trait Role {}
    pub trait Step<const N: u32> {}
}
pub enum KMajor {}
pub enum OperandA {}
pub enum OperandB {}
impl sealed::Role for OperandA {}
impl sealed::Role for OperandB {}
pub trait OperandRole: sealed::Role {}
impl OperandRole for OperandA {}
impl OperandRole for OperandB {}
pub struct DenseSharedF16<G, const M: u32, const N: u32, AMajor, BMajor>(
    PhantomData<(G, AMajor, BMajor)>,
);
pub trait MmaInputFormat: PhysicalFormat<Element = f16> + sealed::Input {}
macro_rules! input { ($($f:ty),*) => { $(impl sealed::Input for $f {} impl MmaInputFormat for $f {})* }; }
input!(F16Tile128, F16TileB96, F16TileB80);
pub trait MmaSpec: sealed::Spec {
    type Group: CtaGroup;
    type AFormat: MmaInputFormat;
    type BFormat: MmaInputFormat;
    type DstFormat: TmemFormat;
    const DESCRIPTOR: Tcgen05InstructionDescriptor;
}
macro_rules! spec {
    ($n:literal,$b:ty) => {
        impl sealed::Spec for DenseSharedF16<CtaGroup2, 256, $n, KMajor, KMajor> {}
        impl MmaSpec for DenseSharedF16<CtaGroup2, 256, $n, KMajor, KMajor> {
            type Group = CtaGroup2;
            type AFormat = F16Tile128;
            type BFormat = $b;
            type DstFormat = TmemF32<256, $n>;
            const DESCRIPTOR: Tcgen05InstructionDescriptor =
                Tcgen05InstructionDescriptor::builder().m(256).n($n).build();
        }
    };
}
spec!(192, F16TileB96);
spec!(160, F16TileB80);

pub struct InstructionDescriptor<S: MmaSpec> {
    raw: Tcgen05InstructionDescriptor,
    spec: PhantomData<S>,
}
impl<S: MmaSpec> InstructionDescriptor<S> {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            raw: S::DESCRIPTOR,
            spec: PhantomData,
        }
    }
    #[inline(always)]
    pub const fn raw(&self) -> u32 {
        self.raw.raw()
    }
}
impl<S: MmaSpec> Default for InstructionDescriptor<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorError {
    Misaligned,
    AddressOutOfRange,
}
#[inline(always)]
fn encode_smem_offset(offset: ByteOffset) -> Result<Tcgen05SmemDescriptor, DescriptorError> {
    if !offset.bytes().is_multiple_of(16) {
        return Err(DescriptorError::Misaligned);
    }
    if offset.bytes() > 0x3fff0 {
        return Err(DescriptorError::AddressOutOfRange);
    }
    Ok(Tcgen05SmemDescriptor::builder()
        .address(offset.bytes() as u64)
        .leading_dim_bytes(16)
        .stride_bytes(1024)
        .swizzle(Tcgen05SwizzleMode::Swizzle128B)
        .build())
}
pub trait ValidKStep<const N: u32>: sealed::Step<N> {}
macro_rules! step { ($($n:literal),*) => { $(impl sealed::Step<$n> for () {} impl ValidKStep<$n> for () {})* }; }
step!(0, 1, 2, 3);

/// A role-specific descriptor borrowing physical shared storage.
pub struct SmemOperand<'s, R: OperandRole, F: MmaInputFormat> {
    stage: Tcgen05SmemDescriptor,
    selected: Tcgen05SmemDescriptor,
    storage: PhantomData<&'s UnsafeCell<f16>>,
    marker: PhantomData<(R, F)>,
}
impl<'s, R: OperandRole, F: MmaInputFormat> SmemOperand<'s, R, F> {
    #[inline(always)]
    pub fn from_region(region: &SharedRegion<'s, F>) -> Result<Self, DescriptorError> {
        let offset = unsafe {
            cuda_device::shared::cvta_generic_to_shared_offset(region.address().as_raw().cast())
        };
        if offset > 0x3ffff {
            return Err(DescriptorError::AddressOutOfRange);
        }
        if offset % F::BASE_ALIGNMENT as u64 != 0 {
            return Err(DescriptorError::Misaligned);
        }
        let stage = encode_smem_offset(ByteOffset::new(offset as usize))?;
        Ok(Self {
            stage,
            selected: stage,
            storage: PhantomData,
            marker: PhantomData,
        })
    }
    /// Select one K16 slice relative to the original K64 stage, even on a slice.
    #[inline(always)]
    pub fn k16<const STEP: u32>(&self) -> Self
    where
        (): ValidKStep<STEP>,
    {
        // A 1024-aligned, representable stage leaves room for all four slices
        // within the address field. Only that field changes, in 16-byte units.
        Self {
            stage: self.stage,
            selected: Tcgen05SmemDescriptor::from_raw(self.stage.raw() + STEP as u64 * 2),
            storage: PhantomData,
            marker: PhantomData,
        }
    }
    #[inline(always)]
    pub(crate) fn raw(&self) -> u64 {
        self.selected.raw()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::groups::CtaGroup2;
    #[test]
    fn instruction_and_smem_descriptors_match_canonical_bits() {
        type L = DenseSharedF16<CtaGroup2, 256, 192, KMajor, KMajor>;
        type R = DenseSharedF16<CtaGroup2, 256, 160, KMajor, KMajor>;
        assert_eq!(InstructionDescriptor::<L>::new().raw(), 0x1030_0010);
        assert_eq!(InstructionDescriptor::<R>::new().raw(), 0x1028_0010);
        assert_eq!(
            encode_smem_offset(ByteOffset::new(0)).unwrap().raw(),
            0x4000_4040_0001_0000
        );
        assert_eq!(
            encode_smem_offset(ByteOffset::new(0x1000)).unwrap().raw(),
            0x4000_4040_0001_0100
        );
        for step in 0..4 {
            assert_eq!(
                encode_smem_offset(ByteOffset::new(0x1000 + step * 32))
                    .unwrap()
                    .raw(),
                0x4000_4040_0001_0100 + step as u64 * 2
            );
        }
        assert!(encode_smem_offset(ByteOffset::new(15)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(0x40000)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(usize::MAX)).is_err());
        assert!(encode_smem_offset(ByteOffset::new(0x3fff0)).is_ok());
    }

    #[test]
    fn k16_selection_is_relative_to_original_stage() {
        let stage = encode_smem_offset(ByteOffset::new(0x1000)).unwrap();
        let a: SmemOperand<'_, OperandA, F16Tile128> = SmemOperand {
            stage,
            selected: stage,
            storage: PhantomData,
            marker: PhantomData,
        };
        assert_eq!(a.k16::<3>().k16::<1>().raw(), 0x4000_4040_0001_0102);
    }
}
