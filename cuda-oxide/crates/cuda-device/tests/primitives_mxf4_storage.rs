/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use cuda_device::{
    barrier::Barrier,
    primitives::{
        memory::{
            F16Tile128, MemoryError, Mxf4ScaleTile, Mxf4Tile128, PhysicalFormat, SharedAddress,
            SharedRegion,
        },
        tma::{Coordinates2D, TensorMap, load_2d},
    },
};

#[test]
fn blockscaled_stage_storage_matches_instruction_footprints() {
    assert_eq!(Mxf4Tile128::FOOTPRINT_BYTES, 8192);
    assert_eq!(Mxf4Tile128::BASE_ALIGNMENT, 512);
    assert_eq!(Mxf4Tile128::SWIZZLE_BYTES, 64);
    assert_eq!(Mxf4ScaleTile::FOOTPRINT_BYTES, 512);
    assert_eq!(Mxf4ScaleTile::BASE_ALIGNMENT, 128);
    assert_eq!(Mxf4ScaleTile::SWIZZLE_BYTES, 0);
    assert_eq!(F16Tile128::FOOTPRINT_BYTES, 16384);
    assert_eq!(size_of::<TensorMap<u8, Mxf4Tile128>>(), 128);
    assert_eq!(size_of::<TensorMap<u16, Mxf4ScaleTile>>(), 128);
    assert_eq!(align_of::<TensorMap<u8, Mxf4Tile128>>(), 64);
    assert_eq!(align_of::<TensorMap<u16, Mxf4ScaleTile>>(), 64);
}

#[test]
fn blockscaled_storage_rejects_short_or_misaligned_stages() {
    unsafe {
        assert!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1080),
                512
            )
            .is_ok()
        );
        assert!(matches!(
            SharedRegion::<Mxf4Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1000),
                8191
            ),
            Err(MemoryError::InsufficientCapacity)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1010),
                8192
            ),
            Err(MemoryError::Misaligned)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1080),
                511
            ),
            Err(MemoryError::InsufficientCapacity)
        ));
        assert!(matches!(
            SharedRegion::<Mxf4ScaleTile>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1010),
                512
            ),
            Err(MemoryError::Misaligned)
        ));
    }
}

#[test]
fn ordinary_tma_load_uses_cta_local_completion_for_both_stage_types() {
    let _: unsafe fn(
        &SharedRegion<'_, Mxf4Tile128>,
        &TensorMap<u8, Mxf4Tile128>,
        Coordinates2D,
        SharedAddress<'_, Barrier>,
    ) = load_2d::<u8, Mxf4Tile128>;
    let _: unsafe fn(
        &SharedRegion<'_, Mxf4ScaleTile>,
        &TensorMap<u16, Mxf4ScaleTile>,
        Coordinates2D,
        SharedAddress<'_, Barrier>,
    ) = load_2d::<u16, Mxf4ScaleTile>;
}
