/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![feature(f16)]
use cuda_device::primitives::{memory::*, tma::TensorMap};

#[test]
fn physical_formats_and_tensor_map_preserve_kernel_abi() {
    assert_eq!(size_of::<TensorMap<f16, F16Tile128>>(), 128);
    assert_eq!(align_of::<TensorMap<f16, F16Tile128>>(), 64);
    assert_eq!(F16Tile128::FOOTPRINT_BYTES, 16384);
    assert_eq!(F16TileB96::FOOTPRINT_BYTES, 12288);
    assert_eq!(F16TileB80::FOOTPRINT_BYTES, 10240);
    assert_eq!(F16TileC::FOOTPRINT_BYTES, 8192);
    assert_eq!(F16Tile128::BASE_ALIGNMENT, 1024);
    assert_eq!(F16TileC::BASE_ALIGNMENT, 512);
    fn copy<T: Copy>() {}
    copy::<TensorMap<f16, F16Tile128>>();
}

#[test]
fn shared_region_rejects_invalid_storage_before_use() {
    // These invalid layouts return Err before any address can be used.
    assert!(
        unsafe { SharedRegion::<F16Tile128>::from_raw_parts(core::ptr::null_mut(), 16384) }
            .is_err()
    );
    assert!(
        unsafe {
            SharedRegion::<F16Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1002),
                16384,
            )
        }
        .is_err()
    );
    assert!(
        unsafe {
            SharedRegion::<F16Tile128>::from_raw_parts(
                core::ptr::without_provenance_mut(0x1000),
                16383,
            )
        }
        .is_err()
    );
}

#[test]
fn multicast_masks_and_signed_coordinates_preserve_instruction_domains() {
    use cuda_device::primitives::{cluster::CtaMask, groups::CtaGroup2, tma::Coordinates2D};
    assert!(CtaMask::<CtaGroup2>::try_from_bits(0).is_err());
    assert_eq!(
        CtaMask::<CtaGroup2>::try_from_bits(0b1111).unwrap().bits(),
        0b1111
    );
    assert_eq!(Coordinates2D::new(-32, 4096).inner(), -32);
    assert_eq!(Coordinates2D::new(-32, 4096).outer(), 4096);
}
