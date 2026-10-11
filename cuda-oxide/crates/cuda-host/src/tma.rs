/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host encoding for the explicit Rust primitives tensor-map API.

use cuda_core::sys as driver;
use cuda_device::{
    TmaDescriptor,
    primitives::{memory::PhysicalFormat, tma::TensorMap},
};
use std::{ffi::c_void, mem::MaybeUninit};

mod sealed {
    pub trait Element {}
}
/// Element types supported by the reviewed tensor-map encoder.
pub trait TmaElement: sealed::Element {
    const DRIVER_TYPE: driver::CUtensorMapDataType;
}
impl sealed::Element for f16 {}
impl TmaElement for f16 {
    const DRIVER_TYPE: driver::CUtensorMapDataType =
        driver::CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum L2Promotion {
    #[default]
    None,
    B128,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct EncodeOptions {
    pub l2_promotion: L2Promotion,
}
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("tensor-map base must be non-null and 16-byte aligned")]
    InvalidBase,
    #[error("tensor-map extents must be between 1 and 2^32 inclusive")]
    InvalidExtent,
    #[error("row stride must cover the row, be 16-byte aligned and be less than 2^40 bytes")]
    InvalidStride,
    #[error("unsupported physical tile format")]
    InvalidFormat,
    #[error("cuTensorMapEncodeTiled failed with CUDA status {0}")]
    Driver(driver::CUresult),
}

struct EncodeParams {
    dimensions: [u64; 2],
    strides: [u64; 1],
    box_dimensions: [u32; 2],
    element_strides: [u32; 2],
    swizzle: driver::CUtensorMapSwizzle,
    l2: driver::CUtensorMapL2promotion,
}

fn validate_2d<T: TmaElement, F: PhysicalFormat<Element = T>>(
    base: usize,
    rows: u64,
    cols: u64,
    ld_elements: u64,
    options: EncodeOptions,
) -> Result<EncodeParams, EncodeError> {
    if base == 0 || base % 16 != 0 {
        return Err(EncodeError::InvalidBase);
    }
    if rows == 0 || cols == 0 || rows > 1u64 << 32 || cols > 1u64 << 32 {
        return Err(EncodeError::InvalidExtent);
    }
    let stride = ld_elements
        .checked_mul(size_of::<T>() as u64)
        .ok_or(EncodeError::InvalidStride)?;
    if ld_elements < cols || stride % 16 != 0 || stride >= 1u64 << 40 {
        return Err(EncodeError::InvalidStride);
    }
    if F::ROWS == 0
        || F::ROWS > 256
        || F::COLS == 0
        || F::COLS > 256
        || F::COLS.checked_mul(size_of::<T>()) != Some(F::SWIZZLE_BYTES)
    {
        return Err(EncodeError::InvalidFormat);
    }
    let swizzle = match F::SWIZZLE_BYTES {
        64 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B,
        128 => driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
        _ => return Err(EncodeError::InvalidFormat),
    };
    Ok(EncodeParams {
        dimensions: [cols, rows],
        strides: [stride],
        box_dimensions: [F::COLS as u32, F::ROWS as u32],
        element_strides: [1, 1],
        swizzle,
        l2: match options.l2_promotion {
            L2Promotion::None => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE
            }
            L2Promotion::B128 => {
                driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B
            }
        },
    })
}

// Only a successful encoder may initialize and publish these opaque words.
fn encode_with<T: TmaElement, F: PhysicalFormat<Element = T>>(
    encode: impl FnOnce(*mut driver::CUtensorMap) -> driver::CUresult,
) -> Result<TensorMap<T, F>, EncodeError> {
    const {
        assert!(size_of::<driver::CUtensorMap>() == 128);
        assert!(align_of::<driver::CUtensorMap>() >= 128);
    }
    let mut output = MaybeUninit::<driver::CUtensorMap>::uninit();
    let result = encode(output.as_mut_ptr());
    if result != driver::cudaError_enum_CUDA_SUCCESS {
        return Err(EncodeError::Driver(result));
    }
    let output = unsafe { output.assume_init() };
    Ok(unsafe { TensorMap::from_encoded(TmaDescriptor::from_raw_words(output.opaque)) })
}

/// Encode a row-major global tensor for a reviewed physical shared format.
///
/// Coordinates use the inner (column) dimension first. Encoding does not
/// retain the allocation: it must cover the specified extents and remain live
/// until every asynchronous device operation using the returned map completes.
pub fn encode_2d<T: TmaElement, F: PhysicalFormat<Element = T>>(
    global_base: *mut c_void,
    rows: u64,
    cols: u64,
    ld_elements: u64,
    options: EncodeOptions,
) -> Result<TensorMap<T, F>, EncodeError> {
    let p = validate_2d::<T, F>(global_base as usize, rows, cols, ld_elements, options)?;
    encode_with(|output| unsafe {
        driver::cuTensorMapEncodeTiled(
            output,
            T::DRIVER_TYPE,
            2,
            global_base,
            p.dimensions.as_ptr(),
            p.strides.as_ptr(),
            p.box_dimensions.as_ptr(),
            p.element_strides.as_ptr(),
            driver::CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            p.swizzle,
            p.l2,
            driver::CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_device::primitives::memory::{F16Tile128, F16TileB80, F16TileB96, F16TileC};

    #[test]
    fn tma_driver_failure_does_not_adopt_uninitialized_output() {
        let result = encode_with::<f16, F16Tile128>(|output| {
            assert_eq!((output as usize) % 128, 0);
            driver::cudaError_enum_CUDA_ERROR_INVALID_VALUE
        });
        assert!(matches!(
            result,
            Err(EncodeError::Driver(
                driver::cudaError_enum_CUDA_ERROR_INVALID_VALUE
            ))
        ));
    }

    #[test]
    fn tma_arguments_preserve_physical_shapes_and_options() {
        let p =
            validate_2d::<f16, F16Tile128>(0x1000, 256, 64, 64, EncodeOptions::default()).unwrap();
        assert_eq!(p.dimensions, [64, 256]);
        assert_eq!(p.strides, [128]);
        assert_eq!(p.box_dimensions, [64, 128]);
        assert_eq!(p.element_strides, [1, 1]);
        assert_eq!(
            p.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B
        );
        let b = validate_2d::<f16, F16TileB96>(
            0x1000,
            192,
            64,
            64,
            EncodeOptions {
                l2_promotion: L2Promotion::B128,
            },
        )
        .unwrap();
        assert_eq!(b.box_dimensions, [64, 96]);
        assert_eq!(
            b.l2,
            driver::CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        );
        assert_eq!(
            validate_2d::<f16, F16TileB80>(0x1000, 160, 64, 64, EncodeOptions::default())
                .unwrap()
                .box_dimensions,
            [64, 80]
        );
        let c =
            validate_2d::<f16, F16TileC>(0x1000, 128, 352, 352, EncodeOptions::default()).unwrap();
        assert_eq!(c.box_dimensions, [32, 128]);
        assert_eq!(
            c.swizzle,
            driver::CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_64B
        );
    }

    #[test]
    fn tma_rejects_invalid_dynamic_layout_before_driver_call() {
        for (base, rows, cols, pitch) in [
            (0, 256, 64, 64),
            (0x1001, 256, 64, 64),
            (0x1000, 0, 64, 64),
            (0x1000, 256, 0, 64),
            (0x1000, (1 << 32) + 1, 64, 64),
            (0x1000, 256, (1 << 32) + 1, 1 << 33),
            (0x1000, 256, 64, 63),
            (0x1000, 256, 64, 65),
            (0x1000, 256, 64, 1 << 39),
            (0x1000, 256, 64, u64::MAX),
        ] {
            assert!(
                validate_2d::<f16, F16Tile128>(base, rows, cols, pitch, EncodeOptions::default())
                    .is_err()
            );
        }
        assert!(
            validate_2d::<f16, F16Tile128>(0x1000, 1 << 32, 64, 64, EncodeOptions::default())
                .is_ok()
        );
    }
}
