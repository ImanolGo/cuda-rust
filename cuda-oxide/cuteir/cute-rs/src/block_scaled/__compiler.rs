/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exact compiler boundaries for block-scaled GEMV.

use super::*;
use crate::numeric::{E2M1, UE8M0, UE8M0x2};
use crate::{Tensor, assume_div, load_tile};
use core::marker::PhantomData;

#[inline(never)]
/// Compiler boundary constructing packed value and canonical scale views for K16 scaling.
pub(super) const fn block_scaled_make<'a, Mode: Copy>(
    values: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    k: usize,
) -> BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mode>> {
    let scale_layout = Sm1xxBlockScaleKMajor {
        rest_m: rows.div_ceil(CANONICAL_SCALE_M),
        rest_k: k.div_ceil(16 * CANONICAL_SCALE_K_GROUPS),
    };
    BlockScaledTensor {
        values: Tensor::from_storage(
            values,
            KMajor {
                rows,
                k,
                mode: PhantomData,
            },
        ),
        scales: Tensor::from_storage(scales, scale_layout),
    }
}

#[inline(never)]
/// Compiler boundary selecting a batch row and computing its value/scale byte bases.
pub(super) fn block_scaled_thread_row<'a, Mode: Copy>(
    tensor: BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mode>>,
    batch: usize,
    row: usize,
) -> BlockScaledThreadRow<'a, Mode> {
    let packed_k = tensor.values.layout.k / 2;
    let value_row_base = (batch * tensor.values.layout.rows + row) * packed_k;
    let scale_row_base = sm1xx_scale_offset!(tensor.scales.layout, batch, row, 0);

    BlockScaledThreadRow {
        values: tensor.values.storage,
        scales: tensor.scales.storage,
        value_row_base,
        scale_row_base,
        mode: PhantomData,
    }
}

#[inline(never)]
/// Compiler boundary selecting one K64 tile: 32 packed bytes and four scale bytes.
pub(super) fn block_scaled_k_tile<'a, Mode: Copy>(
    row: BlockScaledThreadRow<'a, Mode>,
    tile: usize,
) -> BlockScaledTile64<'a, Mode> {
    BlockScaledTile64 {
        values: row.values,
        scales: row.scales,
        value_base: row.value_row_base + tile * 32,
        scale_base: row.scale_row_base + tile * CANONICAL_SCALE_ATOM_BYTES,
        mode: PhantomData,
    }
}

#[inline(never)]
/// Load one K64 tile and decode its four scales.
///
/// # Safety
/// The value and scale ranges must be live, in bounds and aligned as required by the tile’s public load operation.
pub(super) unsafe fn block_scaled_load_k64<'a, Mode: Copy>(
    tile: BlockScaledTile64<'a, Mode>,
) -> LoadedBlockScaledTile64<Mode> {
    // State each alignment fact immediately before its load. The device
    // compiler reads that direct link; putting the value in a struct would
    // hide it.
    let value_base = unsafe { assume_div::<16>(tile.value_base) };
    let scale_base = unsafe { assume_div::<4>(tile.scale_base) };
    let values_lo = unsafe { load_tile::<u8, 16>(tile.values, value_base) };
    let values_hi = unsafe { load_tile::<u8, 16>(tile.values, value_base + 16) };
    let scale_bytes = unsafe { load_tile::<u8, 4>(tile.scales, scale_base) };

    let (scale0, scale1) = UE8M0x2::from_bytes(scale_bytes[0], scale_bytes[1]).to_f32x2();
    let (scale2, scale3) = UE8M0x2::from_bytes(scale_bytes[2], scale_bytes[3]).to_f32x2();

    LoadedBlockScaledTile64 {
        values_lo,
        values_hi,
        scale0,
        scale1,
        scale2,
        scale3,
        mode: PhantomData,
    }
}

#[inline(never)]
/// Accumulate all 64 scaled FP4 products in fixed K-group order.
pub(super) fn block_scaled_dot_k64(
    lhs: LoadedBlockScaledTile64<Mkl>,
    rhs: LoadedBlockScaledTile64<Nkl>,
    mut acc: f32,
) -> f32 {
    /// Expand one K16 scale group into scalar FP4-pair products in fixed order.
    macro_rules! accumulate_group {
        ($group:literal; $($pair:literal),+ $(,)?) => {{
            let scale_a = lhs.scale::<$group>();
            let scale_b = rhs.scale::<$group>();
            $(
                {
                    let (a_low, a_high) = lhs.value_pair::<$pair>().to_f32x2();
                    let (b_low, b_high) = rhs.value_pair::<$pair>().to_f32x2();
                    acc += ((a_low * scale_a) * b_low) * scale_b;
                    acc += ((a_high * scale_a) * b_high) * scale_b;
                }
            )+
        }};
    }

    accumulate_group!(0; 0, 1, 2, 3, 4, 5, 6, 7);
    accumulate_group!(1; 8, 9, 10, 11, 12, 13, 14, 15);
    accumulate_group!(2; 16, 17, 18, 19, 20, 21, 22, 23);
    accumulate_group!(3; 24, 25, 26, 27, 28, 29, 30, 31);
    acc
}
