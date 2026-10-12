/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cute_rs::sm100_attention::{MmaSs, MmaTs, SharedLayout, SharedTileArray, SmemLayout};
use cute_rs::{OperandA, OperandB};

fn check_bijection<L: SmemLayout>() {
    let mut offsets = Vec::new();
    for row in 0..L::ROWS {
        for col in 0..L::COLS {
            offsets.push(L::offset(row, col));
        }
    }
    offsets.sort_unstable();
    assert_eq!(
        offsets,
        (0..(L::ROWS * L::COLS) as usize).collect::<Vec<_>>()
    );
}

#[test]
fn tiled_swizzles_cover_storage_without_overlap() {
    check_bijection::<SharedLayout<128, 128>>();
    check_bijection::<SharedLayout<128, 128, true>>();
    check_bijection::<SharedLayout<64, 64>>();
    check_bijection::<SharedLayout<64, 32>>();
    check_bijection::<SharedLayout<128, 32, true>>();
}

#[test]
fn tma_contiguous_chunks_and_transposed_v_share_the_same_storage() {
    type K = SharedLayout<128, 128>;
    assert_eq!(K::offset(0, 0), 0);
    assert_eq!(K::offset(0, 64), 8192);
    assert_eq!(K::offset(1, 0), 72);
    for row in 0..32 {
        for col in 0..128 {
            assert_eq!(
                SharedLayout::<32, 128>::offset(row, col),
                SharedLayout::<128, 32, true>::offset(col, row)
            );
        }
    }
    assert_eq!(SharedLayout::<64, 32>::SWIZZLE_BITS, 2);
    assert_eq!(SharedLayout::<128, 128>::SWIZZLE_BITS, 3);
}

#[test]
fn independent_qk_and_pv_configurations_are_constructible() {
    let _ = MmaSs::<SharedLayout<128, 128>, SharedLayout<128, 128>, 128, 128, 128, 1>::new();
    let _ = MmaTs::<SharedLayout<128, 128, true>, 128, 128, 128, 1>::new();
    let _ = MmaSs::<SharedLayout<64, 64>, SharedLayout<64, 64>, 128, 128, 64, 2>::new();
    let _ = MmaSs::<SharedLayout<64, 32>, SharedLayout<128, 32, true>, 128, 256, 32, 2>::new();
}

// Host storage models the aligned allocation without accessing it as GPU memory.
#[repr(align(1024))]
struct SharedStorage([u16; 3 * 16384]);

fn check_tile_array<L: SmemLayout, Role>() {
    let mut storage = SharedStorage([0; 3 * 16384]);
    let base = storage.0.as_mut_ptr();
    // SAFETY: storage covers three complete tiles and meets either alignment.
    let tiles = unsafe { SharedTileArray::<L, Role, 3>::from_raw(base.cast()) };
    let tile_elements = (L::ROWS * L::COLS) as usize;
    let period_elements = 1usize << (L::SWIZZLE_BITS + 6);
    assert_eq!(tile_elements % period_elements, 0);
    let mut seen = vec![false; 3 * tile_elements];
    for tile_index in 0..3 {
        // SAFETY: the index is in the array's static extent.
        let tile = unsafe { tiles.tile_unchecked(tile_index) };
        assert_eq!(
            tile.as_ptr().cast::<u16>(),
            base.wrapping_add(tile_index as usize * tile_elements)
        );
        for row in 0..L::ROWS {
            for col in 0..L::COLS {
                let offset = tile_index as usize * tile_elements + L::offset(row, col);
                assert!(!seen[offset], "overlapping physical tile elements");
                seen[offset] = true;
                // The XOR swizzle is unchanged when translated by whole periods.
                let mask = (1usize << L::SWIZZLE_BITS) - 1;
                let unswizzled = offset ^ (((offset >> 6) & mask) << 3);
                let local = L::offset(row, col);
                let local_unswizzled = local ^ (((local >> 6) & mask) << 3);
                assert_eq!(
                    unswizzled,
                    tile_index as usize * tile_elements + local_unswizzled
                );
            }
        }
    }
    assert!(seen.into_iter().all(|value| value));
}

#[test]
fn complete_tile_arrays_preserve_addresses_and_swizzles() {
    check_tile_array::<SharedLayout<64, 64>, OperandB>();
    check_tile_array::<SharedLayout<64, 32>, OperandA>();
    check_tile_array::<SharedLayout<128, 32, true>, OperandB>();
    check_tile_array::<SharedLayout<128, 128>, OperandA>();
    check_tile_array::<SharedLayout<128, 128, true>, OperandB>();
    check_tile_array::<SharedLayout<8, 32>, OperandA>();
    check_tile_array::<SharedLayout<8, 64>, OperandA>();
}

#[test]
fn tile_arrays_reject_operand_role_changes_and_invalid_extents() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/compile_fail/sm100_attention_tile_array_role.rs");
    tests.compile_fail("tests/compile_fail/sm100_attention_tile_array_extent.rs");
}

// Const evaluation rejects inbounds pointer arithmetic after an allocation
// expires. A safe raw-address accessor must not require that allocation live.
const _: () = {
    let tile = {
        let mut storage = SharedStorage([0; 3 * 16384]);
        // SAFETY: two full tiles fit in this aligned, currently live storage.
        let tiles = unsafe {
            SharedTileArray::<SharedLayout<64, 64>, OperandA, 2>::from_raw(
                storage.0.as_mut_ptr().cast(),
            )
        };
        // SAFETY: index one is within the two-tile extent; no asynchronous use.
        unsafe { tiles.tile_unchecked(1) }
    };
    let _ = tile.as_ptr();
};
