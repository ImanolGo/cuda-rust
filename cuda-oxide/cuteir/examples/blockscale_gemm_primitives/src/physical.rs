// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Explicit physical addresses for this kernel, in bytes unless named otherwise.

pub const N_OFFSETS: [usize; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
#[inline(always)]
pub const fn scale_word(row: usize) -> usize {
    (row & 31) * 4 + ((row >> 5) & 3)
}
#[inline(always)]
pub const fn scale_offset(row: usize, group: usize, groups: usize) -> usize {
    (row / 128) * (groups / 4) * 512 + (group / 4) * 512 + scale_word(row) * 4 + group % 4
}
#[inline(always)]
pub const fn matrix_row(atom: usize, half: usize, lane: usize) -> usize {
    let sub = lane / 8;
    let row = 16 * atom + 8 * (sub % 2) + lane % 8;
    let column = 16 * half + 8 * (sub / 2);
    let plain = row * 64 + column * 2;
    plain ^ ((plain & 0x180) >> 3)
}
#[inline(always)]
pub const fn output_element(row: usize, column: usize) -> usize {
    let plain = row * 64 + column % 64;
    (column / 64) * 8192 + (plain ^ ((plain & 0x1c0) >> 3))
}
#[inline(always)]
pub const fn output_row(warp: usize, lane: usize, band: usize, slot: usize) -> usize {
    let row = 16 * (warp % 4) + 64 * band + (lane & 15);
    let column = 16 * (warp / 4) + N_OFFSETS[slot];
    output_element(row, column)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scale_atoms_cover_every_byte_once_and_keep_four_k_groups_adjacent() {
        let mut seen = [false; 512];
        for row in 0..128 {
            for group in 0..4 {
                let offset = scale_word(row) * 4 + group;
                assert!(offset < 512 && !seen[offset]);
                seen[offset] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value));
        assert_eq!((scale_word(0), scale_word(32), scale_word(1)), (0, 1, 4));
        assert_eq!(scale_offset(128, 4, 8), 1536);
    }
    #[test]
    fn ldmatrix_addresses_cover_each_k64_half_with_aligned_rows() {
        for atom in 0..8 {
            for half in 0..2 {
                let mut seen = [false; 512];
                for lane in 0..32 {
                    let offset = matrix_row(atom, half, lane);
                    assert!(offset < 8192 && offset % 16 == 0);
                    assert!(!seen[offset / 16]);
                    seen[offset / 16] = true;
                }
            }
        }
        assert_eq!(matrix_row(0, 0, 1), 64);
        assert_eq!(matrix_row(0, 1, 0), 32);
    }
    #[test]
    fn accumulator_mapping_covers_the_output_tile_exactly_once() {
        let mut seen = [false; 128 * 128];
        for warp in 0..8 {
            for lane in 0..32 {
                for band in 0..2 {
                    for slot in 0..8 {
                        for register in 0..4 {
                            let row = 16 * (warp % 4) + 64 * band + lane / 4 + 8 * (register / 2);
                            let col =
                                16 * (warp / 4) + N_OFFSETS[slot] + 2 * (lane % 4) + register % 2;
                            assert!(row < 128 && col < 128);
                            let physical = output_element(row, col);
                            assert!(physical < seen.len() && !seen[physical]);
                            seen[physical] = true;
                        }
                        // STMATRIX address-provider lanes name the start of each
                        // eight-element row, in the independently computed swizzle.
                        let row = 16 * (warp % 4) + 64 * band + (lane & 15);
                        let col = 16 * (warp / 4) + N_OFFSETS[slot];
                        assert_eq!(output_row(warp, lane, band, slot), output_element(row, col));
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|value| value));
    }
}
