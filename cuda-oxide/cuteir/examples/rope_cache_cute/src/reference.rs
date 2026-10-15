/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent BF16 storage arithmetic and coordinate-wise rotation oracle.
use cute_rs::Bf16;

pub fn round(value: f64) -> Bf16 {
    let value = value as f32;
    let bits = value.to_bits();
    let retained = (bits >> 16) as u16;
    if value.is_nan() {
        return Bf16::from_bits(retained | 0x0040);
    }
    let discarded = bits & 0xffff;
    let increment = discarded > 0x8000 || (discarded == 0x8000 && retained & 1 != 0);
    Bf16::from_bits(retained.wrapping_add(u16::from(increment)))
}

pub fn widen(value: Bf16) -> f64 {
    f64::from(f32::from_bits(u32::from(value.to_bits()) << 16))
}

pub fn rotate(input: &[Bf16], cos: &[Bf16], sin: &[Bf16], heads: usize, dim: usize) -> Vec<Bf16> {
    let half = dim / 2;
    input
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            let coordinate = index % dim;
            let token = index / (heads * dim);
            let frequency = coordinate % half;
            let table = token * half + frequency;
            let rotated = if coordinate < half {
                -widen(input[index + half])
            } else {
                widen(input[index - half])
            };
            let product = round(widen(x) * widen(cos[table]));
            let other = round(rotated * widen(sin[table]));
            round(widen(product) + widen(other))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turn_rotates_split_halves_not_adjacent_pairs() {
        let x = [1.0, 2.0, 3.0, 4.0].map(round);
        let y = rotate(&x, &[round(0.0); 2], &[round(1.0); 2], 1, 4);
        assert_eq!(
            y.iter().copied().map(widen).collect::<Vec<_>>(),
            [-3.0, -4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn rounding_retains_even_ties_and_special_bits() {
        assert_eq!(
            round(f64::from(f32::from_bits(0x3f80_8000))).to_bits(),
            0x3f80
        );
        assert_eq!(
            round(f64::from(f32::from_bits(0x3f81_8000))).to_bits(),
            0x3f82
        );
        assert_eq!(round(-0.0).to_bits(), 0x8000);
        assert_eq!(round(f64::INFINITY).to_bits(), 0x7f80);
        assert_ne!(round(f64::NAN).to_bits() & 0x7f, 0);
    }

    #[test]
    fn each_product_rounds_before_the_sum() {
        let x = [0x3f81, 0x3f02].map(Bf16::from_bits);
        let c = [Bf16::from_bits(0x3f35)];
        let y = rotate(&x, &c, &c, 1, 2);
        // Cast-once would produce 0x3eb5 for the first coordinate.
        assert_eq!(y[0].to_bits(), 0x3eb4);
        assert_eq!(y[1].to_bits(), 0x3f89);
    }
}
