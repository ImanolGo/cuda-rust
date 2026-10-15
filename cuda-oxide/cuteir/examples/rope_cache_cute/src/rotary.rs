/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Checked host coefficients for split-half BF16 rotary embeddings.
use cute_rs::Bf16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeError {
    InvalidDimension,
    InvalidTheta,
    InvalidMaxPosition,
    InvalidSections,
    ExtentOverflow,
    LengthMismatch,
    PositionOutOfBounds,
    UnequalAxes,
    NonFiniteCoefficient,
}

impl core::fmt::Display for RopeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimension => "rotary head width must be positive and even",
            Self::InvalidTheta => "rotary theta must be finite and positive",
            Self::InvalidMaxPosition => "rotary position limit must be positive",
            Self::InvalidSections => "interleaved rotary sections do not fit the half-head",
            Self::ExtentOverflow => "rotary table extent is not addressable",
            Self::LengthMismatch => {
                "rotary coefficient lengths must equal tokens times half-head width"
            }
            Self::PositionOutOfBounds => "rotary position must be below its configured limit",
            Self::UnequalAxes => "standard rotary positions must agree across all three axes",
            Self::NonFiniteCoefficient => "rotary coefficients must be finite",
        })
    }
}

impl core::error::Error for RopeError {}

#[derive(Clone, Copy, Debug)]
enum Axes {
    Standard,
    Interleaved([usize; 3]),
}

/// Positional geometry, independent of hidden width and the number of heads.
#[derive(Debug)]
pub struct RotarySpec<const D: usize> {
    theta: f32,
    max_position: u32,
    axes: Axes,
}

/// One contiguous half-head per token; both rotated halves share these values.
///
/// The const head width prevents passing a table to a differently sized rotary
/// operation. Construction checks dynamic lengths and finite coefficient values.
#[derive(Debug)]
pub struct HostTables<const D: usize> {
    tokens: usize,
    cos: Vec<Bf16>,
    sin: Vec<Bf16>,
}

fn half_width<const D: usize>() -> Result<usize, RopeError> {
    if D == 0 || !D.is_multiple_of(2) {
        return Err(RopeError::InvalidDimension);
    }
    Ok(D / 2)
}

fn table_length<const D: usize>(tokens: usize) -> Result<usize, RopeError> {
    let length = tokens
        .checked_mul(half_width::<D>()?)
        .ok_or(RopeError::ExtentOverflow)?;
    let bytes = length
        .checked_mul(core::mem::size_of::<Bf16>())
        .ok_or(RopeError::ExtentOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(RopeError::ExtentOverflow);
    }
    Ok(length)
}

impl<const D: usize> RotarySpec<D> {
    /// Use one position for every frequency; unequal axis inputs are rejected.
    pub fn standard(theta: f32, max_position: u32) -> Result<Self, RopeError> {
        half_width::<D>()?;
        if !theta.is_finite() || theta <= 0.0 {
            return Err(RopeError::InvalidTheta);
        }
        if max_position == 0 {
            return Err(RopeError::InvalidMaxPosition);
        }
        Ok(Self {
            theta,
            max_position,
            axes: Axes::Standard,
        })
    }

    /// Select height/width frequencies at indices `1+3j`/`2+3j`, leaving
    /// remaining frequencies temporal. The section counts must fit exactly.
    pub fn interleaved(
        theta: f32,
        max_position: u32,
        sections: [usize; 3],
    ) -> Result<Self, RopeError> {
        let mut spec = Self::standard(theta, max_position)?;
        let count = sections.into_iter().try_fold(0usize, |sum, size| {
            sum.checked_add(size).ok_or(RopeError::ExtentOverflow)
        })?;
        if count != D / 2 {
            return Err(RopeError::InvalidSections);
        }
        for (axis, count) in sections.into_iter().enumerate().skip(1) {
            if count > 0 {
                let last = (count - 1)
                    .checked_mul(3)
                    .and_then(|index| index.checked_add(axis))
                    .ok_or(RopeError::ExtentOverflow)?;
                if last >= D / 2 {
                    return Err(RopeError::InvalidSections);
                }
            }
        }
        spec.axes = Axes::Interleaved(sections);
        Ok(spec)
    }

    /// Compute FP32 frequencies, angles and trigonometry, then round tables to
    /// BF16. Host trigonometry is not claimed bit-identical to CUDA or PyTorch.
    pub fn tables(&self, positions: &[[u32; 3]]) -> Result<HostTables<D>, RopeError> {
        let length = table_length::<D>(positions.len())?;
        for position in positions {
            if position.iter().any(|&value| value >= self.max_position) {
                return Err(RopeError::PositionOutOfBounds);
            }
            if matches!(self.axes, Axes::Standard)
                && (position[0] != position[1] || position[0] != position[2])
            {
                return Err(RopeError::UnequalAxes);
            }
        }
        let mut cos = Vec::with_capacity(length);
        let mut sin = Vec::with_capacity(length);
        for position in positions {
            for index in 0..D / 2 {
                let axis = match self.axes {
                    Axes::Standard => 0,
                    Axes::Interleaved(sections) => {
                        let candidate = index % 3;
                        if candidate != 0 && index / 3 < sections[candidate] {
                            candidate
                        } else {
                            0
                        }
                    }
                };
                let inverse_frequency = 1.0 / self.theta.powf(2.0 * index as f32 / D as f32);
                let angle = position[axis] as f32 * inverse_frequency;
                cos.push(Bf16::from_f32(angle.cos()));
                sin.push(Bf16::from_f32(angle.sin()));
            }
        }
        HostTables::from_coefficients(positions.len(), cos, sin)
    }
}

impl<const D: usize> HostTables<D> {
    /// Accept independently supplied coefficients without weakening shape or
    /// dtype checks. Coefficients need not be generated by [`RotarySpec`].
    pub fn from_coefficients(
        tokens: usize,
        cos: Vec<Bf16>,
        sin: Vec<Bf16>,
    ) -> Result<Self, RopeError> {
        let length = table_length::<D>(tokens)?;
        if cos.len() != length || sin.len() != length {
            return Err(RopeError::LengthMismatch);
        }
        if cos
            .iter()
            .chain(&sin)
            .any(|value| !value.to_f32().is_finite())
        {
            return Err(RopeError::NonFiniteCoefficient);
        }
        Ok(Self { tokens, cos, sin })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn cos(&self) -> &[Bf16] {
        &self.cos
    }

    pub fn sin(&self) -> &[Bf16] {
        &self.sin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[Bf16]) -> Vec<u16> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn standard_positions_keep_tokens_and_split_half_frequency_order() {
        let spec = RotarySpec::<4>::standard(16.0, 3).unwrap();
        let tables = spec.tables(&[[0; 3], [1; 3], [2; 3]]).unwrap();
        assert_eq!(tables.tokens(), 3);
        // Frequencies [1, 1/4]; literal BF16 encodings of angles [0,0,1,.25,2,.5].
        assert_eq!(
            bits(tables.cos()),
            [0x3f80, 0x3f80, 0x3f0a, 0x3f78, 0xbed5, 0x3f61]
        );
        assert_eq!(bits(tables.sin()), [0, 0, 0x3f57, 0x3e7d, 0x3f69, 0x3ef5]);
        assert_eq!(
            spec.tables(&[[3; 3]]).unwrap_err(),
            RopeError::PositionOutOfBounds
        );
        assert_eq!(
            spec.tables(&[[1, 2, 1]]).unwrap_err(),
            RopeError::UnequalAxes
        );
        let empty = spec.tables(&[]).unwrap();
        assert_eq!(empty.tokens(), 0);
        assert!(empty.cos().is_empty() && empty.sin().is_empty());
    }

    #[test]
    fn interleaved_selection_uses_the_literal_qwen_axis_mask() {
        let spec = RotarySpec::<128>::interleaved(1.0, 4, [24, 20, 20]).unwrap();
        let tables = spec.tables(&[[0, 1, 2], [2, 0, 1]]).unwrap();
        // Half-head indices 60..64 stay temporal; the mask is not a uniform
        // three-way cycle and does not describe adjacent rotation pairs.
        const AXIS: [usize; 64] = [
            0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
            2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
            1, 2, 0, 0, 0, 0,
        ];
        for (token, (cos, sin)) in [
            ([0x3f80, 0x3f0a, 0xbed5], [0, 0x3f57, 0x3f69]),
            ([0xbed5, 0x3f80, 0x3f0a], [0x3f69, 0, 0x3f57]),
        ]
        .into_iter()
        .enumerate()
        {
            for (i, axis) in AXIS.into_iter().enumerate() {
                assert_eq!(tables.cos()[token * 64 + i].to_bits(), cos[axis]);
                assert_eq!(tables.sin()[token * 64 + i].to_bits(), sin[axis]);
            }
        }
        for position in [[4, 0, 0], [0, 4, 0], [0, 0, 4]] {
            assert_eq!(
                spec.tables(&[position]).unwrap_err(),
                RopeError::PositionOutOfBounds
            );
        }
    }

    #[test]
    fn invalid_geometry_or_interleaved_counts_are_rejected() {
        assert_eq!(
            RotarySpec::<0>::standard(1.0, 1).unwrap_err(),
            RopeError::InvalidDimension
        );
        assert_eq!(
            RotarySpec::<3>::standard(1.0, 1).unwrap_err(),
            RopeError::InvalidDimension
        );
        for theta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                RotarySpec::<4>::standard(theta, 1).unwrap_err(),
                RopeError::InvalidTheta
            );
        }
        assert_eq!(
            RotarySpec::<4>::standard(1.0, 0).unwrap_err(),
            RopeError::InvalidMaxPosition
        );
        assert_eq!(
            RotarySpec::<128>::interleaved(1.0, 4, [23, 20, 20]).unwrap_err(),
            RopeError::InvalidSections
        );
        assert_eq!(
            RotarySpec::<8>::interleaved(1.0, 4, [1, 2, 1]).unwrap_err(),
            RopeError::InvalidSections
        );
        assert_eq!(
            RotarySpec::<8>::interleaved(1.0, 4, [1, 1, 2]).unwrap_err(),
            RopeError::InvalidSections
        );
        assert_eq!(
            RotarySpec::<128>::interleaved(1.0, 4, [usize::MAX, 1, 0]).unwrap_err(),
            RopeError::ExtentOverflow
        );
        assert!(RotarySpec::<8>::interleaved(1.0, 4, [4, 0, 0]).is_ok());
        assert!(RotarySpec::<8>::interleaved(1.0, 4, [2, 1, 1]).is_ok());
    }

    #[test]
    fn coefficient_boundary_checks_length_overflow_and_finiteness() {
        let one = Bf16::from_bits(0x3f80);
        let zero = Bf16::from_bits(0);
        assert_eq!(
            HostTables::<4>::from_coefficients(1, vec![one], vec![zero; 2]).unwrap_err(),
            RopeError::LengthMismatch
        );
        assert_eq!(
            HostTables::<4>::from_coefficients(1, vec![one; 2], vec![zero; 3]).unwrap_err(),
            RopeError::LengthMismatch
        );
        assert_eq!(
            HostTables::<4>::from_coefficients(usize::MAX, vec![], vec![]).unwrap_err(),
            RopeError::ExtentOverflow
        );
        assert_eq!(
            HostTables::<2>::from_coefficients(isize::MAX as usize / 2 + 1, vec![], vec![])
                .unwrap_err(),
            RopeError::ExtentOverflow
        );
        assert_eq!(
            HostTables::<0>::from_coefficients(0, vec![], vec![]).unwrap_err(),
            RopeError::InvalidDimension
        );
        assert_eq!(
            HostTables::<3>::from_coefficients(0, vec![], vec![]).unwrap_err(),
            RopeError::InvalidDimension
        );
        for invalid in [0x7f80, 0xff80, 0x7fc1] {
            assert_eq!(
                HostTables::<2>::from_coefficients(1, vec![Bf16::from_bits(invalid)], vec![zero])
                    .unwrap_err(),
                RopeError::NonFiniteCoefficient
            );
            assert_eq!(
                HostTables::<2>::from_coefficients(1, vec![one], vec![Bf16::from_bits(invalid)])
                    .unwrap_err(),
                RopeError::NonFiniteCoefficient
            );
        }
        let valid = HostTables::<4>::from_coefficients(1, vec![one; 2], vec![zero; 2]).unwrap();
        assert_eq!(valid.tokens(), 1);
        assert_eq!(bits(valid.cos()), [0x3f80; 2]);
        assert_eq!(bits(valid.sin()), [0; 2]);
    }
}
