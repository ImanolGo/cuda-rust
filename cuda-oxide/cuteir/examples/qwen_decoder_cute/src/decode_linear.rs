/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! One/two-row BF16 projections with unchanged MMA arithmetic and smaller CTAs.
use crate::{ffn_kernel::kernels, model::Batch, operator::Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cute_rs::{
    Bf16,
    decoder::BiasWeights,
    ffn::{ProjectionRole, ProjectionWeights, RowMajorShape},
};
use std::sync::Arc;

fn launch_config(rows: usize, outputs: usize) -> Result<LaunchConfig1D> {
    if !(1..=2).contains(&rows) || outputs == 0 {
        return Err("decode projection requires one/two rows and positive output width".into());
    }
    let blocks = outputs.div_ceil(8);
    if blocks > i32::MAX as usize {
        return Err("decode projection grid exceeds CUDA grid-x range".into());
    }
    Ok(LaunchConfig1D::new(blocks as u32, 32, 0))
}

pub struct Runner {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}
impl Runner {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            // SAFETY: this example embeds the artifact generated from the
            // matching kernel source; every launch validates its full-warp contract.
            module: unsafe { kernels::load(stream.context())? },
            stream,
        })
    }

    /// Widths and semantic roles remain in the borrowed weight/bias types.
    /// Strides are checked independently; input/output storage never aliases.
    pub fn project_buffer<Role: ProjectionRole, const IN: usize, const OUT: usize>(
        &self,
        input: &DeviceBuffer<Bf16>,
        rows: usize,
        input_stride: usize,
        weight: &ProjectionWeights<'_, Role, IN, OUT>,
        bias: Option<&BiasWeights<'_, Role, OUT>>,
    ) -> Result<Batch<OUT>> {
        let config = launch_config(rows, OUT)?;
        RowMajorShape::new(rows, IN, input_stride)?.validate_len(input.len())?;
        let bias_buffer = bias.map_or(input, |bias| bias.buffer());
        for buffer in [input, weight.buffer(), bias_buffer] {
            if !Arc::ptr_eq(buffer.context(), self.stream.context()) {
                return Err("decode projection buffers must share the runner context".into());
            }
        }
        self.stream.context().synchronize()?;
        let mut output = Batch::allocate(&self.stream, rows)?;
        // Checked full-warps execute exactly the original 16x8 tile: identical
        // A/B packing, ascending K loop and FP32 epilogue. Only CTA ownership
        // changes. Keep all borrowed buffers live through completion on errors.
        let launched = match bias {
            Some(bias) => {
                let launch = self.module.prepare_decode_linear_bias_bf16(config)?;
                self.module.decode_linear_bias_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    bias.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    input_stride,
                    weight.shape().row_stride(),
                )
            }
            None => {
                let launch = self.module.prepare_decode_linear_bf16(config)?;
                self.module.decode_linear_bf16(
                    &self.stream,
                    &launch,
                    input,
                    weight.buffer(),
                    output.buffer_mut(),
                    rows,
                    IN,
                    OUT,
                    input_stride,
                    weight.shape().row_stride(),
                )
            }
        };
        let completed = self.stream.synchronize();
        launched?;
        completed?;
        Ok(output)
    }
}

fn compare(actual: &[Bf16], expected: &[f64], label: &str) -> Result<f64> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(format!("{label}: comparison requires equal nonempty lengths").into());
    }
    let mut maximum = 0.0f64;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let actual = f64::from(actual.to_f32());
        let expected = f64::from(Bf16::from_f32(*expected as f32).to_f32());
        let difference = (actual - expected).abs();
        // The existing per-operation synthetic gate; no new tolerance for
        // changing the reduction order. FP64 oracle does not use warp grouping.
        if !actual.is_finite()
            || !expected.is_finite()
            || difference > 2e-5 + 0.008 * expected.abs()
        {
            return Err(format!(
                "{label}[{index}]: {actual} != FP64/BF16 reference {expected} (error {difference})"
            )
            .into());
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}

fn exact_bits(actual: &[Bf16], original: &[Bf16], label: &str) -> Result<()> {
    if actual.len() != original.len() || actual.is_empty() {
        return Err(format!("{label}: exact comparison requires equal nonempty lengths").into());
    }
    for (index, (a, b)) in actual.iter().zip(original).enumerate() {
        if a.to_bits() != b.to_bits() {
            return Err(format!("{label}[{index}]: one-warp MMA bits {:04x} differ from original four-warp MMA {:04x}",a.to_bits(),b.to_bits()).into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Regular,
    Cancellation,
    Varied,
}

fn varied_bf16(seed: &mut u32) -> Bf16 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let sign = (*seed >> 16) as u16 & 0x8000;
    let exponent = 116 + ((*seed >> 7) % 12) as u16;
    Bf16::from_bits(sign | (exponent << 7) | (*seed as u16 & 127))
}

fn fixture<const IN: usize, const OUT: usize>(
    runner: &Runner,
    rows: usize,
    input_stride: usize,
    weight_stride: usize,
    has_bias: bool,
    pattern: Pattern,
) -> Result<usize> {
    use cute_rs::ffn::Query;
    let poison = Bf16::from_bits(0x7fc1);
    let mut input = vec![poison; rows * input_stride + 7];
    let mut weight = vec![poison; OUT * weight_stride + 7];
    let mut bias = vec![poison; OUT + 7];
    let mut seed = 0x42b3_f991;
    for row in 0..rows {
        for k in 0..IN {
            input[row * input_stride + k] = match pattern {
                Pattern::Cancellation => {
                    Bf16::from_f32([16.0, 0.03125, -16.0, -0.015625][k % 4] * (row + 1) as f32)
                }
                Pattern::Regular => {
                    Bf16::from_f32(((k * 7 + row * 11) % 41) as f32 / 64.0 - 0.3125)
                }
                Pattern::Varied => varied_bf16(&mut seed),
            };
        }
    }
    for column in 0..OUT {
        bias[column] = Bf16::from_f32((column % 7) as f32 / 128.0 - 0.0234375);
        for k in 0..IN {
            weight[column * weight_stride + k] = match pattern {
                Pattern::Cancellation => Bf16::from_f32((column % 5 + 1) as f32 / 8.0),
                Pattern::Regular => {
                    Bf16::from_f32(((k * 13 + column * 17) % 37) as f32 / 128.0 - 0.140625)
                }
                Pattern::Varied => varied_bf16(&mut seed),
            };
        }
    }
    let mut expected = vec![0.0f64; rows * OUT];
    for row in 0..rows {
        for column in 0..OUT {
            let mut sum = if has_bias {
                f64::from(bias[column].to_f32())
            } else {
                0.0
            };
            for k in 0..IN {
                sum += f64::from(input[row * input_stride + k].to_f32())
                    * f64::from(weight[column * weight_stride + k].to_f32());
            }
            expected[row * OUT + column] = sum;
        }
    }
    let input_dev = DeviceBuffer::from_host(&runner.stream, &input)?;
    let weight_dev = DeviceBuffer::from_host(&runner.stream, &weight)?;
    let bias_dev = DeviceBuffer::from_host(&runner.stream, &bias)?;
    let weights = ProjectionWeights::<Query, IN, OUT>::new(&weight_dev, weight_stride)?;
    let biases = BiasWeights::<Query, OUT>::new(&bias_dev)?;
    let output = runner.project_buffer(
        &input_dev,
        rows,
        input_stride,
        &weights,
        has_bias.then_some(&biases),
    )?;
    let label = format!("decode linear {rows}x{IN}→{OUT}, bias={has_bias}, pattern={pattern:?}");
    let actual = output.read(&runner.stream)?;

    // Run the original four-warp entry on the same buffers. The new launch
    // must be bitwise equal, not merely within the independent oracle's bound.
    let mut original = Batch::<OUT>::allocate(&runner.stream, rows)?;
    let config = cuda_core::simt::LaunchConfig {
        grid_dim: (OUT.div_ceil(8).div_ceil(4) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the successful checked launch above established valid one/two-row
    // extents, pitches and contexts. The original block has four complete warps,
    // and a fresh owned output makes the original accumulator map disjoint.
    let launched = unsafe {
        if has_bias {
            runner.module.linear_bias_bf16(
                &runner.stream,
                config,
                &input_dev,
                &weight_dev,
                &bias_dev,
                original.buffer_mut(),
                rows,
                IN,
                OUT,
                input_stride,
                weight_stride,
            )
        } else {
            runner.module.linear_bf16(
                &runner.stream,
                config,
                &input_dev,
                &weight_dev,
                original.buffer_mut(),
                rows,
                IN,
                OUT,
                input_stride,
                weight_stride,
            )
        }
    };
    let completed = runner.stream.synchronize();
    launched?;
    completed?;
    exact_bits(&actual, &original.read(&runner.stream)?, &label)?;
    let error = if matches!(pattern, Pattern::Varied) {
        // These wide-dynamic-range cancellation cases are an arithmetic-
        // preservation gate. The original MMA itself exceeds the synthetic
        // absolute bound near zero, as confirmed before this classification.
        // Require original bits and finite values, and report FP64 error
        // without claiming an independent numerical-accuracy pass here.
        let mut maximum = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            let actual = f64::from(actual.to_f32());
            let expected = f64::from(Bf16::from_f32(*expected as f32).to_f32());
            if !actual.is_finite() || !expected.is_finite() {
                return Err("decode MMA bit-preservation fixture produced nonfinite values".into());
            }
            maximum = maximum.max((actual - expected).abs());
        }
        maximum
    } else {
        compare(&actual, &expected, &label)?
    };
    for (before, buffer) in [
        (&input, &input_dev),
        (&weight, &weight_dev),
        (&bias, &bias_dev),
    ] {
        if !before
            .iter()
            .zip(buffer.to_host_vec(&runner.stream)?)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Err(
                "decode projection changed input/weight/bias storage or its padding".into(),
            );
        }
    }
    let gate = if matches!(pattern, Pattern::Varied) {
        "finite outputs and original-MMA bit identity (FP64 error reported only)"
    } else {
        "FP64 accuracy and original-MMA bit identity"
    };
    println!(
        "{label}: {} values, {gate}, max FP64/BF16 abs {error:.8}",
        expected.len()
    );
    Ok(expected.len())
}

/// Independent FP64 oracle, poisoned padding, cancellation and bias-boundary checks.
pub fn validate(stream: Arc<CudaStream>) -> Result<()> {
    use Pattern::{Cancellation, Regular, Varied};
    use cute_rs::ffn::Query;
    stream.context().bind_to_thread()?;
    let runner = Runner::new(stream)?;
    let mut checked = 0;
    checked += fixture::<17, 7>(&runner, 1, 23, 22, false, Regular)?;
    checked += fixture::<33, 19>(&runner, 2, 41, 38, true, Regular)?;
    checked += fixture::<129, 5>(&runner, 2, 137, 133, false, Cancellation)?;
    checked += fixture::<1024, 3072>(&runner, 1, 1024, 1024, false, Regular)?;
    checked += fixture::<2048, 6144>(&runner, 2, 2048, 2048, false, Regular)?;
    checked += fixture::<6144, 2048>(&runner, 1, 6144, 6144, true, Cancellation)?;
    checked += fixture::<3072, 1024>(&runner, 2, 3072, 3072, false, Regular)?;
    let mut stress_checked = 0;
    stress_checked += fixture::<1024, 2048>(&runner, 1, 1031, 1029, true, Varied)?;
    stress_checked += fixture::<2048, 1024>(&runner, 2, 2055, 2053, false, Varied)?;
    stress_checked += fixture::<6144, 2048>(&runner, 2, 6144, 6144, true, Varied)?;

    // The dot is halfway between BF16 values. Bias must be added in FP32
    // before rounding; rounding the dot first would incorrectly return 1.0.
    let input = DeviceBuffer::from_host(
        &runner.stream,
        &[
            Bf16::from_f32(1.0),
            Bf16::from_f32(1.0 / 256.0),
            Bf16::from_f32(-1.0),
            Bf16::from_f32(-1.0 / 256.0),
        ],
    )?;
    let weight = DeviceBuffer::from_host(&runner.stream, &[Bf16::from_f32(1.0); 2])?;
    let bias = DeviceBuffer::from_host(&runner.stream, &[Bf16::from_f32(1.0 / 1024.0)])?;
    let weights = ProjectionWeights::<Query, 2, 1>::new(&weight, 2)?;
    let biases = BiasWeights::<Query, 1>::new(&bias)?;
    let output = runner
        .project_buffer(&input, 2, 2, &weights, Some(&biases))?
        .read(&runner.stream)?;
    if output[0].to_bits() != Bf16::from_f32(1.0078125).to_bits()
        || output[1].to_bits() != Bf16::from_f32(-1.0).to_bits()
    {
        return Err("decode linear lost FP32 bias-before-BF16 rounding".into());
    }
    for (rows, stride) in [(0, 2), (3, 2), (2, 1), (2, 4)] {
        if runner
            .project_buffer(&input, rows, stride, &weights, Some(&biases))
            .is_ok()
        {
            return Err("decode linear accepted invalid rows/stride/extent".into());
        }
    }
    println!(
        "decode linear: {} FP64/bias accuracy checks, {} original-MMA bit comparisons (including {} varied-exponent stress values), and invalid-extent checks passed",
        checked + 2,
        checked + stress_checked,
        stress_checked,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_geometry_rejects_unsupported_rows_and_grid_overflow() {
        assert!(launch_config(1, 1).is_ok());
        assert!(launch_config(2, 7).is_ok());
        assert!(launch_config(0, 1).is_err());
        assert!(launch_config(3, 1).is_err());
        assert!(launch_config(1, 0).is_err());
        assert!(launch_config(1, usize::MAX).is_err());
    }
    #[test]
    fn fp64_oracle_rejects_nonfinite_and_wrong_results() {
        assert!(compare(&[Bf16::from_f32(1.0)], &[1.0], "test").is_ok());
        assert!(compare(&[Bf16::from_f32(1.0)], &[0.0], "test").is_err());
        assert!(compare(&[Bf16::from_bits(0x7fc1)], &[0.0], "test").is_err());
        for expected in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(compare(&[Bf16::from_f32(0.0)], &[expected], "test").is_err());
        }
        assert!(compare(&[], &[], "test").is_err());
        // A one-ULP BF16 change fits the numerical oracle's bound but must
        // still fail the arithmetic-preservation gate.
        let one = [Bf16::from_f32(1.0)];
        let next = [Bf16::from_f32(1.0078125)];
        assert!(compare(&next, &[1.0], "test").is_ok());
        assert!(exact_bits(&next, &one, "test").is_err());
    }
}
