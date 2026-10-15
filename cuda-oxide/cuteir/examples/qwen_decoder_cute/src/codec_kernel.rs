/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Waveform kernels use time-major [time, channel] storage throughout.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, launch_contract, thread};
use cute_rs::Bf16;

#[cuda_module]
pub mod kernels {
    use super::*;

    use ctm_rs::mma::{Accumulator16x8, Bf16A, Bf16B, mma_bf16};

    /// Virtual im2col element. Transposed convolution is separated into output
    /// phases, so each phase visits only its one/two contributing input rows.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn input_value(
        input: &[Bf16],
        row: usize,
        k: usize,
        rows: usize,
        inputs: usize,
        in_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u16 {
        if row >= rows || k >= in_group * k_steps {
            return 0;
        }
        let channel = k / k_steps;
        let tap = k % k_steps;
        if transpose && phase + tap * step >= kernel {
            return 0;
        }
        let delay = if transpose {
            tap
        } else {
            (kernel - 1 - tap) * step
        };
        if row < delay {
            0
        } else {
            input[(row - delay) * inputs + group * in_group + channel].to_bits()
        }
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn input_pair(
        input: &[Bf16],
        row: usize,
        k: usize,
        rows: usize,
        inputs: usize,
        in_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u32 {
        u32::from(input_value(
            input, row, k, rows, inputs, in_group, k_steps, group, kernel, step, phase, transpose,
        )) | (u32::from(input_value(
            input,
            row,
            k + 1,
            rows,
            inputs,
            in_group,
            k_steps,
            group,
            kernel,
            step,
            phase,
            transpose,
        )) << 16)
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn weight_value(
        weight: &[Bf16],
        column: usize,
        k: usize,
        in_group: usize,
        out_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u16 {
        if column >= out_group || k >= in_group * k_steps {
            return 0;
        }
        let channel = k / k_steps;
        let tap = k % k_steps;
        let index = if transpose {
            let physical_tap = phase + tap * step;
            if physical_tap >= kernel {
                return 0;
            }
            ((group * in_group + channel) * out_group + column) * kernel + physical_tap
        } else {
            ((group * out_group + column) * in_group + channel) * kernel + tap
        };
        weight[index].to_bits()
    }
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn weight_pair(
        weight: &[Bf16],
        column: usize,
        k: usize,
        in_group: usize,
        out_group: usize,
        k_steps: usize,
        group: usize,
        kernel: usize,
        step: usize,
        phase: usize,
        transpose: bool,
    ) -> u32 {
        u32::from(weight_value(
            weight, column, k, in_group, out_group, k_steps, group, kernel, step, phase, transpose,
        )) | (u32::from(weight_value(
            weight,
            column,
            k + 1,
            in_group,
            out_group,
            k_steps,
            group,
            kernel,
            step,
            phase,
            transpose,
        )) << 16)
    }
    /// Direct implicit-im2col MMA; no packed activation or weight temporary.
    #[launch_contract(domain=1, block=(128,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn conv_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        inputs: usize,
        outputs: usize,
        kernel: usize,
        step: usize,
        groups: usize,
        has_bias: bool,
        transpose: bool,
    ) {
        if !transpose && groups == inputs && groups == outputs {
            // Depthwise channels have only kernel taps to reduce. Padding a
            // single channel into a tensor-core tile would add needless work.
            let index = thread::index_1d().get();
            if index < rows * outputs {
                let row = index / outputs;
                let channel = index % outputs;
                let mut sum = 0.0_f32;
                let mut tap = 0;
                while tap < kernel {
                    let delay = (kernel - 1 - tap) * step;
                    if row >= delay {
                        sum += input[(row - delay) * inputs + channel].to_f32()
                            * weight[channel * kernel + tap].to_f32();
                    }
                    tap += 1;
                }
                if has_bias {
                    // cuDNN stores its BF16 convolution before the separate bias add.
                    sum = Bf16::from_f32(sum).to_f32() + bias[channel].to_f32();
                }
                // SAFETY: one distinct checked output per global thread.
                unsafe {
                    output.as_mut_ptr().add(index).write(Bf16::from_f32(sum));
                }
            }
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tile = thread::blockIdx_x() as usize * 4 + tid / 32;
        let in_group = inputs / groups;
        let out_group = outputs / groups;
        let columns = out_group.div_ceil(8);
        let row_tiles = rows.div_ceil(16);
        let phases = if transpose { step } else { 1 };
        if tile < row_tiles * columns * phases * groups {
            let group = tile / (row_tiles * columns * phases);
            let phase = (tile / (row_tiles * columns)) % phases;
            let m = (tile / columns) % row_tiles * 16;
            let n = tile % columns * 8;
            let lane_group = lane / 4;
            let lane_k = (lane % 4) * 2;
            let k_steps = if transpose {
                kernel.div_ceil(step)
            } else {
                kernel
            };
            let mut accumulator = Accumulator16x8::zero();
            let mut k = 0;
            while k < in_group * k_steps {
                let a = Bf16A::from_registers([
                    input_pair(
                        input,
                        m + lane_group,
                        k + lane_k,
                        rows,
                        inputs,
                        in_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                    input_pair(
                        input,
                        m + lane_group + 8,
                        k + lane_k,
                        rows,
                        inputs,
                        in_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                    input_pair(
                        input,
                        m + lane_group,
                        k + lane_k + 8,
                        rows,
                        inputs,
                        in_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                    input_pair(
                        input,
                        m + lane_group + 8,
                        k + lane_k + 8,
                        rows,
                        inputs,
                        in_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                ]);
                let b = Bf16B::from_registers([
                    weight_pair(
                        weight,
                        n + lane_group,
                        k + lane_k,
                        in_group,
                        out_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                    weight_pair(
                        weight,
                        n + lane_group,
                        k + lane_k + 8,
                        in_group,
                        out_group,
                        k_steps,
                        group,
                        kernel,
                        step,
                        phase,
                        transpose,
                    ),
                ]);
                // SAFETY: the enclosing tile predicate is warp-uniform; all
                // 32 lanes execute identical MMA counts with PTX lane packing.
                accumulator = unsafe { mma_bf16(accumulator, a, b) };
                k += 16;
            }
            let values = accumulator.registers();
            let mut j = 0;
            while j < 4 {
                let (r, c) = ctm_rs::mma::acc_coords(lane, j);
                if m + r < rows && n + c < out_group {
                    let channel = group * out_group + n + c;
                    let value = if has_bias {
                        // Preserve the observable convolution output before bias.
                        Bf16::from_f32(values[j]).to_f32() + bias[channel].to_f32()
                    } else {
                        values[j]
                    };
                    let row = if transpose {
                        (m + r) * step + phase
                    } else {
                        m + r
                    };
                    // SAFETY: MMA coordinates are bijective within each tile;
                    // group and phase map to nonoverlapping output positions.
                    unsafe {
                        output
                            .as_mut_ptr()
                            .add(row * outputs + channel)
                            .write(Bf16::from_f32(value));
                    }
                }
                j += 1;
            }
        }
    }

    /// Affine LayerNorm reduces in FP32 and rounds only the final affine value.
    #[launch_contract(domain=1, block=(128,1,1), min_compute_capability=(8,0))]
    #[kernel]
    pub fn layer_norm_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let row = thread::blockIdx_x() as usize * 4 + tid / 32;
        let lane = tid % 32;
        if row < rows {
            let mut sum = 0.0_f32;
            let mut c = lane;
            while c < width {
                sum += input[row * width + c].to_f32();
                c += 32;
            }
            // SAFETY: complete warp participates; row predicate is warp-uniform.
            let mean = unsafe { ctm_rs::warp::reduce_sum_f32(sum) } / width as f32;
            let mut squared = 0.0_f32;
            c = lane;
            while c < width {
                let d = input[row * width + c].to_f32() - mean;
                squared += d * d;
                c += 32;
            }
            // SAFETY: identical convergent warp participation as the first sum.
            let variance = unsafe { ctm_rs::warp::reduce_sum_f32(squared) } / width as f32;
            let inverse = ctm_rs::float::rsqrt_approx_f32(variance + epsilon);
            c = lane;
            while c < width {
                let value = (input[row * width + c].to_f32() - mean) * inverse * weight[c].to_f32()
                    + bias[c].to_f32();
                // SAFETY: lanes partition columns and warps partition rows.
                unsafe {
                    output
                        .as_mut_ptr()
                        .add(row * width + c)
                        .write(Bf16::from_f32(value));
                }
                c += 32;
            }
        }
    }
    #[inline(always)]
    fn rounded(value: f32) -> f32 {
        Bf16::from_f32(value).to_f32()
    }

    /// Erf-form GELU. The five-coefficient erf approximation has about 1.5e-7
    /// absolute approximation error before FP32 arithmetic; GPU fixtures compare
    /// the observable BF16 result with an independent double-precision libc erf.
    #[inline(always)]
    fn gelu(value: f32) -> f32 {
        let x = value * core::f32::consts::FRAC_1_SQRT_2;
        let magnitude = if x < 0.0 { -x } else { x };
        let t = ctm_rs::float::rcp_approx_ftz_f32(1.0 + 0.3275911 * magnitude);
        let polynomial =
            ((((1.0614054 * t - 1.4531521) * t + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t;
        let tail = polynomial * ctm_rs::float::ex2_approx_f32(-x * x * core::f32::consts::LOG2_E);
        // Use the complementary tail for negative values to avoid cancellation
        // in 1+erf(x), particularly for small negative GELU outputs.
        let factor = if x < 0.0 { tail } else { 2.0 - tail };
        0.5 * value * factor
    }
    /// Preserve the eager BF16 operation boundaries used by SnakeBeta.
    #[inline(always)]
    fn snake(value: f32, log_alpha: f32, log_beta: f32) -> f32 {
        let alpha = rounded(ctm_rs::float::ex2_approx_f32(
            log_alpha * core::f32::consts::LOG2_E,
        ));
        let beta = rounded(ctm_rs::float::ex2_approx_f32(
            log_beta * core::f32::consts::LOG2_E,
        ));
        let reciprocal = rounded(ctm_rs::float::rcp_approx_ftz_f32(rounded(beta + 1e-9)));
        let angle = rounded(value * alpha);
        // BF16 codec activations use modest angles. Split π reduces the angle
        // without the cancellation from a single rounded π constant. Squaring
        // sine makes the omitted odd-multiple sign immaterial.
        let turns = (angle * core::f32::consts::FRAC_1_PI) as i32;
        let reduced = (angle - turns as f32 * 3.1415925) - turns as f32 * 1.509958e-7;
        let sine = rounded(ctm_rs::float::sin_approx_f32(reduced));
        value + rounded(reciprocal * rounded(sine * sine))
    }
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn unary_bf16(
        input: &[Bf16],
        alpha: &[Bf16],
        beta: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        length: usize,
        width: usize,
        operation: u32,
        lower: f32,
        upper: f32,
    ) {
        let index = thread::index_1d().get();
        if index < length {
            let value = input[index].to_f32();
            let result = if operation == 0 {
                snake(
                    value,
                    alpha[index % width].to_f32(),
                    beta[index % width].to_f32(),
                )
            } else if operation == 1 {
                gelu(value)
            } else if operation == 2 {
                if value < lower {
                    lower
                } else if value > upper {
                    upper
                } else {
                    value
                }
            } else if operation == 3 {
                let count = alpha[index / width].to_f32();
                let minimum = rounded(lower);
                value / if count < minimum { minimum } else { count }
            } else {
                value * alpha[index].to_f32()
            };
            // SAFETY: private host operation dispatch validates all parameter
            // extents and one thread owns each element of the output allocation.
            unsafe {
                output.as_mut_ptr().add(index).write(Bf16::from_f32(result));
            }
        }
    }
    #[kernel]
    pub fn scaled_add_bf16(
        input: &[Bf16],
        residual: &[Bf16],
        scale: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        length: usize,
        width: usize,
    ) {
        let index = thread::index_1d().get();
        if index < length {
            let product = rounded(input[index].to_f32() * scale[index % width].to_f32());
            let result = product + residual[index].to_f32();
            // SAFETY: one thread owns each element in the checked logical range.
            unsafe {
                output.as_mut_ptr().add(index).write(Bf16::from_f32(result));
            }
        }
    }
}
