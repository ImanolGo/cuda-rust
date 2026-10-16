/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 linear and activation kernels used by the checked host composition.
use ctm_rs::memory::DisjointSlice;
use ctm_rs::{cuda_module, kernel, launch_contract, thread};
use cute_rs::Bf16;

#[cuda_module]
pub mod kernels {
    use super::*;
    use cute_rs::bf16_mma::{Bf16Mma16x8x16, MmaAccumulator, PackedDecode};
    use cute_rs::numeric::silu_bf16 as silu;

    /// A bounded matrix view: missing logical values are BF16 positive zero.
    #[inline(always)]
    fn element(
        data: &[Bf16],
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        stride: usize,
    ) -> Bf16 {
        if row < rows && col < cols {
            data[row * stride + col]
        } else {
            Bf16::from_bits(0)
        }
    }

    /// Consume a checked packed view without changing the original MMA schedule.
    #[inline(always)]
    unsafe fn packed_decode_impl<const HAS_BIAS: bool>(
        source: &PackedDecode<'_>,
        bias: &[Bf16],
        output: &mut DisjointSlice<Bf16>,
        width: usize,
        outputs: usize,
    ) {
        let lane = thread::threadIdx_x() as usize;
        let n = thread::blockIdx_x() as usize * 8;
        if n >= outputs {
            return;
        }
        let mma = Bf16Mma16x8x16::get_slice(lane);
        let a = mma.partition_a();
        let b = mma.partition_b();
        let mut accumulator = MmaAccumulator::zero();
        let mut k = 0;
        while k < width {
            // SAFETY: PackedDecode validates source extents/alignment; N and K
            // select full tiles. Every lane reaches the same instruction order.
            unsafe {
                if width - k >= 64 {
                    let a0 = a.load_decode_packed(source, k);
                    let b0 = b.load_decode_packed(source, n, k);
                    let a1 = a.load_decode_packed(source, k + 16);
                    let b1 = b.load_decode_packed(source, n, k + 16);
                    let a2 = a.load_decode_packed(source, k + 32);
                    let b2 = b.load_decode_packed(source, n, k + 32);
                    let a3 = a.load_decode_packed(source, k + 48);
                    let b3 = b.load_decode_packed(source, n, k + 48);
                    accumulator = mma.mma(accumulator, a0, b0);
                    accumulator = mma.mma(accumulator, a1, b1);
                    accumulator = mma.mma(accumulator, a2, b2);
                    accumulator = mma.mma(accumulator, a3, b3);
                    k += 64;
                } else {
                    let av = a.load_decode_packed(source, k);
                    let bv = b.load_decode_packed(source, n, k);
                    accumulator = mma.mma(accumulator, av, bv);
                    k += 16;
                }
            }
        }
        let mut output = cute_rs::partition::StridedTensorMut::from_disjoint_slice(output);
        mma.partition_c()
            .store_decode_with(accumulator, source, |row, col, value| {
                let value = if HAS_BIAS {
                    value + bias[n + col].to_f32()
                } else {
                    value
                };
                // SAFETY: C's bijective layout and the checked grid give each lane
                // disjoint initialized output coordinates, with one BF16 rounding.
                unsafe {
                    output.store_at(row * outputs + n + col, Bf16::from_f32(value));
                }
            });
    }

    /// Compute X times stored transposed weights through typed warp partitions.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn linear_bf16_impl<
        const HAS_BIAS: bool,
        const WARPS_PER_BLOCK: usize,
        const PREFETCH: bool,
    >(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tile = thread::blockIdx_x() as usize * WARPS_PER_BLOCK + tid / 32;
        let columns = outputs.div_ceil(8);
        if tile < rows.div_ceil(16) * columns {
            let m = tile / columns * 16;
            let n = tile % columns * 8;
            let mma = Bf16Mma16x8x16::get_slice(lane);
            let load_a = |k| {
                mma.partition_a()
                    .load_with(|r, c| element(input, m + r, k + c, rows, width, input_stride))
            };
            let load_b = |k| {
                mma.partition_b()
                    .load_with(|r, c| element(weight, n + r, k + c, outputs, width, weight_stride))
            };
            let mut accumulator = MmaAccumulator::zero();
            let mut k = 0;
            while k < width {
                if PREFETCH && width - k >= 64 {
                    let a0 = load_a(k);
                    let b0 = load_b(k);
                    let a1 = load_a(k + 16);
                    let b1 = load_b(k + 16);
                    let a2 = load_a(k + 32);
                    let b2 = load_b(k + 32);
                    let a3 = load_a(k + 48);
                    let b3 = load_b(k + 48);
                    // SAFETY: the full warp uses the atom's typed fragments;
                    // arithmetic consumes K tiles in unchanged increasing order.
                    unsafe {
                        accumulator = mma.mma(accumulator, a0, b0);
                        accumulator = mma.mma(accumulator, a1, b1);
                        accumulator = mma.mma(accumulator, a2, b2);
                        accumulator = mma.mma(accumulator, a3, b3);
                    }
                    k += 64;
                    continue;
                }
                let a = load_a(k);
                let b = load_b(k);
                // SAFETY: no lane skips the collective; tails are zero-padded.
                accumulator = unsafe { mma.mma(accumulator, a, b) };
                k += 16;
            }
            let mut output = cute_rs::partition::StridedTensorMut::from_disjoint_slice(&mut output);
            mma.partition_c().store_with(accumulator, |r, c, value| {
                if m + r < rows && n + c < outputs {
                    let value = if HAS_BIAS {
                        value + bias[n + c].to_f32()
                    } else {
                        value
                    };
                    // SAFETY: C's bijective layout and the host-checked tile grid
                    // give disjoint output writes within the validated extent.
                    unsafe {
                        output.store_at((m + r) * outputs + n + c, Bf16::from_f32(value));
                    }
                }
            });
        }
    }

    /// Compute a bias-free projection with FP32 accumulation and BF16 output.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        // The const-false specialization never reads the placeholder bias slice.
        linear_bf16_impl::<false, 4, false>(
            input,
            weight,
            input,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Add the projection bias in FP32 before rounding the output once to BF16.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bias_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        linear_bf16_impl::<true, 4, false>(
            input,
            weight,
            bias,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// The same MMA arithmetic with one warp per CTA, distributing short-row
    /// decode projections over more SMs without changing reduction order.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_linear_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if outputs != 0
            && rows <= output.len() / outputs
            && PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: the borrowed proof establishes packed operand
                    // bounds; output capacity is checked above, and this entry's
                    // launch contract supplies exactly one complete warp.
                    unsafe {
                        packed_decode_impl::<false>(source, input, &mut output, width, outputs);
                    }
                },
            )
        {
            return;
        }
        linear_bf16_impl::<false, 1, true>(
            input,
            weight,
            input,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Bias remains inside the unchanged FP32 accumulator epilogue.
    #[launch_contract(domain=1, block=(32,1,1), min_compute_capability=(8,0))]
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_linear_bias_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if outputs != 0
            && rows <= output.len() / outputs
            && bias.len() >= outputs
            && PackedDecode::with_validated(
                input,
                weight,
                rows,
                width,
                outputs,
                input_stride,
                weight_stride,
                |source| {
                    // SAFETY: the source proof, output/bias extent checks and
                    // exact full-warp launch contract establish packed loads,
                    // disjoint stores, and collective participation.
                    unsafe {
                        packed_decode_impl::<true>(source, bias, &mut output, width, outputs);
                    }
                },
            )
        {
            return;
        }
        linear_bf16_impl::<true, 1, true>(
            input,
            weight,
            bias,
            output,
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        );
    }

    /// Preserve BF16 rounding after SiLU and after multiplication.
    #[kernel]
    pub fn activate_bf16(
        gate: &[Bf16],
        up: &[Bf16],
        mut silu_output: DisjointSlice<Bf16>,
        mut product: DisjointSlice<Bf16>,
        length: usize,
    ) {
        let tid = thread::index_1d().get();
        let gates = cute_rs::Tensor::from_slice(gate)
            .zipped_divide::<8>()
            .slice(tid);
        let ups = cute_rs::Tensor::from_slice(up)
            .zipped_divide::<8>()
            .slice(tid);
        let mut silus = cute_rs::TensorMut::from_disjoint_slice(&mut silu_output)
            .zipped_divide::<8>()
            .slice(tid);
        let mut products = cute_rs::TensorMut::from_disjoint_slice(&mut product)
            .zipped_divide::<8>()
            .slice(tid);
        let base = gates.base();
        if base < length && length - base >= 8 {
            // SAFETY: checked buffers contain `length` values; bases are
            // aligned multiples of 8, and each thread owns one disjoint tile.
            let values = unsafe { gates.load() }.map(silu);
            let multiplied = values.zip_map(unsafe { ups.load() }, |s, u| {
                Bf16::from_f32(s.to_f32() * u.to_f32())
            });
            unsafe {
                silus.store(values);
                products.store(multiplied);
            }
        } else {
            let mut index = base;
            while index < length {
                let s = silu(gate[index]);
                let value = Bf16::from_f32(s.to_f32() * up[index].to_f32());
                unsafe {
                    silus.store_linear(index, s);
                    products.store_linear(index, value);
                }
                index += 1;
            }
        }
    }
}
