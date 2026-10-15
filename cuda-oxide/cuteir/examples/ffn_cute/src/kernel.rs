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
    use ctm_rs::mma::{Accumulator16x8, Bf16A, Bf16B, mma_bf16};

    /// Zero-pad logical matrix tails without reading padding or another row.
    #[inline(always)]
    fn element(
        data: &[Bf16],
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
        stride: usize,
    ) -> u16 {
        if row < rows && col < cols {
            data[row * stride + col].to_bits()
        } else {
            0
        }
    }
    /// Pack two adjacent values in the instruction's low/high BF16 order.
    #[inline(always)]
    fn pair(data: &[Bf16], row: usize, col: usize, rows: usize, cols: usize, stride: usize) -> u32 {
        u32::from(element(data, row, col, rows, cols, stride))
            | (u32::from(element(data, row, col + 1, rows, cols, stride)) << 16)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn prefetch_a(
        input: &[Bf16],
        m: usize,
        group: usize,
        k: usize,
        k_lane: usize,
        rows: usize,
        width: usize,
        stride: usize,
    ) -> Bf16A {
        Bf16A::from_registers([
            pair(input, m + group, k + k_lane, rows, width, stride),
            pair(input, m + group + 8, k + k_lane, rows, width, stride),
            pair(input, m + group, k + k_lane + 8, rows, width, stride),
            pair(input, m + group + 8, k + k_lane + 8, rows, width, stride),
        ])
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn prefetch_b(
        weight: &[Bf16],
        n: usize,
        group: usize,
        k: usize,
        k_lane: usize,
        outputs: usize,
        width: usize,
        stride: usize,
    ) -> Bf16B {
        Bf16B::from_registers([
            pair(weight, n + group, k + k_lane, outputs, width, stride),
            pair(weight, n + group, k + k_lane + 8, outputs, width, stride),
        ])
    }

    /// One/two-row inputs repeat their valid row addresses across the warp.
    /// Masking the unused MMA rows avoids divergent global-load branches while
    /// retaining exactly the original A fragment, including its eight zero rows.
    #[inline(always)]
    unsafe fn packed_decode_a(
        input: *const Bf16,
        group: usize,
        k: usize,
        k_lane: usize,
        rows: usize,
        stride: usize,
    ) -> Bf16A {
        let row = group & (rows - 1);
        let mask = 0u32.wrapping_sub((group < rows) as u32);
        // SAFETY: the dispatch proof establishes aligned full pairs in every
        // row. rows is 1 or 2, so row selects a valid initialized source row.
        let (low, high) = unsafe {
            (
                ctm_rs::memory::load_aligned_u32(input.add(row * stride + k + k_lane).cast()),
                ctm_rs::memory::load_aligned_u32(input.add(row * stride + k + k_lane + 8).cast()),
            )
        };
        Bf16A::from_registers([low & mask, 0, high & mask, 0])
    }

    #[inline(always)]
    unsafe fn packed_decode_b(
        weight: *const Bf16,
        row: usize,
        k: usize,
        k_lane: usize,
        stride: usize,
    ) -> Bf16B {
        // SAFETY: dispatch proves full output tiles, complete K fragments,
        // even strides, four-byte aligned bases and sufficient readable bytes.
        unsafe {
            Bf16B::from_registers([
                ctm_rs::memory::load_aligned_u32(weight.add(row * stride + k + k_lane).cast()),
                ctm_rs::memory::load_aligned_u32(weight.add(row * stride + k + k_lane + 8).cast()),
            ])
        }
    }

    /// Prove the packed path's storage requirements once, outside the K loop.
    /// Division checks extents without overflowing row-count * stride.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn packed_decode_supported<const HAS_BIAS: bool>(
        input: &[Bf16],
        weight: &[Bf16],
        bias: &[Bf16],
        output_len: usize,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) -> bool {
        matches!(rows, 1 | 2)
            && width != 0
            && width.is_multiple_of(16)
            && outputs != 0
            && outputs.is_multiple_of(8)
            && input_stride >= width
            && weight_stride >= width
            && input_stride.is_multiple_of(2)
            && weight_stride.is_multiple_of(2)
            && (input.as_ptr() as usize).is_multiple_of(4)
            && (weight.as_ptr() as usize).is_multiple_of(4)
            && rows <= input.len() / input_stride
            && outputs <= weight.len() / weight_stride
            && rows <= output_len / outputs
            && (!HAS_BIAS || bias.len() >= outputs)
    }

    /// This path changes memory transactions, never accumulator arithmetic.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn packed_decode_impl<const HAS_BIAS: bool>(
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
        let lane = thread::threadIdx_x() as usize;
        let n = thread::blockIdx_x() as usize * 8;
        // Uniform CTA predicate; all 32 lanes participate in every MMA.
        if n >= outputs {
            return;
        }
        let group = lane / 4;
        let k_lane = lane % 4 * 2;
        let mut accumulator = Accumulator16x8::zero();
        let mut k = 0;
        while k < width {
            // SAFETY: the caller established packed_decode_supported. Full
            // 64-K chunks load ahead, but consume identical MMA operands in
            // exactly k, k+16, k+32, k+48 order, with no partial reduction.
            unsafe {
                if width - k >= 64 {
                    let a0 = packed_decode_a(input.as_ptr(), group, k, k_lane, rows, input_stride);
                    let b0 = packed_decode_b(weight.as_ptr(), n + group, k, k_lane, weight_stride);
                    let a1 =
                        packed_decode_a(input.as_ptr(), group, k + 16, k_lane, rows, input_stride);
                    let b1 =
                        packed_decode_b(weight.as_ptr(), n + group, k + 16, k_lane, weight_stride);
                    let a2 =
                        packed_decode_a(input.as_ptr(), group, k + 32, k_lane, rows, input_stride);
                    let b2 =
                        packed_decode_b(weight.as_ptr(), n + group, k + 32, k_lane, weight_stride);
                    let a3 =
                        packed_decode_a(input.as_ptr(), group, k + 48, k_lane, rows, input_stride);
                    let b3 =
                        packed_decode_b(weight.as_ptr(), n + group, k + 48, k_lane, weight_stride);
                    accumulator = mma_bf16(accumulator, a0, b0);
                    accumulator = mma_bf16(accumulator, a1, b1);
                    accumulator = mma_bf16(accumulator, a2, b2);
                    accumulator = mma_bf16(accumulator, a3, b3);
                    k += 64;
                } else {
                    let a = packed_decode_a(input.as_ptr(), group, k, k_lane, rows, input_stride);
                    let b = packed_decode_b(weight.as_ptr(), n + group, k, k_lane, weight_stride);
                    accumulator = mma_bf16(accumulator, a, b);
                    k += 16;
                }
            }
        }
        // Only the first two accumulator registers can own one/two-row output.
        // No BF16 conversion happens before optional bias, as in the original.
        if group < rows {
            let values = accumulator.registers();
            let column = n + lane % 4 * 2;
            let mut j = 0;
            while j < 2 {
                let value = if HAS_BIAS {
                    values[j] + bias[column + j].to_f32()
                } else {
                    values[j]
                };
                // SAFETY: each active lane owns a unique adjacent output pair;
                // dispatch checked the complete one/two-row output extent.
                unsafe {
                    output
                        .as_mut_ptr()
                        .add(group * outputs + column + j)
                        .write(Bf16::from_f32(value));
                }
                j += 1;
            }
        }
    }

    /// Compute X times transposed stored weights, with one full warp per 16x8 tile.
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
        // The predicate is uniform across all 32 lanes. Individual missing
        // matrix elements are zero-filled; they never skip a collective.
        if tile < rows.div_ceil(16) * columns {
            let m = tile / columns * 16;
            let n = tile % columns * 8;
            let group = lane / 4;
            let k_lane = (lane % 4) * 2;
            let mut accumulator = Accumulator16x8::zero();
            let mut k = 0;
            while k < width {
                if PREFETCH && width - k >= 64 {
                    // Decode is latency-bound: issue independent global loads
                    // before consuming their fragments. Arithmetic still visits
                    // k, k+16, k+32, k+48 in the original MMA order. Full64-K
                    // chunks only; the unchanged loop below handles the tail.
                    let a0 = prefetch_a(input, m, group, k, k_lane, rows, width, input_stride);
                    let b0 = prefetch_b(weight, n, group, k, k_lane, outputs, width, weight_stride);
                    let a1 = prefetch_a(input, m, group, k + 16, k_lane, rows, width, input_stride);
                    let b1 = prefetch_b(
                        weight,
                        n,
                        group,
                        k + 16,
                        k_lane,
                        outputs,
                        width,
                        weight_stride,
                    );
                    let a2 = prefetch_a(input, m, group, k + 32, k_lane, rows, width, input_stride);
                    let b2 = prefetch_b(
                        weight,
                        n,
                        group,
                        k + 32,
                        k_lane,
                        outputs,
                        width,
                        weight_stride,
                    );
                    let a3 = prefetch_a(input, m, group, k + 48, k_lane, rows, width, input_stride);
                    let b3 = prefetch_b(
                        weight,
                        n,
                        group,
                        k + 48,
                        k_lane,
                        outputs,
                        width,
                        weight_stride,
                    );
                    // SAFETY: the same complete converged warp and lane packing
                    // as the original path. No split reduction or extra MMA.
                    unsafe {
                        accumulator = mma_bf16(accumulator, a0, b0);
                        accumulator = mma_bf16(accumulator, a1, b1);
                        accumulator = mma_bf16(accumulator, a2, b2);
                        accumulator = mma_bf16(accumulator, a3, b3);
                    }
                    k += 64;
                    continue;
                }
                let a = Bf16A::from_registers([
                    pair(input, m + group, k + k_lane, rows, width, input_stride),
                    pair(input, m + group + 8, k + k_lane, rows, width, input_stride),
                    pair(input, m + group, k + k_lane + 8, rows, width, input_stride),
                    pair(
                        input,
                        m + group + 8,
                        k + k_lane + 8,
                        rows,
                        width,
                        input_stride,
                    ),
                ]);
                let b = Bf16B::from_registers([
                    pair(weight, n + group, k + k_lane, outputs, width, weight_stride),
                    pair(
                        weight,
                        n + group,
                        k + k_lane + 8,
                        outputs,
                        width,
                        weight_stride,
                    ),
                ]);
                // SAFETY: complete converged warp, exact PTX BF16 A/B lane
                // packing, FP32 C registers, and an SM120-supported instruction.
                accumulator = unsafe { mma_bf16(accumulator, a, b) };
                k += 16;
            }
            let values = accumulator.registers();
            let mut j = 0;
            while j < 4 {
                let (r, c) = ctm_rs::mma::acc_coords(lane, j);
                if m + r < rows && n + c < outputs {
                    let value = if HAS_BIAS {
                        // Bias is applied before the only BF16 output rounding.
                        values[j] + bias[n + c].to_f32()
                    } else {
                        values[j]
                    };
                    // SAFETY: the accumulator coordinate map is bijective;
                    // tile ownership and checked output extents prevent overlap.
                    unsafe {
                        output
                            .as_mut_ptr()
                            .add((m + r) * outputs + n + c)
                            .write(Bf16::from_f32(value));
                    }
                }
                j += 1;
            }
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
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if packed_decode_supported::<false>(
            input,
            weight,
            input,
            output.len(),
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        ) {
            // SAFETY: the complete storage proof above and this entry's checked
            // full-warp launch contract establish every helper precondition.
            unsafe {
                packed_decode_impl::<false>(
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
        output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        if packed_decode_supported::<true>(
            input,
            weight,
            bias,
            output.len(),
            rows,
            width,
            outputs,
            input_stride,
            weight_stride,
        ) {
            // SAFETY: the complete storage proof above includes the bias extent.
            unsafe {
                packed_decode_impl::<true>(
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

    /// Evaluate SiLU in FP32, then expose its observable BF16 result.
    #[inline(always)]
    fn silu(value: Bf16) -> Bf16 {
        let value = value.to_f32();
        let sigmoid = if value >= 0.0 {
            let e = ctm_rs::float::ex2_approx_f32(-value * core::f32::consts::LOG2_E);
            ctm_rs::float::rcp_approx_ftz_f32(1.0 + e)
        } else {
            let e = ctm_rs::float::ex2_approx_f32(value * core::f32::consts::LOG2_E);
            e * ctm_rs::float::rcp_approx_ftz_f32(1.0 + e)
        };
        Bf16::from_f32(value * sigmoid)
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
