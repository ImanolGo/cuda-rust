/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! BF16 linear and activation kernels used by the checked host composition.
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
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

    /// Compute X times transposed stored weights, with one full warp per 16x8 tile.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_bf16(
        input: &[Bf16],
        weight: &[Bf16],
        mut output: DisjointSlice<Bf16>,
        rows: usize,
        width: usize,
        outputs: usize,
        input_stride: usize,
        weight_stride: usize,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tile = thread::blockIdx_x() as usize * 4 + tid / 32;
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
                let (r, c) = cuda_device::mma_frag::acc_coords(lane, j);
                if m + r < rows && n + c < outputs {
                    // SAFETY: the accumulator coordinate map is bijective;
                    // tile ownership and checked output extents prevent overlap.
                    unsafe {
                        output
                            .as_mut_ptr()
                            .add((m + r) * outputs + n + c)
                            .write(Bf16::from_f32(values[j]));
                    }
                }
                j += 1;
            }
        }
    }

    /// Evaluate SiLU in FP32, then expose its observable BF16 result.
    #[inline(always)]
    fn silu(value: Bf16) -> Bf16 {
        let value = value.to_f32();
        let sigmoid = if value >= 0.0 {
            let e = cuda_device::float::ex2_approx_f32(-value * core::f32::consts::LOG2_E);
            cuda_device::float::rcp_approx_ftz_f32(1.0 + e)
        } else {
            let e = cuda_device::float::ex2_approx_f32(value * core::f32::consts::LOG2_E);
            e * cuda_device::float::rcp_approx_ftz_f32(1.0 + e)
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
