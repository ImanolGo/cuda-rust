/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Event-timed projection microbenchmarks; excludes allocation and model orchestration.
use crate::{ffn_kernel::kernels, operator::Result};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::Bf16;
use std::sync::Arc;

pub fn run(stream: Arc<CudaStream>) -> Result<()> {
    // SAFETY: this executable embeds the artifact from the same kernel source.
    let module = unsafe { kernels::load(stream.context())? };
    const ITERATIONS: usize = 128;
    const SAMPLES: usize = 7;
    println!(
        "event-timed repeated projection launches; {SAMPLES} samples of {ITERATIONS}; warm weights; no end-to-end claim"
    );
    for (rows, width, outputs) in [
        (1, 1024, 3072),
        (1, 2048, 1024),
        (1, 2048, 6144),
        (1, 6144, 2048),
        (2, 2048, 1024),
        (2, 6144, 2048),
    ] {
        let input = DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.25); rows * width])?;
        let weight =
            DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.125); outputs * width])?;
        let mut output =
            DeviceBuffer::from_host(&stream, &vec![Bf16::from_f32(0.0); rows * outputs])?;
        let config = LaunchConfig1D::new(outputs.div_ceil(8) as u32, 32, 0);
        let prepared = module.prepare_decode_linear_bf16(config)?;
        let original = cuda_core::simt::LaunchConfig {
            grid_dim: (outputs.div_ceil(32) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let start = stream
            .context()
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        let end = stream
            .context()
            .new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
        for decode in [false, true] {
            let mut enqueue = || -> Result<()> {
                if decode {
                    module.decode_linear_bf16(
                        &stream,
                        &prepared,
                        &input,
                        &weight,
                        &mut output,
                        rows,
                        width,
                        outputs,
                        width,
                        width,
                    )?;
                } else {
                    // SAFETY: full warps, exact contiguous extents and an
                    // exclusively borrowed output with one owner per tile.
                    unsafe {
                        module.linear_bf16(
                            &stream,
                            original,
                            &input,
                            &weight,
                            &mut output,
                            rows,
                            width,
                            outputs,
                            width,
                            width,
                        )?;
                    }
                }
                Ok(())
            };
            // On any enqueue/event failure, drain the stream before the borrowed
            // buffers or module may drop. Timed batches have no internal waits.
            let measured = (|| -> Result<Vec<f64>> {
                for _ in 0..20 {
                    enqueue()?;
                }
                stream.synchronize()?;
                let mut samples = Vec::with_capacity(SAMPLES);
                for _ in 0..SAMPLES {
                    start.record(&stream)?;
                    for _ in 0..ITERATIONS {
                        enqueue()?;
                    }
                    end.record(&stream)?;
                    samples.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / ITERATIONS as f64);
                }
                Ok(samples)
            })();
            let completed = stream.synchronize();
            let mut samples = measured?;
            completed?;
            let actual = output.to_host_vec(&stream)?;
            let expected = Bf16::from_f32(width as f32 / 32.0).to_bits();
            if actual.iter().any(|value| value.to_bits() != expected) {
                return Err(
                    "projection microbenchmark output differs from exact constant-input result"
                        .into(),
                );
            }
            samples.sort_by(f64::total_cmp);
            println!(
                "{} rows={rows} width={width} outputs={outputs} median_us={:.6} min_us={:.6} max_us={:.6}",
                if decode { "decode" } else { "original" },
                samples[SAMPLES / 2],
                samples[0],
                samples[SAMPLES - 1]
            );
        }
    }
    Ok(())
}
