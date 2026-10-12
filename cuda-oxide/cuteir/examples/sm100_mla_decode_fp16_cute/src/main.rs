/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

mod comparison;
mod kernel;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, sys};
use cute_rs::sm100_attention::make_tma_desc;
use serde::Deserialize;
use serde_json::json;
use std::{
    error::Error,
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    batch: u32,
    kv_length: u32,
    splits: u32,
    heads: u32,
    query_length: u32,
    latent: u32,
    rope: u32,
    page_size: u32,
    softmax_scale: f32,
    output_scale: f32,
}
impl Config {
    fn validate(&self) -> Result<()> {
        if (
            self.heads,
            self.query_length,
            self.latent,
            self.rope,
            self.page_size,
        ) != (128, 1, 512, 64, 128)
        {
            return Err("supported profile: H128, Sq1, latent512, RoPE64, page128".into());
        }
        if self.batch == 0
            || self.batch > 65535
            || self.kv_length == 0
            || self.kv_length > (1 << 30)
            || self.splits == 0
            || self.splits > 256
        {
            return Err("batch/KV length/splits outside supported bounds".into());
        }
        if !self.softmax_scale.is_finite()
            || self.softmax_scale <= 0.0
            || !self.output_scale.is_finite()
        {
            return Err("scales must be finite; softmax scale must be positive".into());
        }
        let work = self
            .batch
            .checked_mul(self.splits)
            .ok_or("work count overflow")?;
        if work > i32::MAX as u32 / 2 {
            return Err("work count exceeds CUDA grid limit".into());
        }
        let rows = u64::from(self.batch) * u64::from(self.pages()) * 128;
        if rows > i32::MAX as u64 {
            return Err("physical cache row coordinates exceed i32".into());
        }
        for n in [
            rows * 512 * 2,
            u64::from(self.batch) * 128 * 512 * 2,
            u64::from(self.batch) * 128 * u64::from(self.splits) * 513 * 4,
        ] {
            if n > isize::MAX as u64 {
                return Err("allocation exceeds host addressable limit".into());
            }
        }
        Ok(())
    }
    fn pages(&self) -> u32 {
        self.kv_length.div_ceil(128)
    }
    fn active_splits(&self) -> u32 {
        let pages = self.pages();
        pages.div_ceil(pages.div_ceil(self.splits))
    }
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    clusters: Option<u32>,
    warmup: usize,
    iterations: usize,
    rounds: usize,
    validation_only: bool,
}
impl Args {
    fn parse() -> Result<Self> {
        let mut input = None;
        let mut output = None;
        let mut clusters = None;
        let mut warmup = 10;
        let mut iterations = 100;
        let mut rounds = 3;
        let mut validation_only = false;
        let mut args = std::env::args().skip(1);
        while let Some(key) = args.next() {
            if key == "--validation-only" || key == "--check-only" {
                validation_only = true;
                continue;
            }
            if key == "--help" || key == "-h" {
                println!(
                    "sm100_mla_decode_fp16_cute --input-dir DIR --output-dir DIR [--clusters N]\n  [--validation-only] [--warmup 10 --iterations 100 --rounds 3]\nUse compare.py for seeded inputs, independent output/LSE validation, and matched CuTeDSL timing.\nThe input directory contains config.json and five little-endian array files."
                );
                std::process::exit(0);
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            match key.as_str() {
                "--input-dir" => input = Some(PathBuf::from(value)),
                "--output-dir" => output = Some(PathBuf::from(value)),
                "--clusters" => clusters = Some(value.parse()?),
                "--warmup" => warmup = value.parse()?,
                "--iterations" | "--iters" => iterations = value.parse()?,
                "--rounds" | "--windows" => rounds = value.parse()?,
                _ => return Err(format!("unknown argument {key}").into()),
            }
        }
        if iterations == 0 || rounds == 0 || clusters == Some(0) {
            return Err("iterations, rounds and clusters must be positive".into());
        }
        Ok(Self {
            input: input.ok_or("--input-dir is required")?,
            output: output.ok_or("--output-dir is required")?,
            clusters,
            warmup,
            iterations,
            rounds,
            validation_only,
        })
    }
}

fn load_u16(path: &Path, count: usize) -> Result<Vec<u16>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != count.checked_mul(2).ok_or("input byte count overflow")? {
        return Err(format!("wrong byte count for {}", path.display()).into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    if values.iter().any(|v| !f16::from_bits(*v).is_finite()) {
        return Err(format!("nonfinite input in {}", path.display()).into());
    }
    Ok(values)
}

fn load_pages(path: &Path, count: usize, physical_pages: usize) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != count.checked_mul(4).ok_or("page table overflow")? {
        return Err("wrong page table byte count".into());
    }
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    if values
        .iter()
        .any(|v| *v < 0 || *v as usize >= physical_pages)
    {
        return Err("page table index out of range".into());
    }
    Ok(values)
}

fn main() -> Result<()> {
    if comparison::dispatch(&std::env::args_os().skip(1).collect::<Vec<_>>())? {
        return Ok(());
    }
    let args = Args::parse()?;
    let cfg: Config = serde_json::from_slice(&std::fs::read(args.input.join("config.json"))?)?;
    cfg.validate()?;
    let batch = cfg.batch as usize;
    let pages = cfg.pages() as usize;
    let q = load_u16(&args.input.join("q_latent.f16"), batch * 128 * 512)?;
    let qr = load_u16(&args.input.join("q_rope.f16"), batch * 128 * 64)?;
    let k = load_u16(&args.input.join("c_latent.f16"), batch * pages * 128 * 512)?;
    let kr = load_u16(&args.input.join("c_rope.f16"), batch * pages * 128 * 64)?;
    let pt = load_pages(
        &args.input.join("page_table.i32"),
        batch * pages,
        batch * pages,
    )?;
    let context = CudaContext::new(0)?;
    if context.compute_capability()? != (10, 0) {
        return Err("this example requires B200/SM100".into());
    }
    let stream = context.new_stream()?;
    let dq = DeviceBuffer::from_host(&stream, &q)?;
    let dqr = DeviceBuffer::from_host(&stream, &qr)?;
    let dk = DeviceBuffer::from_host(&stream, &k)?;
    let dkr = DeviceBuffer::from_host(&stream, &kr)?;
    let dpt = DeviceBuffer::from_host(&stream, &pt)?;
    let dout = DeviceBuffer::from_host(&stream, &vec![f16::NAN.to_bits(); batch * 128 * 512])?;
    let dlse = DeviceBuffer::from_host(&stream, &vec![f32::NAN; batch * 128])?;
    let workspace_values = if cfg.splits > 1 {
        batch * 128 * cfg.splits as usize * 513
    } else {
        1
    };
    let workspace = DeviceBuffer::from_host(&stream, &vec![f32::NAN; workspace_values])?;
    let q_desc = make_tma_desc::<kernel::QLayout>(
        dq.cu_deviceptr() as *mut _,
        cfg.batch as u64 * 128,
        512,
        512,
    )?;
    let qr_desc = make_tma_desc::<kernel::QLayout>(
        dqr.cu_deviceptr() as *mut _,
        cfg.batch as u64 * 128,
        64,
        64,
    )?;
    let physical_rows = (batch * pages * 128) as u64;
    let k_desc =
        make_tma_desc::<kernel::KLayout>(dk.cu_deviceptr() as *mut _, physical_rows, 512, 512)?;
    let kr_desc =
        make_tma_desc::<kernel::KLayout>(dkr.cu_deviceptr() as *mut _, physical_rows, 64, 64)?;
    let v_desc =
        make_tma_desc::<kernel::VLayout>(dk.cu_deviceptr() as *mut _, physical_rows, 512, 512)?;
    let module = unsafe { kernel::kernels::load(&context) }?;
    let work = cfg.batch * cfg.splits;
    let probe = module.prepare_sm100_mla_decode_fp16_cute_rs(LaunchConfig1D::new(
        2,
        kernel::THREADS,
        kernel::DYNAMIC_SMEM_BYTES,
    ))?;
    let capacity = probe.function().max_active_clusters(
        (work * 2, 1, 1),
        (kernel::THREADS, 1, 1),
        kernel::DYNAMIC_SMEM_BYTES,
        (2, 1, 1),
    )?;
    let clusters = args.clusters.unwrap_or(capacity.min(work));
    if clusters == 0 || clusters > capacity || clusters > work {
        return Err(
            format!("requested {clusters} clusters; capacity={capacity}, work={work}").into(),
        );
    }
    let registers = probe.function().num_registers()?;
    let local_bytes_per_thread = probe.function().local_size_bytes()?;
    drop(probe);
    let launch = module.prepare_sm100_mla_decode_fp16_cute_rs(LaunchConfig1D::new(
        clusters * 2,
        kernel::THREADS,
        kernel::DYNAMIC_SMEM_BYTES,
    ))?;
    let reduction = module.prepare_sm100_mla_decode_fp16_reduce_cute_rs(LaunchConfig1D::new(
        cfg.batch * 128,
        128,
        1024,
    ))?;
    let partial = workspace.cu_deviceptr() as *mut f32;
    let partial_lse = if cfg.splits > 1 {
        unsafe { partial.add(batch * 128 * cfg.splits as usize * 512) }
    } else {
        partial
    };
    let active_splits = cfg.active_splits();
    let run = || -> Result<()> {
        unsafe {
            module.sm100_mla_decode_fp16_cute_rs(
                &stream,
                &launch,
                q_desc,
                qr_desc,
                k_desc,
                kr_desc,
                v_desc,
                dpt.cu_deviceptr() as *const i32,
                dout.cu_deviceptr() as *mut f16,
                dlse.cu_deviceptr() as *mut f32,
                partial,
                partial_lse,
                cfg.batch as u64,
                cfg.kv_length as u64,
                cfg.splits as u64,
                cfg.softmax_scale.to_bits() as u64,
                cfg.output_scale.to_bits() as u64,
            )?;
            if cfg.splits > 1 {
                module.sm100_mla_decode_fp16_reduce_cute_rs(
                    &stream,
                    &reduction,
                    partial,
                    partial_lse,
                    dout.cu_deviceptr() as *mut f16,
                    dlse.cu_deviceptr() as *mut f32,
                    active_splits as u64,
                    cfg.splits as u64,
                )?;
            }
        }
        Ok(())
    };
    run()?;
    let checked_out = dout.to_host_vec(&stream)?;
    let checked_lse = dlse.to_host_vec(&stream)?;
    if checked_out.iter().any(|x| !f16::from_bits(*x).is_finite())
        || checked_lse.iter().any(|x| !x.is_finite())
    {
        return Err("kernel produced nonfinite output or left poisoned output unwritten".into());
    }
    let mut samples = Vec::new();
    if !args.validation_only {
        for _ in 0..args.rounds {
            for _ in 0..args.warmup {
                run()?;
            }
            stream.synchronize()?;
            let start = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            let end = context.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            start.record(&stream)?;
            for _ in 0..args.iterations {
                run()?;
            }
            end.record(&stream)?;
            end.synchronize()?;
            samples.push(f64::from(start.elapsed_ms(&end)?) * 1000.0 / args.iterations as f64);
        }
    }
    let result_out = dout.to_host_vec(&stream)?;
    let result_lse = dlse.to_host_vec(&stream)?;
    if result_out != checked_out || result_lse != checked_lse {
        return Err("output changed across identical launches".into());
    }
    std::fs::create_dir_all(&args.output)?;
    std::fs::write(
        args.output.join("output.f16"),
        result_out
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(
        args.output.join("lse.f32"),
        result_lse
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.is_empty() {
        None
    } else {
        Some((sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0)
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "implementation":"sm100_mla_decode_fp16_cute_rs", "samples_us":samples,"median_us":median,
            "validation":if args.validation_only {
                "finite; independent oracle supplied by compare.py"
            } else {
                "finite-and-repeatable; independent oracle supplied by compare.py"
            },
            "gpu":context.device_name()?,"clusters":clusters,"resident_capacity":capacity,
            "registers":registers,"threads":kernel::THREADS,"shared_bytes":kernel::DYNAMIC_SMEM_BYTES,
            "local_bytes_per_thread":local_bytes_per_thread,"min_blocks_per_sm":1,
            "cutlass_optimization_level":option_env!("CUDA_OXIDE_CUTLASS_OPT_LEVEL").unwrap_or("3"),
            "method":"cuda-events-repeated-complete-invocation","includes_split_reduction":cfg.splits > 1,
            "warmup":args.warmup,"iterations":args.iterations,"rounds":args.rounds,
            "validation_only":args.validation_only,
        }))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probability_vectors_remain_contiguous_and_aligned_under_swizzle() {
        for row in 0..64 {
            for col in (0..32).step_by(8) {
                let start = kernel::PLayout::offset(row, col);
                assert_eq!(start % 8, 0);
                for lane in 0..8 {
                    assert_eq!(
                        kernel::PLayout::offset(row, col + lane),
                        start + lane as usize
                    );
                }
            }
        }
    }

    #[test]
    fn split_workspace_offset_does_not_wrap_at_four_gib_elements() {
        assert_eq!(
            kernel::partial_offset(131_071, 255, 256, 511),
            17_179_869_183usize
        );
    }
    #[test]
    fn reject_large_physical_cache_coordinates_before_allocation() {
        let cfg = Config {
            batch: 65535,
            kv_length: 1 << 30,
            splits: 1,
            heads: 128,
            query_length: 1,
            latent: 512,
            rope: 64,
            page_size: 128,
            softmax_scale: 0.0416,
            output_scale: 1.0,
        };
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn active_splits_ignore_empty_work_without_changing_partitioning() {
        let mut cfg = Config {
            batch: 1,
            kv_length: 128,
            splits: 1,
            heads: 128,
            query_length: 1,
            latent: 512,
            rope: 64,
            page_size: 128,
            softmax_scale: 0.0416,
            output_scale: 1.0,
        };
        for (length, splits, expected) in
            [(129, 256, 2), (257, 2, 2), (1024, 3, 3), (32768, 256, 256)]
        {
            cfg.kv_length = length;
            cfg.splits = splits;
            assert_eq!(cfg.active_splits(), expected);
        }
    }
    #[test]
    fn reject_nonpositive_softmax_scale() {
        let cfg = Config {
            batch: 1,
            kv_length: 128,
            splits: 1,
            heads: 128,
            query_length: 1,
            latent: 512,
            rope: 64,
            page_size: 128,
            softmax_scale: 0.0,
            output_scale: 1.0,
        };
        assert!(cfg.validate().is_err());
    }
}
