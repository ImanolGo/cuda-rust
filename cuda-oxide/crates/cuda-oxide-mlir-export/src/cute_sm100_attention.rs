/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Independent layout-aware SM100 MMA and tiled TMA operations.
use crate::cute_sm100::{Emit, integer_property, llvm_ptr_type, property, segments};
use dialect_cute::{attributes::CuteComposedLayoutAttr, sm100_attention_ops::*};
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    operation::Operation,
};
use pliron_mlir_export::{
    DropAttribute, MlirOperation, MlirType, MlirValueUse, OperationInput, OperationTranslation,
    TranslationError, TranslationRegistry, TranslationSession,
};

pub(crate) fn register_cute_sm100_attention_pack(
    r: &mut TranslationRegistry,
) -> Result<(), TranslationError> {
    r.register_attribute::<CuteSm100MmaAttr>(DropAttribute)?;
    r.register_attribute::<CuteSm100TmaAttr>(DropAttribute)?;
    r.register_operation::<CuteSm100MmaSsOp>(Mma(false))?;
    r.register_operation::<CuteSm100MmaTsOp>(Mma(true))?;
    r.register_operation::<CuteSm100TmaLoadOp>(Tma(false))?;
    r.register_operation::<CuteSm100TmaStoreOp>(Tma(true))?;
    Ok(())
}
fn contract(input: &OperationInput) -> Result<(), String> {
    if !input.results.is_empty()
        || !input.attributes.is_empty()
        || !input.regions.is_empty()
        || !input.successors.is_empty()
    {
        return Err(
            "SM100 operation retained unexpected results, attributes or control flow".into(),
        );
    }
    Ok(())
}
struct Mma(bool);
impl OperationTranslation for Mma {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let p = if self.0 {
            let o = CuteSm100MmaTsOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        } else {
            let o = CuteSm100MmaSsOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        }
        .ok_or("missing SM100 MMA plan")?;
        contract(&input)?;
        let mut e = Emit {
            session,
            location: input.location,
            ops: vec![],
        };
        let a = input.operands;
        let ad = if self.0 {
            None
        } else {
            Some(descriptor(
                &mut e,
                a[1].clone(),
                a[4].clone(),
                &p.a_layout,
                p.m / p.cta_group,
                p.k,
                p.a_mn,
            )?)
        };
        let bd = descriptor(
            &mut e,
            a[2].clone(),
            a[if self.0 { 4 } else { 5 }].clone(),
            &p.b_layout,
            p.n / p.cta_group,
            p.k,
            p.b_mn,
        )?;
        let dst = e.cast("llvm.inttoptr", a[0].clone(), llvm_ptr_type(6)?)?;
        let instruction = (1u32 << 4)
            | ((p.n >> 3) << 17)
            | ((p.m >> 4) << 24)
            | ((p.a_mn as u32) << 15)
            | ((p.b_mn as u32) << 16);
        let inst = e.constant(32, instruction as i128)?;
        let yes = e.constant(1, 1)?;
        for k in (0..p.k).step_by(16) {
            let av = if let Some(ad) = &ad {
                let offset = e.constant(
                    64,
                    descriptor_offset(p.m / p.cta_group, p.k, p.a_mn, k) as i128,
                )?;
                e.binary("arith.addi", ad.clone(), offset)?
            } else {
                // Two FP16 probability columns occupy one 32-bit TMEM column.
                let offset = e.constant(32, (k / 2) as i128)?;
                let address = e.binary("arith.addi", a[1].clone(), offset)?;
                e.cast("llvm.inttoptr", address, llvm_ptr_type(6)?)?
            };
            let offset = e.constant(
                64,
                descriptor_offset(p.n / p.cta_group, p.k, p.b_mn, k) as i128,
            )?;
            let bv = e.binary("arith.addi", bd.clone(), offset)?;
            let mut mma = e.operation(
                "nvvm.tcgen05.mma",
                vec![
                    dst.clone(),
                    av,
                    bv,
                    inst.clone(),
                    if k == 0 { a[3].clone() } else { yes.clone() },
                ],
            )?;
            property(&mut mma, "mmaKind", "#nvvm.tcgen05_mma_kind<f16>")?;
            property(
                &mut mma,
                "ctaGroup",
                &format!("#nvvm.cta_group<cta_{}>", p.cta_group),
            )?;
            property(
                &mut mma,
                "collectorOp",
                "#nvvm.tcgen05_mma_collectorop<discard>",
            )?;
            segments(&mut mma, &[1, 1, 1, 1, 1, 0, 0]);
            e.ops.push(mma);
        }
        Ok(e.ops)
    }
}
/// The address field is measured in 16-byte units, independently of swizzle.
fn descriptor_offset(rows: u32, k_extent: u32, mn: bool, k: u32) -> u32 {
    if mn {
        k * rows.min(64) / 8
    } else {
        let span = k_extent.min(64);
        ((k / span) * rows * span + k % span) / 8
    }
}
fn descriptor(
    e: &mut Emit<'_, '_>,
    ptr: MlirValueUse,
    byte_offset: MlirValueUse,
    layout: &CuteComposedLayoutAttr,
    rows: u32,
    k: u32,
    mn: bool,
) -> Result<MlirValueUse, String> {
    let sw = layout.0.outer();
    let ptr = e.cute_ptr(
        ptr,
        "f16",
        "smem",
        128,
        Some(format!("S<{},{},{}>", sw.bits, sw.base + 1, sw.shift)),
    )?;
    let (result, value) = e.fresh(MlirType::dialect("!cute_nvgpu.smem_desc")?);
    let mut op = e.operation("cute_nvgpu.make_umma_smem_desc", vec![ptr])?;
    op.results.push(result);
    let span = if mn { rows } else { k }.min(64);
    let native = if mn {
        format!(
            "((({span},{}),16),1):(((1,{}),{span}),0)",
            rows / span,
            span * k
        )
    } else {
        format!("(({rows},16),1):(({span},1),0)")
    };
    property(&mut op, "layout", &format!("#cute.layout<\"{native}\">"))?;
    property(
        &mut op,
        "major",
        if mn {
            "#cute_nvgpu.major<mn>"
        } else {
            "#cute_nvgpu.major<k>"
        },
    )?;
    e.ops.push(op);
    let base = e.cast(
        "builtin.unrealized_conversion_cast",
        value,
        MlirType::Integer(64),
    )?;
    // Preserve a loop-invariant allocation descriptor independently of the
    // selected tile. Complete-tile offsets cannot change swizzle phase, and
    // valid shared allocations cannot carry out of the address field.
    let offset = e.cast("arith.extui", byte_offset, MlirType::Integer(64))?;
    let shift = e.constant(64, 4)?;
    let offset = e.binary("arith.shrui", offset, shift)?;
    e.binary("arith.addi", base, offset)
}
struct Tma(bool);
impl OperationTranslation for Tma {
    fn translate(
        &self,
        ctx: &Context,
        source: Ptr<Operation>,
        input: OperationInput,
        session: &mut TranslationSession<'_>,
    ) -> Result<Vec<MlirOperation>, String> {
        let p = if self.0 {
            let o = CuteSm100TmaStoreOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        } else {
            let o = CuteSm100TmaLoadOp::wrap(source);
            o.verify(ctx).map_err(|e| e.to_string())?;
            o.plan(ctx)
        }
        .ok_or("missing SM100 TMA plan")?;
        contract(&input)?;
        let mut e = Emit {
            session,
            location: input.location,
            ops: vec![],
        };
        let a = input.operands;
        let desc = e.cute_ptr(
            a[0].clone(),
            "!cute_nvgpu.tma_descriptor_tiled",
            "generic",
            64,
            None,
        )?;
        let data = e.cute_ptr(a[1].clone(), "f16", "smem", 128, None)?;
        if self.0 {
            let mut op = e.operation(
                "cute_nvgpu.arch.copy.SM100.tma_store",
                vec![desc, data, a[2].clone(), a[3].clone(), a[4].clone()],
            )?;
            property(&mut op, "mode", "#cute_nvgpu.tma_store_mode<tiled>")?;
            segments(&mut op, &[1, 1, 3, 0]);
            e.ops.push(op);
        } else {
            let local = e.llvm_ptr(a[5].clone(), 3)?;
            let bar = if p.cta_group == 2 {
                let zero = e.constant(32, 0)?;
                let remote = e.value("nvvm.mapa", vec![local, zero], llvm_ptr_type(7)?)?;
                let bits = e.cast("llvm.ptrtoint", remote, MlirType::Integer(32))?;
                e.cast("llvm.inttoptr", bits, llvm_ptr_type(3)?)?
            } else {
                local
            };
            let bar = e.cute_ptr(bar, "i64", "smem", 8, None)?;
            // Each issuing CTA copies to its own shared tile. Group2 changes
            // completion routing, not the number of destination CTAs; the
            // ordinary two-CTA TMA operation therefore has no multicast mask.
            let args = vec![desc, data, bar, a[2].clone(), a[3].clone(), a[4].clone()];
            let mut op = e.operation("cute_nvgpu.arch.copy.SM100.tma_load", args)?;
            property(&mut op, "mode", "#cute_nvgpu.tma_load_mode<tiled>")?;
            integer_property(&mut op, "num_cta", 32, p.cta_group as i128);
            segments(&mut op, &[1, 1, 1, 3, 0, 0, 0]);
            e.ops.push(op);
        }
        Ok(e.ops)
    }
}
