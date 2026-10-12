/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
use dialect_cute::sm100_attention_ops::{CuteSm100MmaAttr, shared_layout};
use pliron::{common_traits::Verify, context::Context};
fn qk() -> CuteSm100MmaAttr {
    CuteSm100MmaAttr {
        m: 128,
        n: 128,
        k: 64,
        cta_group: 2,
        a_mn: false,
        b_mn: false,
        a_layout: shared_layout(64, 64, false).unwrap(),
        b_layout: shared_layout(64, 64, false).unwrap(),
    }
}
#[test]
fn independent_qk_pv_layouts_verify() {
    let ctx = Context::new();
    assert!(qk().verify(&ctx).is_ok());
    let pv = CuteSm100MmaAttr {
        m: 128,
        n: 256,
        k: 32,
        cta_group: 2,
        a_mn: false,
        b_mn: true,
        a_layout: shared_layout(64, 32, false).unwrap(),
        b_layout: shared_layout(128, 32, true).unwrap(),
    };
    assert!(pv.verify(&ctx).is_ok());
}
#[test]
fn rejects_wrong_group_shape_and_major() {
    let ctx = Context::new();
    let mut p = qk();
    p.cta_group = 3;
    assert!(p.verify(&ctx).is_err());
    let mut p = qk();
    p.n = 256;
    assert!(p.verify(&ctx).is_err());
    let mut p = qk();
    p.b_mn = true;
    assert!(p.verify(&ctx).is_err());
}
#[test]
fn rejects_invalid_shared_layout_extents() {
    assert!(shared_layout(63, 64, false).is_err());
    assert!(shared_layout(64, 48, false).is_err());
    assert!(shared_layout(0, 64, false).is_err());
}

#[test]
fn tma_rejects_noncanonical_layout() {
    use dialect_cute::attributes::CuteComposedLayoutAttr;
    use dialect_cute::layout::{ComposedLayout, OffsetUnit, Swizzle};
    use dialect_cute::sm100_attention_ops::CuteSm100TmaAttr;
    let ctx = Context::new();
    for mn in [false, true] {
        assert!(
            CuteSm100TmaAttr {
                layout: shared_layout(128, 128, mn).unwrap(),
                cta_group: 1
            }
            .verify(&ctx)
            .is_ok()
        );
    }
    let layout = CuteComposedLayoutAttr(
        ComposedLayout::new(
            Swizzle::new(3, 3, 3),
            0,
            "(128,128):(128,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap(),
    );
    assert!(
        CuteSm100TmaAttr {
            layout,
            cta_group: 1
        }
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn tma_load_rejects_readonly_shared_destination() {
    use dialect_cute::sm100_attention_ops::{CuteSm100TmaAttr, CuteSm100TmaLoadOp};
    use dialect_mir::types::{MirFP16Type, MirPtrType, address_space};
    use pliron::{
        basic_block::BasicBlock,
        builtin::types::{IntegerType, Signedness},
        r#type::TypeHandle,
    };
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let half: TypeHandle = MirFP16Type::get(&ctx).into();
    let u32t: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let i32t: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
    let u64t: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let desc = MirPtrType::get(&mut ctx, u64t, false, address_space::GENERIC).into();
    let barrier = MirPtrType::get(&mut ctx, u64t, true, address_space::SHARED).into();
    for mutable in [false, true] {
        let destination = MirPtrType::get(&mut ctx, half, mutable, address_space::SHARED).into();
        let block = BasicBlock::new(
            &mut ctx,
            None,
            vec![desc, destination, i32t, i32t, i32t, barrier, u32t],
        );
        let arguments = block.deref(&ctx).arguments().collect();
        let op = CuteSm100TmaLoadOp::new(
            &mut ctx,
            arguments,
            CuteSm100TmaAttr {
                layout: shared_layout(128, 128, false).unwrap(),
                cta_group: 1,
            },
        );
        assert_eq!(op.verify(&ctx).is_ok(), mutable);
    }
}

#[test]
fn mma_tile_offsets_require_explicit_u32_operands() {
    use dialect_cute::sm100_attention_ops::{CuteSm100MmaSsOp, CuteSm100MmaTsOp};
    use dialect_mir::types::{MirFP16Type, MirPtrType};
    use pliron::{
        basic_block::BasicBlock,
        builtin::types::{IntegerType, Signedness},
        r#type::TypeHandle,
    };
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_cute::register(&mut ctx);
    let half: TypeHandle = MirFP16Type::get(&ctx).into();
    let u32t: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
    let u64t: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let boolt: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Unsigned).into();
    let shared: TypeHandle = MirPtrType::get(&mut ctx, half, true, 3).into();
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![u32t, shared, shared, boolt, u32t, u64t],
    );
    let a = block.deref(&ctx).arguments().collect::<Vec<_>>();
    for (offset, valid) in [(a[4], true), (a[5], false), (a[1], false)] {
        let ss = CuteSm100MmaSsOp::new(&mut ctx, vec![a[0], a[1], a[2], a[3], offset, a[4]], qk());
        assert_eq!(ss.verify(&ctx).is_ok(), valid);
        let ss = CuteSm100MmaSsOp::new(&mut ctx, vec![a[0], a[1], a[2], a[3], a[4], offset], qk());
        assert_eq!(ss.verify(&ctx).is_ok(), valid);
        let mut plan = qk();
        plan.cta_group = 1;
        plan.a_layout = shared_layout(128, 64, false).unwrap();
        plan.b_layout = plan.a_layout.clone();
        let ts = CuteSm100MmaTsOp::new(&mut ctx, vec![a[0], a[4], a[2], a[3], offset], plan);
        assert_eq!(ts.verify(&ctx).is_ok(), valid);
    }
    let missing = CuteSm100MmaSsOp::new(&mut ctx, vec![a[0], a[1], a[2], a[3]], qk());
    assert!(missing.verify(&ctx).is_err());
}
