/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical defining paths and the closed source-abstraction boundary.

use super::registry::{CUTE_FNS, CuteFnSpec};
#[cfg(test)]
use super::registry::{CuteFn, SubstKind};
use rustc_public::{CrateDef, ty::FnDef};

/// Classification of one callee at the closed CuTe compiler boundary.
pub(super) enum CuteIdentity {
    /// Callee is not from the cute_rs crate; normal translation proceeds.
    NotCuteCrate,
    /// A recognized entry point.
    Known(#[doc = "Registered compiler-hook specification."] &'static CuteFnSpec),
    /// Ordinary zero-cost Rust whose body is intentionally translated.
    SourceAbstraction,
    /// In the cute_rs crate but not in the table: hard error (fall-through
    /// would body-translate and scalarize the layout).
    Unknown(#[doc = "Unrecognized defining path for the rejection diagnostic."] String),
}

/// Definition modules containing ordinary Rust source abstractions rather
/// than importer-owned recognition stubs.
///
/// This is deliberately a closed list. Registered [`CUTE_FNS`] entries are
/// matched first, including compiler stubs nested below these modules.
/// Every remaining function must have a real, device-safe body, even when an
/// ergonomic method can be recognized separately. An unregistered panic-only
/// stub could otherwise fall through to body translation. Adding a module
/// requires auditing that fallback and updating the exact-list test.
const CUTE_SOURCE_ABSTRACTION_MODULES: &[&str] = &[
    "arch",
    "bf16_mma",
    "block_scaled",
    "block_scaled_mma",
    "cooperative",
    "epilogue",
    "numeric",
    "partition",
    "pipeline",
    "reduction",
    "scheduler",
    "sm100",
    "sm100_attention",
    "static_layout",
    "tensor",
    "tiled_copy",
];

/// Accept only the explicitly listed source-helper module prefixes.
fn is_source_abstraction_path(path: &str, crate_name: &str) -> bool {
    let Some(rest) = path
        .strip_prefix(crate_name)
        .and_then(|rest| rest.strip_prefix("::"))
    else {
        return false;
    };
    let first_module = rest.split("::").next().unwrap_or_default();
    CUTE_SOURCE_ABSTRACTION_MODULES.contains(&first_module)
}

/// Source-only modules whose function bodies are also callable through Rust's
/// `Fn` traits. Modules such as `tensor` and `pipeline` intentionally remain
/// excluded: their ergonomic methods include compiler-owned stubs even though
/// the surrounding modules also contain ordinary source helpers.
const CUTE_SOURCE_CALLBACK_MODULES: &[&str] = &[
    "arch",
    "bf16_mma",
    "numeric",
    "partition",
    "reduction",
    "static_layout",
];

/// Whether a function item may bypass direct CuTe call recognition.
fn is_source_callback_path(path: &str, crate_name: &str) -> bool {
    if !matches!(crate_name, "cute_rs" | "cute-rs") || !is_source_abstraction_path(path, crate_name)
    {
        return false;
    }
    let Some(relative) = path
        .strip_prefix(crate_name)
        .and_then(|path| path.strip_prefix("::"))
    else {
        return false;
    };
    // Registered hooks always take precedence, including any future hook
    // added under a currently source-only module. Normalize the crate spelling
    // in the same way for both accepted crate identities.
    if CUTE_FNS
        .iter()
        .any(|spec| spec.canonical.strip_prefix("cute_rs::") == Some(relative))
    {
        return false;
    }
    let module = relative.split("::").next().unwrap_or_default();
    CUTE_SOURCE_CALLBACK_MODULES.contains(&module)
}

/// Keep compiler-owned CuTe targets on the direct-call recognition path while
/// permitting audited real Rust helpers to serve as tensor-map callbacks.
pub(in crate::translator) fn function_item_requires_direct_dispatch(fn_def: &FnDef) -> bool {
    let crate_name = fn_def.krate().name.to_string();
    matches!(crate_name.as_str(), "cute_rs" | "cute-rs")
        && !is_source_callback_path(&canonical_path(fn_def, &crate_name), &crate_name)
}

/// Rebuild the canonical definition path by walking DefId parents,
/// mirroring `intrinsics::generated::classify_raw_intrinsic`.
pub(super) fn canonical_path(fn_def: &FnDef, crate_name: &str) -> String {
    let mut segments = Vec::new();
    let mut current = Some(fn_def.def_id());
    while let Some(def_id) = current {
        let printed = def_id.name();
        let segment = printed.as_str().rsplit("::").next().unwrap_or_default();
        segments.push(segment.to_owned());
        current = def_id.parent();
    }
    canonical_path_from_leaf_segments(crate_name, segments)
}

/// Join a DefId parent walk without erasing a same-named nested module.
pub(in crate::translator) fn canonical_path_from_leaf_segments(
    crate_name: &str,
    mut segments: Vec<String>,
) -> String {
    segments.reverse();
    // Only the first segment is the crate root. A nested module may legally
    // have the same spelling and must remain part of the identity:
    //
    // cute_rs::cute_rs::tile::load_tile != cute_rs::tile::load_tile
    if segments
        .first()
        .is_some_and(|segment| segment == crate_name)
    {
        segments.remove(0);
    }
    format!("{crate_name}::{}", segments.join("::"))
}

/// Classify a callee as a registered hook, approved source helper, foreign call or error.
pub(super) fn classify_cute_call(fn_def: &FnDef) -> CuteIdentity {
    let crate_name = fn_def.krate().name.to_string();
    if !matches!(crate_name.as_str(), "cute_rs" | "cute-rs") {
        return CuteIdentity::NotCuteCrate;
    }
    let path = canonical_path(fn_def, &crate_name);
    match CUTE_FNS.iter().find(|spec| spec.canonical == path) {
        Some(spec) => CuteIdentity::Known(spec),
        None if is_source_abstraction_path(&path, &crate_name) => CuteIdentity::SourceAbstraction,
        None => CuteIdentity::Unknown(path),
    }
}

/// Regression tests for exact source identity and semantic type preservation.
#[cfg(test)]
mod canonical_path_tests {
    use super::super::classify::{
        block_scaled_method_name_from_path, pipeline_method_name_from_path,
        scheduler_method_name_from_path, smem_mma_method_name_from_path,
        tensor_method_name_from_path,
    };
    use super::{
        canonical_path_from_leaf_segments, is_source_abstraction_path, is_source_callback_path,
    };

    /// Higher-order tensor operations may call real numeric/helper bodies,
    /// but neither registered hooks nor ergonomic compiler stubs may escape
    /// the direct-call recognition boundary through a function item.
    #[test]
    fn function_items_allow_only_audited_source_bodies() {
        for path in [
            "cute_rs::numeric::silu_bf16",
            "cute_rs::numeric::{impl#3}::to_f32",
            "cute-rs::numeric::Bf16::from_f32",
            "cute_rs::arch::exp2_f32",
            "cute_rs::partition::{impl#3}::get",
            "cute_rs::static_layout::{impl#1}::offset",
            "cute_rs::bf16_mma::{impl#1}::get_slice",
            "cute_rs::reduction::{impl#0}::observe",
        ] {
            let crate_name = path.split_once("::").unwrap().0;
            assert!(is_source_callback_path(path, crate_name), "rejected {path}");
        }
        for path in [
            "cute_rs::tile::load_tile",
            "cute_rs::tensor::make_tensor_read",
            "cute_rs::tensor::{impl#7}::load",
            "cute_rs::pipeline::{impl#4}::advance",
            "cute_rs::sm100::tcgen05_alloc",
            "cute_rs::numeric_extra::silu_bf16",
            "cute_rs::other::numeric::silu_bf16",
            "other::numeric::silu_bf16",
        ] {
            assert!(!is_source_callback_path(path, "cute_rs"), "accepted {path}");
        }
        for spec in super::CUTE_FNS {
            assert!(
                !is_source_callback_path(spec.canonical, "cute_rs"),
                "registered compiler hook escaped direct dispatch: {}",
                spec.canonical
            );
        }
    }

    /// Check that path normalization removes only the crate-root segment.
    #[test]
    fn removes_only_the_crate_root_segment() {
        let path = canonical_path_from_leaf_segments(
            "cute_rs",
            ["load_tile", "tile", "cute_rs", "cute_rs"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        );
        assert_eq!(path, "cute_rs::cute_rs::tile::load_tile");
    }

    /// Check that cache-aware stores match only their exact compiler entry point.
    #[test]
    fn cache_aware_global_store_has_an_exact_compiler_boundary() {
        let spec = super::CUTE_FNS
            .iter()
            .find(|spec| spec.canonical == "cute_rs::tile::store_global_no_allocate")
            .expect("global store marker");
        assert_eq!(spec.func, super::CuteFn::StoreGlobalNoAllocate);
        assert_eq!(
            spec.schema,
            &[super::SubstKind::Type, super::SubstKind::Const]
        );
    }

    /// Check that only known source modules are recognized.
    #[test]
    fn source_abstraction_modules_are_exact_and_closed() {
        assert!(is_source_abstraction_path(
            "cute_rs::arch::assume_warp_uniform",
            "cute_rs"
        ));
        assert_eq!(
            super::CUTE_SOURCE_ABSTRACTION_MODULES,
            &[
                "arch",
                "bf16_mma",
                "block_scaled",
                "block_scaled_mma",
                "cooperative",
                "epilogue",
                "numeric",
                "partition",
                "pipeline",
                "reduction",
                "scheduler",
                "sm100",
                "sm100_attention",
                "static_layout",
                "tensor",
                "tiled_copy",
            ]
        );
        assert!(is_source_abstraction_path(
            "cute_rs::block_scaled::tensor::BlockScaledTensor<...>::thread_row",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute-rs::numeric::PackedE2M1x2::to_f32x2",
            "cute-rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::tensor::Tensor<...>::layout",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::tiled_copy::TiledCopy<...>::copy",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::block_scaled_mma::tiled::Mxfp4TiledMma<...>::gemm",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::epilogue::Sm120EpilogueWarp128x128::store_atom",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::pipeline::TmaLoadPipeline<...>::consumer_wait",
            "cute_rs"
        ));
        assert!(is_source_abstraction_path(
            "cute_rs::scheduler::StaticPersistentTileScheduler<...>::current",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::tile::load_tile",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::block_scaled_extra::load",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "other::block_scaled::load",
            "cute_rs"
        ));
    }

    /// Tensor partitions and typed BF16 atoms have real Rust bodies; adjacent
    /// modules and same-named modules in other crates must remain rejected.
    #[test]
    fn typed_partition_and_bf16_modules_have_closed_source_boundaries() {
        for (module, method) in [
            ("cooperative", "SharedRow<...>::zip_fold"),
            ("partition", "StridedTensor<...>::partition"),
            ("static_layout", "L<...>::offset"),
            ("bf16_mma", "Bf16Mma16x8x16::get_slice"),
            ("reduction", "argmax_bf16"),
        ] {
            for crate_name in ["cute_rs", "cute-rs"] {
                assert!(is_source_abstraction_path(
                    &format!("{crate_name}::{module}::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("{crate_name}::{module}_extra::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("{crate_name}::other::{module}::{method}"),
                    crate_name
                ));
                assert!(!is_source_abstraction_path(
                    &format!("other::{module}::{method}"),
                    crate_name
                ));
            }
        }
        // Hardware/compiler stubs continue to require explicit registration.
        assert!(!is_source_abstraction_path(
            "cute_rs::mma::unregistered_instruction",
            "cute_rs"
        ));
        assert!(!is_source_abstraction_path(
            "cute_rs::tile::unregistered_copy",
            "cute_rs"
        ));
    }

    /// Check that tensor methods accept the supported rustc impl-owner spellings.
    #[test]
    fn tensor_method_path_accepts_rustc_impl_owner_spellings() {
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::Contiguous1D>::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::{impl#6}::from_slice"),
            Some("from_slice")
        );
        assert_eq!(
            tensor_method_name_from_path("cute-rs::tensor::Tile1D<TILE>>::store_linear"),
            Some("store_linear")
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::tensor::from_slice"),
            None,
            "a free function is not an inherent method"
        );
        assert_eq!(
            tensor_method_name_from_path("cute_rs::other::Tile1D<TILE>>::base"),
            None
        );
    }

    /// Check that block-scaled method matching ignores only the impl-owner spelling.
    #[test]
    fn block_scaled_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::tensor::KMajor<Mode>>::from_slices"
            ),
            Some("from_slices")
        );
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::gemv::{impl#21}::dot_accumulate"
            ),
            Some("dot_accumulate")
        );
        assert_eq!(
            block_scaled_method_name_from_path(
                "cute_rs::block_scaled::__compiler::block_scaled_dot_k64"
            ),
            None,
            "compiler hooks are recognized only by the exact helper table"
        );
        assert_eq!(
            block_scaled_method_name_from_path("cute_rs::other::{impl#1}::load"),
            None
        );
    }

    /// Check that scheduler method matching ignores only the impl-owner spelling.
    #[test]
    fn scheduler_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            scheduler_method_name_from_path(
                "cute_rs::scheduler::StaticPersistentTileScheduler<...>>::has_work"
            ),
            Some("has_work")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::{impl#4}::coordinates"),
            Some("coordinates")
        );
        assert_eq!(
            scheduler_method_name_from_path("cute_rs::scheduler::__compiler::scheduler_has_work"),
            Some("scheduler_has_work"),
            "the exact helper table takes precedence over method classification"
        );
    }

    /// Check that pipeline method matching ignores only the impl-owner spelling.
    #[test]
    fn pipeline_method_path_ignores_only_the_impl_owner_spelling() {
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::TmaLoadPipeline<...>>::producer_expect_tx"
            ),
            Some("producer_expect_tx")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::{impl#4}::advance"),
            Some("advance")
        );
        assert_eq!(
            pipeline_method_name_from_path("cute_rs::pipeline::__compiler::pipeline_consumer_wait"),
            Some("pipeline_consumer_wait"),
            "the exact helper table takes precedence over method classification"
        );
    }

    /// Check that shared MMA matching requires the exact module and method.
    #[test]
    fn shared_mma_method_path_keeps_module_and_method_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::tiled::Mxfp4TiledMma<...>>::load_a_128",
                "block_scaled_mma::tiled"
            ),
            Some("load_a_128")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute-rs::tiled_copy::{impl#7}::from_raw_parts",
                "tiled_copy"
            ),
            Some("from_raw_parts")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::other::Mxfp4TiledMma<...>>::load_a_128",
                "block_scaled_mma::tiled"
            ),
            None
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::load_a_128",
                "block_scaled_mma::tiled"
            ),
            None,
            "a free function is not an inherent method"
        );
    }

    /// Check that epilogue matching requires each exact owner module.
    #[test]
    fn epilogue_method_paths_keep_each_owner_module_exact() {
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::Sm120Epilogue128x128::sync_ready_for_tma",
                "epilogue"
            ),
            Some("sync_ready_for_tma")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::block_scaled_mma::epilogue::{impl#9}::store_tile",
                "block_scaled_mma::epilogue"
            ),
            Some("store_tile")
        );
        assert_eq!(
            pipeline_method_name_from_path(
                "cute_rs::pipeline::TmaStorePipeline<1>::producer_commit"
            ),
            Some("producer_commit")
        );
        assert_eq!(
            smem_mma_method_name_from_path(
                "cute_rs::epilogue::Sm120EpilogueWarp128x128::store_tile",
                "block_scaled_mma::epilogue"
            ),
            None,
            "store_tile is defined in block_scaled_mma::epilogue, not epilogue"
        );
    }
}
