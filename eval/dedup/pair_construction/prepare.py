# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build the existing 5a/5b evaluation population from a validated Parquet handoff."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from eval.dedup.core.config import DatasetConfig, EvaluationConfig, load_config
from eval.dedup.core.validation import require, write_json_atomic
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.handoff.sut import load_sut_arrays
from eval.dedup.pair_construction.anchors import sample_anchors
from eval.dedup.pair_construction.canonicalize import canonicalize_selected_pairs
from eval.dedup.pair_construction.outcomes import build_document_outcomes
from eval.dedup.pair_construction.removal_pairs import sample_removal_pairs
from eval.dedup.pair_construction.retrieval.lexical import build_minhash_cache, choose_lsh_configuration
from eval.dedup.pair_construction.retrieval.selection import _pilot_anchor_ids, retrieve_and_select_cross_group_pairs


def preparation_config(root: Path, dataset: DatasetConfig) -> EvaluationConfig:
    """Keep the released sampling, retrieval, tokenizer, and visible-text budgets."""
    from eval.dedup.runtime.contract import LOGICAL_MODEL

    raw = {
        "schema_version": 1,
        "handoff_root": str(root / "preparation"),
        "output_root": str(root / "preparation/data"),
        "cache_root": str(root / "preparation/cache"),
        "verify_checksums": True,
        "dataset": asdict(dataset),
        "tokenizer": {
            "kind": "huggingface",
            "model_id": LOGICAL_MODEL,
            "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",  # pragma: allowlist secret - public model commit
            "cache_root": str(root / "preparation/tokenizer"),
        },
        "judge": {
            "backend": "nvidia_openai",
            "base_url": "https://inference-api.nvidia.com/v1",
            "model": "nvidia/qwen/qwen3.8-27b",
            "api_key_env": "NVIDIA_API_KEY",  # pragma: allowlist secret - environment variable name
            "structured_output_mode": "json_schema",
            "thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 4096,
            "concurrency": 2,
            "requests_per_minute": 30,
            "timeout_seconds": 600.0,
            "max_retries": 2,
            "max_visible_tokens": 20000,
            "window_tokens": 4096,
            "window_overlap_tokens": 512,
            "prompt_version": "dedup-judge-v0.7",
            "schema_version": "dedup-judge-output-v3",
        },
        "retrieval": {
            "backend": "gpu_cudf",
            "minhash_seed": 42,
            "char_ngram_width": 24,
            "num_hashes": 260,
            "feature_ngram_width": 5,
            "lsh_grid": [[5, 1], [6, 1], [7, 1], [8, 1]],
            "pilot_target_min": 20,
            "pilot_target_max": 50,
            "pilot_target_center": 35,
            "top_k": 50,
            "signature_chunk_rows": 16384,
            "semantic_chunk_rows": 32768,
            "max_candidates_per_anchor": 500000,
        },
        "seeds": {
            name + "_seed": 26081200 + index
            for index, name in enumerate(("pilot", "anchor", "pair", "judge_order", "qa"))
        },
        "canonical_pair_id_version": "cp1",
        "profiles": {
            "full": {
                "anchor_quotas": {
                    "singleton": 500,
                    "size_2": 125,
                    "size_3_5": 125,
                    "size_6_20": 125,
                    "size_21_plus": 125,
                },
                "removal_pair_budget": 10000,
                "cross_group_pair_budget": 10000,
                "qa_pair_budget": 200,
                "minimum_diff_budget": 0,
                "formal_v0": True,
            },
            "smoke": {
                "anchor_quotas": {"singleton": 10, "size_2": 2, "size_3_5": 2, "size_6_20": 3, "size_21_plus": 3},
                "removal_pair_budget": 50,
                "cross_group_pair_budget": 50,
                "qa_pair_budget": 100,
                "minimum_diff_budget": 0,
                "formal_v0": False,
            },
        },
    }
    path = root / "preparation/config.json"
    write_json_atomic(path, raw)
    return load_config(path)


def build_population(config: EvaluationConfig, corpus: dict, sut: dict, tokenizer: TokenCounter) -> dict:
    """Run the same construction functions for a fresh full population."""
    profile = config.profile("full")
    output = config.output_root
    outcomes, anchors = output / "document_outcomes.parquet", output / "anchors.parquet"
    removal, cross = output / "removal_pairs.parquet", output / "cross_group_pairs.parquet"
    summary = {}

    def record(stage: str, value: dict) -> None:
        summary[stage] = value
        write_json_atomic(output / f"{stage}_summary.json", value)
        print(json.dumps({"prepared_stage": stage, **value}), flush=True)

    require(
        config.dataset.expected_removals >= profile.removal_pair_budget,
        "REMOVAL_BUDGET_UNFILLED",
        "new corpus cannot fill the unchanged removal-pair budget",
        available=config.dataset.expected_removals,
        required=profile.removal_pair_budget,
    )
    signature, signature_manifest = build_minhash_cache(config, corpus_manifest=corpus, cache_dir=config.cache_root)
    record("minhash", signature_manifest)
    # Check retrieval limits before spending hours tokenizing the whole corpus.
    lexical_pilot = _preflight_lexical_pilot(config, sut, signature)
    record(
        "lexical_pilot", {"anchor_ids": lexical_pilot[0], "selected_lsh": lexical_pilot[1], "trials": lexical_pilot[2]}
    )
    record(
        "outcomes",
        build_document_outcomes(
            config,
            evaluation_manifest={"evaluation_run_id": config.dataset.dataset_version},
            corpus_manifest=corpus,
            sut_manifest=sut,
            destination=outcomes,
            tokenizer=tokenizer,
        ),
    )
    record(
        "anchors",
        sample_anchors(outcomes, profile=profile, anchor_seed=config.seeds["anchor_seed"], destination=anchors),
    )
    record(
        "removal",
        sample_removal_pairs(outcomes, profile=profile, pair_seed=config.seeds["pair_seed"], destination=removal),
    )
    require(
        summary["removal"]["rows"] == profile.removal_pair_budget,
        "REMOVAL_BUDGET_UNFILLED",
        "new corpus cannot fill the unchanged removal-pair budget",
        available=summary["removal"]["rows"],
        required=profile.removal_pair_budget,
    )
    record(
        "cross_group",
        retrieve_and_select_cross_group_pairs(
            config,
            profile=profile,
            corpus_manifest=corpus,
            outcomes_path=outcomes,
            anchors_path=anchors,
            signature_path=signature,
            signature_manifest=signature_manifest,
            destination=cross,
            retrieval_config_destination=output / "retrieval_config.json",
            lexical_pilot=lexical_pilot,
        ),
    )
    require(
        summary["cross_group"]["unique_selected_pairs"] == profile.cross_group_pair_budget,
        "CROSS_GROUP_BUDGET_UNFILLED",
        "new corpus cannot fill the unchanged cross-group budget",
        available=summary["cross_group"]["unique_selected_pairs"],
        required=profile.cross_group_pair_budget,
    )
    record(
        "pairs",
        canonicalize_selected_pairs(
            config,
            corpus_manifest=corpus,
            outcomes_path=outcomes,
            removal_pairs_path=removal,
            cross_group_pairs_path=cross,
            tokenizer=tokenizer,
            candidate_destination=output / "candidate_pairs.parquet",
            provenance_destination=output / "pair_provenance.parquet",
        ),
    )
    return summary


def _preflight_lexical_pilot(
    config: EvaluationConfig, sut_manifest: dict, signature: Path
) -> tuple[list[int], tuple[int, int], list[dict]]:
    import numpy as np

    sut = load_sut_arrays(
        config,
        groups_path=Path(sut_manifest["duplicate_groups"]["path"]),
        removals_path=Path(sut_manifest["removal_ids"]["path"]),
    )
    rows = config.dataset.expected_rows
    doc_ids = np.arange(rows, dtype=np.int64)
    group_ids = np.full(rows, -1, dtype=np.int64)
    group_sizes = np.ones(rows, dtype=np.int64)
    group_ids[sut.grouped_doc_ids] = sut.grouped_group_ids
    group_sizes[sut.grouped_doc_ids] = sut.group_sizes[np.searchsorted(sut.group_ids, sut.grouped_group_ids)]
    pilot_ids = _pilot_anchor_ids(
        doc_ids, group_ids, group_sizes, seed=config.seeds["pilot_seed"], target=min(100, rows)
    )
    selected, trials = choose_lsh_configuration(
        signature, config=config, pilot_anchor_ids=pilot_ids, predicted_group_ids=group_ids
    )
    return pilot_ids, selected, trials
