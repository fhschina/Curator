# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from eval.dedup.config import HS_MINHASH_PROMPT_VERSION
from eval.dedup.rejudge_comparison import (
    CORE_FIELDS,
    DEFAULT_RUNNER_CONFIG,
    RUNNER_CONFIG_BY_PROMPT,
    _judge_config,
    _parser,
    _resolve_runner_config,
    _resource_hashes,
    build_agreement_summary,
)
from eval.dedup.validation import DedupEvaluationError


def _result(pair_id: str, **overrides: str) -> dict[str, str]:
    row = {
        "canonical_pair_id": pair_id,
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
    }
    row.update(overrides)
    return row


def test_agreement_summary_pairs_only_common_valid_results() -> None:
    baseline = [_result("same"), _result("changed"), _result("baseline-only")]
    current = [
        _result("same"),
        _result("changed", same_duplicate_group="NO", relation_type="RELATED_NON_DUPLICATE"),
        _result("new-only"),
    ]

    summary, disagreements = build_agreement_summary(baseline, current)

    assert summary["baseline_valid"] == 3
    assert summary["common_valid"] == 2
    assert summary["all_core_fields_agreement"] == 0.5
    assert summary["same_duplicate_group_matrix"] == {"YES": {"NO": 1, "YES": 1}}
    assert summary["field_agreement"]["same_duplicate_group"] == 0.5
    assert summary["field_agreement"]["relation_type"] == 0.5
    unchanged = CORE_FIELDS[1:3] + CORE_FIELDS[4:]
    assert all(summary["field_agreement"][field] == 1.0 for field in unchanged)
    assert disagreements[0]["canonical_pair_id"] == "changed"
    assert disagreements[0]["changed_fields"] == ["same_duplicate_group", "relation_type"]


def test_agreement_summary_handles_no_common_valid_results() -> None:
    summary, disagreements = build_agreement_summary([_result("old")], [_result("new")])

    assert summary["common_valid"] == 0
    assert summary["all_core_fields_agreement"] is None
    assert all(value is None for value in summary["field_agreement"].values())
    assert disagreements == []


def test_resource_hashes_include_only_selected_runner_dependencies(tmp_path: Path) -> None:
    (tmp_path / "system.jinja").write_text("system")
    (tmp_path / "pair.jinja").write_text("pair")
    (tmp_path / "unrelated.jinja").write_text("unrelated")
    runner = tmp_path / "judge.yaml"
    runner.write_text("system_prompt_path: system.jinja\nprompt_path: pair.jinja\n")

    assert set(_resource_hashes(runner)) == {"judge.yaml", "pair.jinja", "system.jinja"}


def test_hs_policy_selects_hs_runner_and_manifest_contract(tmp_path: Path) -> None:
    runner = _resolve_runner_config(HS_MINHASH_PROMPT_VERSION, None)
    assert runner == RUNNER_CONFIG_BY_PROMPT[HS_MINHASH_PROMPT_VERSION].resolve()

    manifest = {
        "settings": {
            "prompt_version": HS_MINHASH_PROMPT_VERSION,
            "hub_model": "nvidia/qwen/qwen3.8-27b",
            "runner_config": str(runner),
            "ray_temp_dir": str(tmp_path / "ray"),
            "max_retries": 2,
            "max_visible_tokens": 20_000,
            "window_tokens": 4_096,
            "window_overlap_tokens": 512,
        }
    }
    judge = _judge_config(tmp_path, manifest)

    assert judge.prompt_version == HS_MINHASH_PROMPT_VERSION
    assert judge.runner_config == runner

    args = _parser().parse_args(["prepare", "--judge-policy", "hs"])
    assert args.judge_policy == "hs"
    assert args.runner_config is None


def test_hs_policy_rejects_sarah_builtin_runner() -> None:
    with pytest.raises(DedupEvaluationError) as error:
        _resolve_runner_config(HS_MINHASH_PROMPT_VERSION, DEFAULT_RUNNER_CONFIG)

    assert error.value.issue.code == "REJUDGE_PROMPT_RUNNER_MISMATCH"
