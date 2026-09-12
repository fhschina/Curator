# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval.dedup.config import (
    HS_MINHASH_PROMPT_VERSION,
    HS_MINHASH_V061_PROMPT_VERSION,
    HS_V062_DEV_BASELINE_PROMPT_VERSION,
    HS_V062_DEV_GATE_PROMPT_VERSION,
    HS_V062_PROMPT_VERSION,
    HS_V0621_PROMPT_VERSION,
    HS_V06210_PROMPT_VERSION,
    HS_V06211_POLICY_PROMPT_VERSION,
    HS_V06211_PROMPT_VERSION,
    HS_V06212_PROMPT_VERSION,
    HS_V06212_ROUTE_PROMPT_VERSION,
    HS_V06213_EXACT_PROMPT_VERSION,
    HS_V06213_PROMPT_VERSION,
    HS_V06214_CONTROL_PROMPT_VERSION,
    HS_V06214_PROMPT_VERSION,
    HS_V06215_CONTROL_PROMPT_VERSION,
    HS_V06215_PROMPT_VERSION,
    HS_V0622_PROMPT_VERSION,
    HS_V0623_PROMPT_VERSION,
    HS_V0624_PROMPT_VERSION,
    HS_V0625_PROMPT_VERSION,
    HS_V0626_PROMPT_VERSION,
    HS_V0627_PROMPT_VERSION,
    HS_V0628_PROMPT_VERSION,
    HS_V0629_PROMPT_VERSION,
)
from eval.dedup.rejudge_comparison import (
    CORE_FIELDS,
    DEFAULT_RUNNER_CONFIG,
    RUNNER_CONFIG_BY_PROMPT,
    _carry_forward_failed_attempts,
    _judge_config,
    _parser,
    _read_pair_ids,
    _release_approval,
    _resolve_runner_config,
    _resource_hashes,
    _validated_preflight,
    build_agreement_summary,
    run_hub,
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


def test_preflight_command_stops_at_the_requested_technical_pilot() -> None:
    args = _parser().parse_args(["preflight", "--run-root", "/example/run", "--pairs", "20"])
    assert args.command == "preflight"
    assert args.pairs == 20
    assert _parser().parse_args(["run", "--run-root", "/example/run"]).command == "run"


@pytest.mark.parametrize(
    "change", [{"judge_contract_digest": "other"}, {"status": "failed"}, {"valid": 19}, {"requested": 10, "valid": 10}]
)
def test_preflight_resume_rejects_incomplete_wrong_contract_or_smaller_pilot(change: dict) -> None:
    value = {
        "schema_version": "dedup-rejudge-hub-preflight-v1",
        "judge_contract_digest": "contract",
        "status": "pass",
        "requested": 20,
        "valid": 20,
    }
    assert _validated_preflight(value, "contract", 20) == value
    with pytest.raises(DedupEvaluationError, match="REJUDGE_HUB_PREFLIGHT_INVALID"):
        _validated_preflight({**value, **change}, "contract", 20)


def test_preflight_rejects_empty_size_before_runtime_or_network(tmp_path: Path) -> None:
    with pytest.raises(DedupEvaluationError, match="REJUDGE_PREFLIGHT_SIZE_INVALID"):
        run_hub(tmp_path, preflight_only=True, preflight_pairs=0)


def test_resume_preserves_prior_failed_attempts_in_retry_accounting() -> None:
    result = {"canonical_pair_id": "pair-1", "attempts": 1, "retried": False}
    failed_rows = [
        {"canonical_pair_id": "pair-1", "attempts": 3},
        {"canonical_pair_id": "pair-2", "attempts": 3},
        {"canonical_pair_id": "pair-1", "attempts": 2},
    ]

    carried = _carry_forward_failed_attempts(result, failed_rows)

    assert carried["attempts"] == 6
    assert carried["retried"] is True
    assert carried["prior_failed_run_attempts"] == 5


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


def test_prepare_parser_accepts_explicit_parallel_request_limit() -> None:
    args = _parser().parse_args(["prepare", "--max-parallel-requests", "64"])

    assert args.max_parallel_requests == 64


def test_hs_v061_policy_uses_v2_contract_and_separate_score_axes(tmp_path: Path) -> None:
    runner = _resolve_runner_config(HS_MINHASH_V061_PROMPT_VERSION, None)
    manifest = {
        "settings": {
            "prompt_version": HS_MINHASH_V061_PROMPT_VERSION,
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
    assert judge.schema_version == "dedup-judge-output-v2"

    baseline = [_result("same"), _result("scope-only")]
    current = [
        {key: value for key, value in _result("same").items() if key != "fuzzy_scope"},
        {
            **{key: value for key, value in _result("scope-only").items() if key != "fuzzy_scope"},
            "expected_minhash_action": "KEEP_SEPARATE",
            "surface_evidence_sufficiency": "INSUFFICIENT",
            "expected_minhash_outcome": "FALSE_NEGATIVE_RISK",
        },
    ]
    summary, disagreements = build_agreement_summary(baseline, current)

    assert summary["primary_decision_agreement"] == 1.0
    assert summary["descriptive_taxonomy_agreement"] == 1.0
    assert summary["legacy_fuzzy_scope_comparable"] is False
    assert summary["all_core_fields_agreement"] == 1.0
    assert disagreements == []


def test_hs_v062_policy_uses_v3_semantic_only_contract(tmp_path: Path) -> None:
    runner = _resolve_runner_config(HS_V062_PROMPT_VERSION, None)
    manifest = {
        "settings": {
            "prompt_version": HS_V062_PROMPT_VERSION,
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

    assert judge.schema_version == "dedup-judge-output-v3"
    assert runner.name == "hs_v062_qwen.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v062"]).judge_policy == "hs-v062"

    v0621_runner = _resolve_runner_config(HS_V0621_PROMPT_VERSION, None)
    assert v0621_runner.name == "hs_v0621_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0621"]).judge_policy == "hs-v0621"

    v0622_runner = _resolve_runner_config(HS_V0622_PROMPT_VERSION, None)
    assert v0622_runner.name == "hs_v0622_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0622"]).judge_policy == "hs-v0622"

    v0623_runner = _resolve_runner_config(HS_V0623_PROMPT_VERSION, None)
    assert v0623_runner.name == "hs_v0623_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0623"]).judge_policy == "hs-v0623"

    v0624_runner = _resolve_runner_config(HS_V0624_PROMPT_VERSION, None)
    assert v0624_runner.name == "hs_v0624_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0624"]).judge_policy == "hs-v0624"

    v0625_runner = _resolve_runner_config(HS_V0625_PROMPT_VERSION, None)
    assert v0625_runner.name == "hs_v0625_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0625"]).judge_policy == "hs-v0625"

    v0626_runner = _resolve_runner_config(HS_V0626_PROMPT_VERSION, None)
    assert v0626_runner.name == "hs_v0626_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0626"]).judge_policy == "hs-v0626"
    v0627_runner = _resolve_runner_config(HS_V0627_PROMPT_VERSION, None)
    assert v0627_runner.name == "hs_v0627_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0627"]).judge_policy == "hs-v0627"
    v0628_runner = _resolve_runner_config(HS_V0628_PROMPT_VERSION, None)
    assert v0628_runner.name == "hs_v0628_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0628"]).judge_policy == "hs-v0628"
    v0629_runner = _resolve_runner_config(HS_V0629_PROMPT_VERSION, None)
    assert v0629_runner.name == "hs_v0629_qwen_c64.yaml"
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v0629"]).judge_policy == "hs-v0629"
    span_manifest = {
        "settings": {
            **manifest["settings"],
            "prompt_version": HS_V0626_PROMPT_VERSION,
            "runner_config": str(v0626_runner),
        }
    }
    span_judge = _judge_config(tmp_path, span_manifest)
    assert span_judge.visible_payload_version == "judge-visible-payload-v3"


def test_v062_development_variants_have_separate_immutable_prompt_resources() -> None:
    versions = (
        HS_V062_DEV_BASELINE_PROMPT_VERSION,
        HS_V062_DEV_GATE_PROMPT_VERSION,
        HS_V062_PROMPT_VERSION,
        HS_V0621_PROMPT_VERSION,
        HS_V0622_PROMPT_VERSION,
        HS_V0623_PROMPT_VERSION,
        HS_V0624_PROMPT_VERSION,
        HS_V0625_PROMPT_VERSION,
        HS_V0626_PROMPT_VERSION,
        HS_V0627_PROMPT_VERSION,
        HS_V0628_PROMPT_VERSION,
        HS_V0629_PROMPT_VERSION,
        HS_V06210_PROMPT_VERSION,
        HS_V06211_POLICY_PROMPT_VERSION,
        HS_V06211_PROMPT_VERSION,
    )
    runners = [_resolve_runner_config(version, None) for version in versions]

    assert len(set(runners)) == len(versions)
    assert all(path.is_file() for path in runners)
    assert all("expected_minhash_action" not in path.read_text() for path in runners)
    assert len({tuple(sorted(_resource_hashes(path).items())) for path in runners}) == len(versions)
    for path in runners:
        judge = yaml.safe_load(path.read_text())["execution"]["stages"][0]["judges"][0]
        expected_scores = [
            "a_can_replace_b",
            "b_can_replace_a",
            "relation_type",
            "material_difference",
            "primary_material_difference",
            "dominant_overlap_source",
            "primary_risk_factor",
            "confidence_tier",
            "quote_evidence",
        ]
        if path.name == "hs_v0622_qwen_c64.yaml":
            expected_scores = [
                "content_profile_a",
                "content_profile_b",
                "shared_content_basis",
                "hard_conflict",
                "decisive_difference_location",
                *expected_scores,
            ]
        if path.name == "hs_v0623_qwen_c64.yaml":
            expected_scores = [
                "content_profile_a",
                "content_profile_b",
                "shared_content_basis",
                "hard_conflict",
                "decisive_difference_location",
                "record_alignment",
                "non_main_difference",
                "translation_status",
                *expected_scores,
            ]
        if path.name == "hs_v0624_qwen_c64.yaml":
            expected_scores = [
                "content_profile_a",
                "content_profile_b",
                "shared_content_basis",
                "hard_conflict",
                "decisive_difference_location",
                "record_alignment",
                "non_main_difference",
                "translation_status",
                "record_identity_support",
                "overlap_scope",
                "surface_delta_type",
                *expected_scores,
            ]
        if path.name == "hs_v0625_qwen_c64.yaml":
            expected_scores = [
                "content_profile_a",
                "content_profile_b",
                "shared_content_basis",
                "hard_conflict",
                "decisive_difference_location",
                "record_alignment",
                "non_main_difference",
                "translation_status",
                "record_identity_support",
                "overlap_scope",
                "surface_delta_type",
                "boundary_delta_class",
                *expected_scores,
            ]
        if path.name in {
            "hs_v0626_qwen_c64.yaml",
            "hs_v0627_qwen_c64.yaml",
            "hs_v0628_qwen_c64.yaml",
            "hs_v0629_qwen_c64.yaml",
            "hs_v06210_qwen_c64.yaml",
            "hs_v06211_policy_qwen_c64.yaml",
            "hs_v06211_qwen_c64.yaml",
            "hs_v06212_route_qwen_c64.yaml",
            "hs_v06212_qwen_c64.yaml",
            "hs_v06213_exact_qwen_c64.yaml",
            "hs_v06213_qwen_c64.yaml",
            "hs_v06214_arbitration_qwen_c64.yaml",
            "hs_v06214_qwen_c64.yaml",
            "hs_v06215_arbitration_qwen_c64.yaml",
            "hs_v06215_qwen_c64.yaml",
        }:
            expected_scores = [
                "span_content_profile_a",
                "span_content_profile_b",
                "span_shared_basis",
                "span_a_delta",
                "span_b_delta",
                "span_hard_conflict",
                "span_translation_status",
                "a_can_replace_b",
                "b_can_replace_a",
                "relation_type",
                "material_difference",
                "primary_material_difference",
                "dominant_overlap_source",
                "primary_risk_factor",
                "confidence_tier",
            ]
        assert [score["name"] for score in judge["scores"]] == expected_scores
        if path.name in {
            "hs_v0627_qwen_c64.yaml",
            "hs_v0628_qwen_c64.yaml",
            "hs_v0629_qwen_c64.yaml",
        }:
            judges = yaml.safe_load(path.read_text())["execution"]["stages"][0]["judges"]
            assert len(judges) == 2
            assert [score["name"] for score in judges[1]["scores"]] == ["record_binding_verdict"]


def test_pair_subset_loader_accepts_calibration_csv_and_holdout_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "development.csv"
    csv_path.write_text("canonical_pair_id,label\ncp1_a,YES\ncp1_b,NO\n")
    jsonl_path = tmp_path / "holdout.jsonl"
    jsonl_path.write_text('{"canonical_pair_id":"cp1_c"}\n')

    assert _read_pair_ids(csv_path) == ["cp1_a", "cp1_b"]
    assert _read_pair_ids(jsonl_path) == ["cp1_c"]


def test_v06210_runner_binds_three_independent_critic_axes_and_new_resources() -> None:
    from eval.dedup.judging.boundary_critic import BOUNDARY_CRITIC_OPTIONS

    runner = _resolve_runner_config(HS_V06210_PROMPT_VERSION, None)
    assert _parser().parse_args(["prepare", "--judge-policy", "hs-v06210"]).judge_policy == "hs-v06210"
    judges = yaml.safe_load(runner.read_text())["execution"]["stages"][0]["judges"]
    assert len(judges) == 2
    assert {
        score["name"]: {str(option).upper() for option in score["options"]} for score in judges[1]["scores"]
    } == BOUNDARY_CRITIC_OPTIONS
    resources = _resource_hashes(runner)
    assert all(judge[field] in resources for judge in judges for field in ("system_prompt_path", "prompt_path"))
    assert all("hs_v06210" in judge[field] for judge in judges for field in ("system_prompt_path", "prompt_path"))


def test_v06211_ablations_preserve_the_main_judge_and_legacy_record_score() -> None:
    baseline = yaml.safe_load(_resolve_runner_config(HS_V0629_PROMPT_VERSION, None).read_text())
    judges = baseline["execution"]["stages"][0]["judges"]
    for version, axis_count in [(HS_V06211_POLICY_PROMPT_VERSION, 2), (HS_V06211_PROMPT_VERSION, 3)]:
        current = yaml.safe_load(_resolve_runner_config(version, None).read_text())
        current_judges = current["execution"]["stages"][0]["judges"]
        assert current["models"] == baseline["models"]
        assert current_judges[0] == judges[0]
        assert current_judges[1]["scores"][0] == judges[1]["scores"][0]
        assert len(current_judges[1]["scores"]) == axis_count


def test_v06212_route_is_an_exact_prompt_control_and_final_has_its_own_proof_contract() -> None:
    from eval.dedup.judging.retained_conflict import RETAINED_CONFLICT_OPTIONS
    from eval.dedup.judging.scoped_critic import LEGACY_BINDING_OPTIONS

    baseline = yaml.safe_load(_resolve_runner_config(HS_V0629_PROMPT_VERSION, None).read_text())
    route = yaml.safe_load(_resolve_runner_config(HS_V06212_ROUTE_PROMPT_VERSION, None).read_text())
    candidate_path = _resolve_runner_config(HS_V06212_PROMPT_VERSION, None)
    candidate = yaml.safe_load(candidate_path.read_text())
    assert route["execution"]["stages"][0]["judges"] == baseline["execution"]["stages"][0]["judges"]
    assert route["models"] == candidate["models"] == baseline["models"]
    judges = candidate["execution"]["stages"][0]["judges"]
    assert {s["name"]: {str(k).upper() for k in s["options"]} for s in judges[1]["scores"]} == {
        "record_binding_verdict": LEGACY_BINDING_OPTIONS,
        "retained_conflict": RETAINED_CONFLICT_OPTIONS,
    }
    resources = _resource_hashes(candidate_path)
    assert all(
        j[field] in resources and "v06212" in j[field]
        for j in judges
        for field in ("system_prompt_path", "prompt_path")
    )
    for policy in ("hs-v06212-route", "hs-v06212"):
        args = _parser().parse_args(["prepare", "--judge-policy", policy])
        assert args.judge_policy == policy


def test_v06213_exact_control_preserves_prompts_and_final_scope_contract_is_registered() -> None:
    from eval.dedup.judging.record_scope import RECORD_SCOPE_OPTIONS

    baseline = yaml.safe_load(_resolve_runner_config(HS_V06212_PROMPT_VERSION, None).read_text())
    exact = yaml.safe_load(_resolve_runner_config(HS_V06213_EXACT_PROMPT_VERSION, None).read_text())
    path = _resolve_runner_config(HS_V06213_PROMPT_VERSION, None)
    final = yaml.safe_load(path.read_text())
    assert exact["execution"]["stages"][0]["judges"] == baseline["execution"]["stages"][0]["judges"]
    assert final["models"] == exact["models"] == baseline["models"]
    judges = final["execution"]["stages"][0]["judges"]
    scores = {s["name"]: s for s in judges[1]["scores"]}
    assert {k.upper() for k in scores["record_scope"]["options"]} == RECORD_SCOPE_OPTIONS
    resources = _resource_hashes(path)
    assert all(j[field] in resources for j in judges for field in ("system_prompt_path", "prompt_path"))
    for policy in ("hs-v06213-exact", "hs-v06213"):
        assert _parser().parse_args(["prepare", "--judge-policy", policy]).judge_policy == policy


def test_v06214_arbitration_control_preserves_v13_prompts_and_uses_fresh_contract() -> None:
    baseline = yaml.safe_load(_resolve_runner_config(HS_V06213_PROMPT_VERSION, None).read_text())
    control_path = _resolve_runner_config(HS_V06214_CONTROL_PROMPT_VERSION, None)
    control = yaml.safe_load(control_path.read_text())
    final_path = _resolve_runner_config(HS_V06214_PROMPT_VERSION, None)
    final = yaml.safe_load(final_path.read_text())
    assert control["execution"]["stages"][0]["judges"] == baseline["execution"]["stages"][0]["judges"]
    assert final["models"] == control["models"] == baseline["models"]
    assert _resource_hashes(final_path) != _resource_hashes(control_path)
    for policy in ("hs-v06214-arbitration", "hs-v06214"):
        assert _parser().parse_args(["prepare", "--judge-policy", policy]).judge_policy == policy


def test_v06215_preserves_main_and_control_but_versions_critic_contract() -> None:
    baseline = yaml.safe_load(_resolve_runner_config(HS_V06214_PROMPT_VERSION, None).read_text())
    control = yaml.safe_load(_resolve_runner_config(HS_V06215_CONTROL_PROMPT_VERSION, None).read_text())
    path = _resolve_runner_config(HS_V06215_PROMPT_VERSION, None)
    final = yaml.safe_load(path.read_text())
    old_judges = baseline["execution"]["stages"][0]["judges"]
    new_judges = final["execution"]["stages"][0]["judges"]
    assert control["execution"]["stages"][0]["judges"] == old_judges
    assert new_judges[0] == old_judges[0]
    assert new_judges[1] != old_judges[1]
    assert final["models"] == control["models"] == baseline["models"]
    resources = _resource_hashes(path)
    assert all(j[field] in resources for j in new_judges for field in ("system_prompt_path", "prompt_path"))
    for policy in ("hs-v06215-arbitration", "hs-v06215"):
        assert _parser().parse_args(["prepare", "--judge-policy", policy]).judge_policy == policy


def test_release_approval_requires_passed_one_look_holdout(tmp_path: Path) -> None:
    path = tmp_path / "approval.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "dedup-v062-release-approval-v1",
                "status": "PASSED",
                "prompt_version": HS_V062_PROMPT_VERSION,
                "judge_contract_digest": "contract",
                "holdout_manifest_sha256": "manifest",
                "holdout_evaluated_once": True,
                "gates": {"passed": True},
            }
        )
    )

    assert _release_approval(path, prompt_version=HS_V062_PROMPT_VERSION)["status"] == "PASSED"
