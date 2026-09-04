# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from eval.dedup.config import LocalNddJudgeConfig
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output, run_local_ndd_pending
from eval.dedup.validation import DedupEvaluationError


def _rubric(*, relation_type: str = "exact") -> dict[str, dict[str, Any]]:
    values: dict[str, Any] = {
        "same_duplicate_group": "yes",
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": relation_type,
        "material_difference": "none",
        "fuzzy_scope": "in_scope",
        "confidence": "0.98",
        "reason_number_change": "no",
        "reason_date_time_change": "no",
        "reason_product_version_change": "no",
        "reason_url_change": "no",
        "reason_named_entity_change": "no",
        "reason_negation_change": "no",
        "reason_code_literal_change": "no",
        "reason_code_output_change": "no",
        "reason_insertion_deletion": "no",
        "reason_boilerplate": "yes",
        "reason_parser_noise": "no",
        "reason_language_mismatch": "no",
        "reason_topic_only": "no",
        "reason_insufficient_evidence": "no",
        "reason_other_material": "no",
    }
    return {name: {"score": score, "reasoning": f"private-{name}"} for name, score in values.items()}


def _local_config(tmp_path: Path) -> SimpleNamespace:
    judge = LocalNddJudgeConfig(
        backend="local_ndd",
        model="Qwen/Qwen3.8-27B",
        model_path=tmp_path / "model",
        runner_config=tmp_path / "judge.yaml",
        ray_temp_dir=tmp_path / "ray",
        checkpoint_root=tmp_path / "checkpoints",
        num_cpus=None,
        num_gpus=1,
        max_retries=2,
        max_visible_tokens=128,
        window_tokens=32,
        window_overlap_tokens=4,
        prompt_version="dedup-judge-sarah-minhash-v1",
        schema_version="dedup-judge-output-v0",
        visible_payload_version="judge-visible-payload-v2",
    )
    return SimpleNamespace(judge=judge)


def test_adapt_ndd_output_preserves_existing_contract_without_reasoning() -> None:
    result = adapt_ndd_judge_output(_rubric())

    assert result == {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
        "confidence": 0.98,
        "reason_codes": ["BOILERPLATE"],
        "evidence": [],
    }
    assert "private" not in json.dumps(result)


def test_adapt_ndd_output_accepts_hs_ab_diagnostics_without_changing_v0_output() -> None:
    rubric = _rubric()
    rubric.update(
        {
            "dominant_overlap_source": {"score": "main_content", "reasoning": "private-overlap"},
            "primary_risk_factor": {"score": "none", "reasoning": "private-risk"},
            "evidence_quality": {"score": "sufficient", "reasoning": "private-evidence"},
        }
    )

    assert adapt_ndd_judge_output(rubric) == adapt_ndd_judge_output(_rubric())


def test_adapt_ndd_output_rejects_cross_field_inconsistency() -> None:
    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(_rubric(relation_type="unrelated"))

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_adapt_ndd_output_rejects_invalid_enum() -> None:
    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(_rubric(relation_type="invented"))

    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"


def test_adapt_ndd_output_rejects_non_discrete_confidence() -> None:
    rubric = _rubric()
    rubric["confidence"]["score"] = "0.91"

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(rubric)

    assert error.value.issue.code == "LOCAL_NDD_OUTPUT_INVALID"


def test_local_ndd_retries_only_missing_rows_and_persists_terminal_results(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": f"cp1_{index}",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": f"hash-{index}",
        }
        for index in range(2)
    ]
    prepared = {
        row["canonical_pair_id"]: {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
        for row in pending
    }
    call_sizes: list[int] = []

    class FakeRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def run(self, **kwargs: object) -> None:
            input_rows = [json.loads(line) for line in Path(str(kwargs["input_path"])).read_text().splitlines()]
            call_sizes.append(len(input_rows))
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True)
            with (output_path / "part.jsonl").open("w", encoding="utf-8") as file:
                for row in input_rows[:1]:
                    file.write(json.dumps({**row, "qwen_minhash_fuzzy_dedup_judge": _rubric()}) + "\n")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "work",
        persist=persisted.append,
        runtime_factory=FakeRuntime,
    )

    assert call_sizes == [2, 1]
    assert [row["record_type"] for row in persisted] == ["result", "result"]
    assert [row["attempts"] for row in persisted] == [1, 2]
    assert all("private" not in json.dumps(row) for row in persisted)
    retry_input = json.loads((tmp_path / "work" / "attempt_02" / "input.jsonl").read_text())
    assert retry_input["repair_feedback"]["code"] == "LOCAL_NDD_MISSING_ROW"


def test_local_ndd_duplicate_rows_exhaust_retry_budget_and_write_terminal_error(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": "cp1_duplicate",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": "hash-duplicate",
        }
    ]
    prepared = {
        "cp1_duplicate": {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
    }
    call_sizes: list[int] = []

    class DuplicateRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def run(self, **kwargs: object) -> None:
            row = json.loads(Path(str(kwargs["input_path"])).read_text())
            call_sizes.append(1)
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True)
            output = json.dumps({**row, "qwen_minhash_fuzzy_dedup_judge": _rubric()}) + "\n"
            (output_path / "part.jsonl").write_text(output + output, encoding="utf-8")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "duplicate-work",
        persist=persisted.append,
        runtime_factory=DuplicateRuntime,
    )

    assert call_sizes == [1, 1, 1]
    assert len(persisted) == 1
    assert persisted[0]["record_type"] == "error"
    assert persisted[0]["attempts"] == 3
    assert [error["validation_issue"]["code"] for error in persisted[0]["errors"]] == [
        "LOCAL_NDD_DUPLICATE_ROW",
        "LOCAL_NDD_DUPLICATE_ROW",
        "LOCAL_NDD_DUPLICATE_ROW",
    ]
    assert "private-" not in json.dumps(persisted[0])


def test_local_ndd_runtime_failures_exhaust_retry_budget_and_write_terminal_error(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": "cp1_runtime",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": "hash-runtime",
        }
    ]
    prepared = {
        "cp1_runtime": {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
    }
    calls: list[dict[str, Any] | None] = []
    lifecycle: list[str] = []

    class FailingRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            lifecycle.append("enter")
            return self

        def __exit__(self, *args: object) -> None:
            del args
            lifecycle.append("exit")

        def run(self, **kwargs: object) -> None:
            row = json.loads(Path(str(kwargs["input_path"])).read_text())
            calls.append(row["repair_feedback"])
            raise RuntimeError("sensitive provider detail must not enter artifacts or retries")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "runtime-failure-work",
        persist=persisted.append,
        runtime_factory=FailingRuntime,
    )

    assert lifecycle == ["enter", "exit"]
    assert len(persisted) == 1
    assert persisted[0]["record_type"] == "error"
    assert persisted[0]["attempts"] == 3
    assert [error["validation_issue"]["code"] for error in persisted[0]["errors"]] == [
        "RuntimeError",
        "RuntimeError",
        "RuntimeError",
    ]
    assert calls[0] is None
    assert calls[1] == {
        "code": "RuntimeError",
        "message": "local NDD runtime failed before producing a valid batch",
    }
    assert calls[2] == calls[1]
    assert "sensitive provider detail" not in json.dumps(persisted)
