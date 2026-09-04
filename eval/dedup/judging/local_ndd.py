# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sarah/NDD batch execution and adaptation for dedup evaluation Step 6."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from eval.dedup.config import HS_MINHASH_PROMPT_VERSION, EvaluationConfig, LocalNddJudgeConfig
from eval.dedup.contracts import canonical_json_bytes, stable_record_id
from eval.dedup.judging.payload import align_evidence_offsets
from eval.dedup.judging.schema import validate_judge_output
from eval.dedup.validation import DedupEvaluationError, require, sha256_json, write_text_atomic

JUDGE_COLUMN = "qwen_minhash_fuzzy_dedup_judge"

_REASON_FIELDS = {
    "reason_number_change": "NUMBER_CHANGE",
    "reason_date_time_change": "DATE_TIME_CHANGE",
    "reason_product_version_change": "PRODUCT_VERSION_CHANGE",
    "reason_url_change": "URL_CHANGE",
    "reason_named_entity_change": "NAMED_ENTITY_CHANGE",
    "reason_negation_change": "NEGATION_CHANGE",
    "reason_code_literal_change": "CODE_LITERAL_CHANGE",
    "reason_code_output_change": "CODE_OUTPUT_CHANGE",
    "reason_insertion_deletion": "INSERTION_DELETION",
    "reason_boilerplate": "BOILERPLATE",
    "reason_parser_noise": "PARSER_NOISE",
    "reason_language_mismatch": "LANGUAGE_MISMATCH",
    "reason_topic_only": "TOPIC_ONLY",
    "reason_insufficient_evidence": "INSUFFICIENT_EVIDENCE",
    "reason_other_material": "OTHER_MATERIAL",
}
_CORE_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
)

_CONFIDENCE_VALUES = {
    "0.2": 0.2,
    "0.5": 0.5,
    "0.75": 0.75,
    "0.9": 0.9,
    "0.98": 0.98,
}

_EVIDENCE_REASONING_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
)
_QUOTED_EVIDENCE_PATTERN = re.compile(
    r"(?:\bdocument\s+)?(?P<side>[AB])\s*:\s*"
    r'(?:"(?P<straight>[^"\n]{1,240})"|\u201c(?P<curly>[^\u201d\n]{1,240})\u201d)',
    re.IGNORECASE,
)


def _score(judge: dict[str, Any], name: str) -> Any:
    value = judge.get(name)
    require(
        isinstance(value, dict) and set(value) >= {"score", "reasoning"},
        "LOCAL_NDD_OUTPUT_INVALID",
        "NDD rubric result is missing score or reasoning",
        field=name,
    )
    return value["score"]


def _quoted_evidence(
    judge: dict[str, Any],
    visible_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if visible_payload is None:
        return []
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for field in _EVIDENCE_REASONING_FIELDS:
        rubric = judge.get(field)
        reasoning = rubric.get("reasoning") if isinstance(rubric, dict) else None
        if not isinstance(reasoning, str):
            continue
        for match in _QUOTED_EVIDENCE_PATTERN.finditer(reasoning):
            side = match.group("side").upper()
            quote = (match.group("straight") or match.group("curly")).strip()
            key = (side, quote)
            if not quote or key in seen:
                continue
            aligned, _ = align_evidence_offsets(
                {"evidence": [{"side": side, "start_char": 0, "end_char": 0, "quote": quote}]},
                visible_payload,
            )
            if not aligned["evidence"]:
                continue
            evidence.append(aligned["evidence"][0])
            seen.add(key)

    selected: list[dict[str, Any]] = []
    for side in ("A", "B"):
        first = next((item for item in evidence if item["side"] == side), None)
        if first is not None:
            selected.append(first)
    selected.extend(item for item in evidence if item not in selected)
    return selected[:4]


def adapt_ndd_judge_output(
    value: Any,
    *,
    visible_payload: dict[str, Any] | None = None,
    require_quote_evidence: bool = False,
) -> dict[str, Any]:
    """Convert Sarah's NDD score objects to the existing v0 judge contract."""

    require(isinstance(value, dict), "LOCAL_NDD_OUTPUT_INVALID", "NDD judge result must be an object")
    parsed: dict[str, Any] = {}
    for field in _CORE_FIELDS:
        score = _score(value, field)
        require(isinstance(score, str), "LOCAL_NDD_OUTPUT_INVALID", "categorical score must be text", field=field)
        parsed[field] = score.upper()
    confidence = _score(value, "confidence")
    confidence_key = str(confidence)
    require(
        not isinstance(confidence, bool) and confidence_key in _CONFIDENCE_VALUES,
        "LOCAL_NDD_OUTPUT_INVALID",
        "confidence score must use the configured discrete rubric",
        field="confidence",
    )
    parsed["confidence"] = _CONFIDENCE_VALUES[confidence_key]
    parsed["reason_codes"] = [
        reason for field, reason in _REASON_FIELDS.items() if str(_score(value, field)).lower() == "yes"
    ]
    parsed["evidence"] = _quoted_evidence(value, visible_payload)
    if require_quote_evidence and parsed["relation_type"] not in {"EXACT", "UNRESOLVED"}:
        require(
            {item["side"] for item in parsed["evidence"]} == {"A", "B"},
            "LOCAL_NDD_EVIDENCE_INVALID",
            'HS non-exact judgment requires exact visible quotes from both documents using A: "..." and B: "..."',
            field="same_duplicate_group",
        )
    return validate_judge_output(parsed, "dedup-judge-output-v0")


def _safe_retry_feedback(error: Exception) -> dict[str, Any]:
    if isinstance(error, DedupEvaluationError):
        details = {
            key: item
            for key, item in error.issue.details.items()
            if key in {"actual_fields", "expected_fields", "extra_fields", "field", "missing_fields"}
        }
        return {"code": error.issue.code, "message": error.issue.message, "details": details}
    return {
        "code": error.__class__.__name__,
        "message": "local NDD runtime failed before producing a valid batch",
    }


def _record_attempt_error(
    *,
    errors: dict[str, list[dict[str, Any]]],
    feedback: dict[str, dict[str, Any] | None],
    pair_ids: list[str],
    attempt: int,
    error: Exception,
) -> None:
    """Record one sanitized failure for every pair affected by an attempt."""

    issue = _safe_retry_feedback(error)
    for pair_id in pair_ids:
        errors[pair_id].append(
            {
                "attempt": attempt,
                "error_type": error.__class__.__name__,
                "message": issue["message"],
                "validation_issue": issue,
            }
        )
        feedback[pair_id] = issue


def _read_output_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for part in sorted(path.rglob("*.jsonl")):
        with part.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DedupEvaluationError(
                        "LOCAL_NDD_OUTPUT_INVALID_JSON",
                        "NDD output contains invalid JSON",
                        path=str(part),
                        line=line_number,
                    ) from exc
                require(isinstance(row, dict), "LOCAL_NDD_OUTPUT_INVALID", "NDD output row must be an object")
                rows.append(row)
    return rows


def _runtime_factory() -> Callable[..., AbstractContextManager[Any]]:
    from eval.llm_judge.run_llm_judge import LocalJudgeRuntime

    return LocalJudgeRuntime


def _result_record(
    *,
    row: dict[str, Any],
    parsed: dict[str, Any],
    raw_judge: dict[str, Any],
    config: EvaluationConfig,
    contract_digest: str,
    contract_version: str,
    attempt: int,
) -> dict[str, Any]:
    return {
        "record_type": "result",
        "evaluation_run_id": row["evaluation_run_id"],
        "sut_run_id": row["sut_run_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "canonical_pair_id_version": row["canonical_pair_id_version"],
        "judge_result_id": stable_record_id(
            "judge-result-v2",
            contract_digest,
            row["canonical_pair_id"],
            row["judge_payload_hash"],
            config.judge.model,
            config.judge.prompt_version,
            config.judge.schema_version,
        ),
        "judge_payload_hash": row["judge_payload_hash"],
        "judge_model": config.judge.model,
        "judge_contract_digest": contract_digest,
        "judge_contract_version": contract_version,
        "prompt_version": config.judge.prompt_version,
        "schema_version": config.judge.schema_version,
        "attempts": attempt,
        "retried": attempt > 1,
        "provider_response_sha256": hashlib.sha256(canonical_json_bytes(raw_judge)).hexdigest(),
        "deterministic_repair_events": [],
        "deterministic_repair_version": "visible-evidence-align-v1",
        **parsed,
    }


def _error_record(
    *,
    row: dict[str, Any],
    errors: list[dict[str, Any]],
    config: EvaluationConfig,
    contract_digest: str,
    contract_version: str,
) -> dict[str, Any]:
    return {
        "record_type": "error",
        "evaluation_run_id": row["evaluation_run_id"],
        "sut_run_id": row["sut_run_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "canonical_pair_id_version": row["canonical_pair_id_version"],
        "judge_payload_hash": row["judge_payload_hash"],
        "judge_model": config.judge.model,
        "judge_contract_digest": contract_digest,
        "judge_contract_version": contract_version,
        "prompt_version": config.judge.prompt_version,
        "schema_version": config.judge.schema_version,
        "attempts": config.judge.max_retries + 1,
        "errors": errors,
        "status": "judge_error",
    }


def run_local_ndd_pending(
    config: EvaluationConfig,
    *,
    pending: list[dict[str, Any]],
    prepared: dict[str, dict[str, Any]],
    contract_digest: str,
    contract_version: str,
    work_root: Path,
    persist: Callable[[dict[str, Any]], None],
    runtime_factory: Callable[..., AbstractContextManager[Any]] | None = None,
) -> None:
    """Run pending pairs in batches, retrying only missing or invalid rows."""

    require(isinstance(config.judge, LocalNddJudgeConfig), "INVALID_JUDGE_CONFIG", "local NDD config required")
    if not pending:
        return
    factory = runtime_factory or _runtime_factory()
    by_id = {row["canonical_pair_id"]: row for row in pending}
    remaining = list(by_id)
    errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feedback: dict[str, dict[str, Any] | None] = dict.fromkeys(remaining)
    work_root.mkdir(parents=True, exist_ok=True)

    with factory(
        config.judge.runner_config,
        model_overrides={"judge": str(config.judge.model_path)},
        ray_temp_dir=str(config.judge.ray_temp_dir),
        num_cpus=config.judge.num_cpus,
        num_gpus=config.judge.num_gpus,
    ) as runtime:
        for attempt in range(1, config.judge.max_retries + 2):
            if not remaining:
                break
            attempt_root = work_root / f"attempt_{attempt:02d}"
            input_path = attempt_root / "input.jsonl"
            output_path = attempt_root / "output"
            batch_rows = [
                {
                    "canonical_pair_id": pair_id,
                    "judge_payload_hash": by_id[pair_id]["judge_payload_hash"],
                    "payload": prepared[pair_id],
                    "repair_feedback": feedback[pair_id],
                }
                for pair_id in remaining
            ]
            write_text_atomic(
                input_path,
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in batch_rows),
            )
            batch_digest = sha256_json([[row["canonical_pair_id"], row["judge_payload_hash"]] for row in batch_rows])[
                :16
            ]
            checkpoint_path = (
                config.judge.checkpoint_root
                / by_id[remaining[0]]["evaluation_run_id"]
                / contract_digest
                / f"{work_root.name}-attempt-{attempt:02d}-{batch_digest}"
            )
            try:
                runtime.run(
                    input_path=str(input_path),
                    input_format="jsonl",
                    output_path=str(output_path),
                    output_format="jsonl",
                    checkpoint_path=str(checkpoint_path),
                    files_per_partition=1,
                )
                output_rows = _read_output_rows(output_path)
                output_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for output_row in output_rows:
                    pair_id = output_row.get("canonical_pair_id")
                    require(
                        isinstance(pair_id, str) and pair_id in by_id,
                        "LOCAL_NDD_UNEXPECTED_PAIR",
                        "NDD returned an unexpected pair ID",
                        canonical_pair_id=pair_id,
                    )
                    output_by_id[pair_id].append(output_row)
            except Exception as exc:  # noqa: BLE001 - batch failures use the same bounded retry budget
                _record_attempt_error(
                    errors=errors,
                    feedback=feedback,
                    pair_ids=remaining,
                    attempt=attempt,
                    error=exc,
                )
                continue

            next_remaining = []
            for pair_id in remaining:
                candidates = output_by_id.get(pair_id, [])
                try:
                    require(candidates, "LOCAL_NDD_MISSING_ROW", "NDD produced no output row")
                    require(len(candidates) == 1, "LOCAL_NDD_DUPLICATE_ROW", "NDD produced duplicate output rows")
                    output_row = candidates[0]
                    require(
                        output_row.get("judge_payload_hash") == by_id[pair_id]["judge_payload_hash"],
                        "LOCAL_NDD_PAYLOAD_HASH_MISMATCH",
                        "NDD output belongs to a different blind payload",
                    )
                    raw_judge = output_row.get(JUDGE_COLUMN)
                    parsed = adapt_ndd_judge_output(
                        raw_judge,
                        visible_payload=prepared[pair_id],
                        require_quote_evidence=config.judge.prompt_version == HS_MINHASH_PROMPT_VERSION,
                    )
                    persist(
                        _result_record(
                            row=by_id[pair_id],
                            parsed=parsed,
                            raw_judge=raw_judge,
                            config=config,
                            contract_digest=contract_digest,
                            contract_version=contract_version,
                            attempt=attempt,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - all row failures share the retry contract
                    _record_attempt_error(
                        errors=errors,
                        feedback=feedback,
                        pair_ids=[pair_id],
                        attempt=attempt,
                        error=exc,
                    )
                    next_remaining.append(pair_id)
            remaining = next_remaining

    for pair_id in remaining:
        persist(
            _error_record(
                row=by_id[pair_id],
                errors=errors[pair_id],
                config=config,
                contract_digest=contract_digest,
                contract_version=contract_version,
            )
        )
