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

"""Step 6 batching, retry, resume, and result accounting."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from eval.dedup.config import EvaluationConfig, LocalNddJudgeConfig
from eval.dedup.contracts import stable_record_id
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.judging.client import JudgeClient, create_judge_client
from eval.dedup.judging.local_ndd import run_local_ndd_pending
from eval.dedup.judging.payload import (
    EVIDENCE_ALIGNMENT_VERSION,
    align_evidence_offsets,
    assert_blind_payload,
    build_visible_payload,
    validate_evidence_offsets,
)
from eval.dedup.judging.schema import judge_output_schema, parse_judge_json, validate_judge_output
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

RETRY_FEEDBACK_VERSION = "local-validation-structured-v2"
JUDGE_CONTRACT_VERSION = "judge-execution-contract-v2"
_SAFE_VALIDATION_DETAIL_KEYS = frozenset(
    {
        "actual_fields",
        "column",
        "expected_fields",
        "extra_fields",
        "field",
        "index",
        "line",
        "missing_fields",
    }
)


def _parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for judging"
        raise RuntimeError(msg) from exc
    return pq


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _local_resource_hashes(config: EvaluationConfig) -> dict[str, str]:
    if not isinstance(config.judge, LocalNddJudgeConfig):
        return {}
    root = config.judge.runner_config.parent
    resources = [item for item in root.rglob("*") if item.is_file() and item.suffix in {".yaml", ".jinja", ".py"}]
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(resources)}


def _judge_contract_digest(config: EvaluationConfig) -> str:
    return sha256_json(
        {
            "contract_version": JUDGE_CONTRACT_VERSION,
            "deterministic_evidence_alignment_version": EVIDENCE_ALIGNMENT_VERSION,
            "judge_config": _jsonable(asdict(config.judge)),
            "judge_output_schema": judge_output_schema(config.judge.schema_version),
            "local_ndd_resources": _local_resource_hashes(config),
            "retry_feedback_version": RETRY_FEEDBACK_VERSION,
        }
    )


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DedupEvaluationError(
                    "CORRUPT_JUDGE_CACHE", "judge cache contains invalid JSON", path=str(path), line=line_number
                ) from exc
            pair_id = row.get("canonical_pair_id")
            require(
                isinstance(pair_id, str), "CORRUPT_JUDGE_CACHE", "judge cache row has no pair ID", line=line_number
            )
            require(
                pair_id not in result,
                "DUPLICATE_JUDGE_CACHE_KEY",
                "judge cache contains a duplicate pair",
                pair_id=pair_id,
            )
            result[pair_id] = row
    return result


def _safe_validation_issue(error: Exception) -> dict[str, Any] | None:
    if not isinstance(error, DedupEvaluationError):
        return None
    safe_details = {key: value for key, value in error.issue.details.items() if key in _SAFE_VALIDATION_DETAIL_KEYS}
    return {
        "code": error.issue.code,
        "message": error.issue.message,
        "details": safe_details,
    }


def _repair_retry_prompt(prompt: str, error: Exception) -> str:
    issue = _safe_validation_issue(error)
    feedback = issue or {
        "error_type": error.__class__.__name__,
        "message": str(error).replace("\n", " ")[:400],
    }
    failure = json.dumps(feedback, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return (
        f"{prompt}\n\nREPAIR RETRY: the previous response failed strict local validation: {failure}. "
        "Produce a fresh JSON object from the visible documents. Do not repeat the invalid response. "
        "Use every required key exactly once, only allowed enum values, and evidence=[] unless quote offsets "
        "are exact."
    )


def _judge_one(
    row: dict[str, Any],
    *,
    client: JudgeClient,
    prompt: str,
    payload: dict[str, Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    errors = []
    request_prompt = prompt
    for attempt in range(config.judge.max_retries + 1):
        raw = None
        repair_events: list[dict[str, Any]] = []
        try:
            raw = client.judge(system_prompt=request_prompt, payload=payload)
            provider_response_sha256 = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            aligned, repair_events = align_evidence_offsets(parse_judge_json(raw), payload)
            parsed = validate_judge_output(aligned, config.judge.schema_version)
            validate_evidence_offsets(parsed, payload)
            return {
                "record_type": "result",
                "evaluation_run_id": row["evaluation_run_id"],
                "sut_run_id": row["sut_run_id"],
                "canonical_pair_id": row["canonical_pair_id"],
                "canonical_pair_id_version": row["canonical_pair_id_version"],
                "judge_result_id": stable_record_id(
                    "judge-result-v2",
                    _judge_contract_digest(config),
                    row["canonical_pair_id"],
                    row["judge_payload_hash"],
                    config.judge.model,
                    config.judge.prompt_version,
                    config.judge.schema_version,
                ),
                "judge_payload_hash": row["judge_payload_hash"],
                "judge_model": config.judge.model,
                "judge_contract_digest": _judge_contract_digest(config),
                "judge_contract_version": JUDGE_CONTRACT_VERSION,
                "prompt_version": config.judge.prompt_version,
                "schema_version": config.judge.schema_version,
                "attempts": attempt + 1,
                "retried": attempt > 0,
                "provider_response_sha256": provider_response_sha256,
                "deterministic_repair_events": repair_events,
                "deterministic_repair_version": EVIDENCE_ALIGNMENT_VERSION,
                **parsed,
            }
        except Exception as exc:  # noqa: BLE001 - provider and schema failures share the frozen retry policy
            error = {
                "attempt": attempt + 1,
                "error_type": exc.__class__.__name__,
                "message": str(exc)[:1000],
            }
            if raw is not None:
                error["provider_response_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            validation_issue = _safe_validation_issue(exc)
            if validation_issue is not None:
                error["validation_issue"] = validation_issue
            if repair_events:
                error["deterministic_repair_events"] = repair_events
            errors.append(error)
            request_prompt = _repair_retry_prompt(prompt, exc)
    return {
        "record_type": "error",
        "evaluation_run_id": row["evaluation_run_id"],
        "sut_run_id": row["sut_run_id"],
        "canonical_pair_id": row["canonical_pair_id"],
        "canonical_pair_id_version": row["canonical_pair_id_version"],
        "judge_payload_hash": row["judge_payload_hash"],
        "judge_model": config.judge.model,
        "judge_contract_digest": _judge_contract_digest(config),
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "prompt_version": config.judge.prompt_version,
        "schema_version": config.judge.schema_version,
        "attempts": config.judge.max_retries + 1,
        "errors": errors,
        "status": "judge_error",
    }


def run_judging(
    config: EvaluationConfig,
    *,
    corpus_manifest: dict[str, Any],
    candidate_pairs_path: Path,
    tokenizer: TokenCounter,
    cache_path: Path,
    payloads_destination: Path,
    results_destination: Path,
    errors_destination: Path,
    judge_config_destination: Path,
) -> dict[str, int]:
    """Judge every unique payload exactly once, reusing only exact-contract cache rows."""

    pq = _parquet()
    candidate_rows = pq.read_table(candidate_pairs_path).to_pylist()
    require(candidate_rows, "EMPTY_JUDGE_QUEUE", "candidate_pairs contains no rows")
    require(
        len({row["canonical_pair_id"] for row in candidate_rows}) == len(candidate_rows),
        "DUPLICATE_JUDGE_PAIR",
        "candidate_pairs is not unique by canonical pair ID",
    )
    endpoint_ids = sorted(
        {int(row["presented_doc_a"]) for row in candidate_rows}
        | {int(row["presented_doc_b"]) for row in candidate_rows}
    )
    documents = load_documents_by_ids(corpus_manifest, endpoint_ids, columns=("text", "url", "timestamp", "language"))
    if isinstance(config.judge, LocalNddJudgeConfig):
        prompt_path = config.judge.runner_config
        prompt = ""
    else:
        resource_root = Path(__file__).resolve().parents[1] / "resources"
        prompt_suffix = config.judge.prompt_version.rsplit("-", 1)[-1]
        prompt_path = resource_root / f"judge_prompt_{prompt_suffix}.txt"
        require(prompt_path.is_file(), "JUDGE_PROMPT_NOT_FOUND", "configured judge prompt file is missing")
        prompt = prompt_path.read_text(encoding="utf-8")
    payload_rows = []
    prepared: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        doc_a = int(row["presented_doc_a"])
        doc_b = int(row["presented_doc_b"])
        payload, payload_hash = build_visible_payload(
            documents[doc_a], documents[doc_b], counter=tokenizer, config=config.judge
        )
        assert_blind_payload(payload)
        require(
            payload_hash == row["judge_payload_hash"],
            "JUDGE_PAYLOAD_HASH_MISMATCH",
            "Step 6 payload differs from frozen Step 5 payload",
            canonical_pair_id=row["canonical_pair_id"],
        )
        prepared[row["canonical_pair_id"]] = payload
        payload_rows.append(
            {
                "canonical_pair_id": row["canonical_pair_id"],
                "judge_payload_hash": payload_hash,
                "payload": payload,
            }
        )
    write_text_atomic(
        payloads_destination,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in payload_rows),
    )
    judge_manifest = {
        "schema_version": "judge-config-v1",
        "backend": config.judge.backend,
        "model": config.judge.model,
        "judge_contract_digest": _judge_contract_digest(config),
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "provider_response_storage": "sha256_only",
        "deterministic_evidence_alignment": {
            "version": EVIDENCE_ALIGNMENT_VERSION,
            "scope": "evidence_only",
            "unalignable_evidence": "drop",
        },
        "retry_feedback_version": RETRY_FEEDBACK_VERSION,
        "max_visible_tokens": config.judge.max_visible_tokens,
        "window_tokens": config.judge.window_tokens,
        "window_overlap_tokens": config.judge.window_overlap_tokens,
        "max_retries": config.judge.max_retries,
        "prompt_version": config.judge.prompt_version,
        "prompt_sha256": sha256_file(prompt_path),
        "visible_payload_schema_version": payload_rows[0]["payload"]["payload_schema_version"],
        "judge_schema_version": config.judge.schema_version,
        "judge_order_seed": config.seeds["judge_order_seed"],
    }
    if isinstance(config.judge, LocalNddJudgeConfig):
        judge_manifest.update(
            {
                "model_path": str(config.judge.model_path),
                "runner_config": str(config.judge.runner_config),
                "runner_resources": _local_resource_hashes(config),
                "ray_temp_dir": str(config.judge.ray_temp_dir),
                "checkpoint_root": str(config.judge.checkpoint_root),
                "num_cpus": config.judge.num_cpus,
                "num_gpus": config.judge.num_gpus,
                "credential_source": None,
                "structured_output_mode": "ndd_score_rubric_plus_local_schema",
                "evidence_policy": "empty_not_requested_from_ndd",
            }
        )
    else:
        judge_manifest.update(
            {
                "base_url": config.judge.base_url,
                "credential_source": {"kind": "environment", "name": config.judge.api_key_env, "value_stored": False},
                "structured_output_mode": config.judge.structured_output_mode,
                "thinking": config.judge.thinking,
                "temperature": config.judge.temperature,
                "top_p": config.judge.top_p,
                "max_output_tokens": config.judge.max_output_tokens,
            }
        )
    write_json_atomic(judge_config_destination, judge_manifest)
    cache = _read_cache(cache_path)
    for row in cache.values():
        require(
            row.get("judge_payload_hash")
            == next(
                item["judge_payload_hash"]
                for item in candidate_rows
                if item["canonical_pair_id"] == row["canonical_pair_id"]
            ),
            "JUDGE_CACHE_CONTRACT_MISMATCH",
            "cached result belongs to a different payload",
            canonical_pair_id=row["canonical_pair_id"],
        )
        require(row.get("judge_model") == config.judge.model, "JUDGE_CACHE_CONTRACT_MISMATCH", "cached model differs")
        require(
            row.get("judge_contract_digest") == _judge_contract_digest(config),
            "JUDGE_CACHE_CONTRACT_MISMATCH",
            "cached judge execution contract differs",
        )
        require(
            row.get("prompt_version") == config.judge.prompt_version,
            "JUDGE_CACHE_CONTRACT_MISMATCH",
            "cached prompt differs",
        )
        require(
            row.get("schema_version") == config.judge.schema_version,
            "JUDGE_CACHE_CONTRACT_MISMATCH",
            "cached schema differs",
        )
    pending = [row for row in candidate_rows if row["canonical_pair_id"] not in cache]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def persist(result: dict[str, Any]) -> None:
        with write_lock, cache_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
            cache[result["canonical_pair_id"]] = result

    if isinstance(config.judge, LocalNddJudgeConfig):
        run_local_ndd_pending(
            config,
            pending=pending,
            prepared=prepared,
            contract_digest=_judge_contract_digest(config),
            contract_version=JUDGE_CONTRACT_VERSION,
            work_root=results_destination.parents[1] / "local_ndd_runtime",
            persist=persist,
        )
    else:
        client = create_judge_client(config.judge)
        with ThreadPoolExecutor(max_workers=config.judge.concurrency) as executor:
            futures = {
                executor.submit(
                    _judge_one,
                    row,
                    client=client,
                    prompt=prompt,
                    payload=prepared[row["canonical_pair_id"]],
                    config=config,
                ): row["canonical_pair_id"]
                for row in pending
            }
            for future in as_completed(futures):
                persist(future.result())
    require(
        len(cache) == len(candidate_rows),
        "JUDGE_ACCOUNTING_MISMATCH",
        "not every candidate has a terminal judge record",
    )
    ordered = [cache[row["canonical_pair_id"]] for row in candidate_rows]
    results = [row for row in ordered if row["record_type"] == "result"]
    errors = [row for row in ordered if row["record_type"] == "error"]
    write_text_atomic(
        results_destination,
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n" for row in results),
    )
    write_text_atomic(
        errors_destination,
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n" for row in errors),
    )
    repaired_results = sum(bool(row.get("deterministic_repair_events")) for row in results)
    repair_events = sum(len(row.get("deterministic_repair_events", [])) for row in results)
    unresolved = sum(row["same_duplicate_group"] == "UNRESOLVED" for row in results)
    retried = sum(bool(row["retried"]) for row in results)
    return {
        "requested": len(candidate_rows),
        "valid": len(results),
        "unresolved": unresolved,
        "retried": retried,
        "failed": len(errors),
        "deterministically_repaired_results": repaired_results,
        "deterministic_repair_events": repair_events,
    }
