# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rejudge a frozen dedup pair population and compare it with its original Judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

from eval.dedup.analysis.comparison import build_pair_comparisons
from eval.dedup.analysis.holdout_evaluation import RELEASE_APPROVAL_SCHEMA
from eval.dedup.analysis.metrics import compute_metrics
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
    HS_V0622_PROMPT_VERSION,
    HS_V0623_PROMPT_VERSION,
    HS_V0624_PROMPT_VERSION,
    HS_V0625_PROMPT_VERSION,
    HS_V0626_PROMPT_VERSION,
    HS_V0627_PROMPT_VERSION,
    HS_V0628_PROMPT_VERSION,
    HS_V0629_PROMPT_VERSION,
    LOCAL_NDD_JUDGE_CONTRACTS,
    SARAH_MINHASH_PROMPT_VERSION,
    V062_RELEASE_PROMPT_VERSIONS,
    LocalNddJudgeConfig,
    TokenizerConfig,
)
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.judging.local_ndd import run_local_ndd_pending
from eval.dedup.judging.payload import assert_blind_payload, build_visible_payload
from eval.dedup.judging.request_relay import RelayContext, RequestRelay
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic
from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

DEFAULT_SOURCE_RUN = Path("/raid/hfang/dedup_eval_runs/dedup-full-20260813T220949Z-d4c37bb483/v0_run")
DEFAULT_RUNS_ROOT = Path("/raid/hfang/ihb/runs")
LOCAL_NDD_RESOURCES = Path(__file__).resolve().parent / "resources" / "local_ndd"
RUNNER_CONFIG_BY_PROMPT = {
    SARAH_MINHASH_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "sarah_minhash_qwen.yaml",
    HS_MINHASH_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_qwen.yaml",
    HS_MINHASH_V061_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v061_qwen.yaml",
    HS_V062_DEV_BASELINE_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v062_dev_baseline_qwen.yaml",
    HS_V062_DEV_GATE_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v062_dev_gate_qwen.yaml",
    HS_V062_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v062_qwen.yaml",
    HS_V0621_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0621_qwen_c64.yaml",
    HS_V0622_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0622_qwen_c64.yaml",
    HS_V0623_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0623_qwen_c64.yaml",
    HS_V0624_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0624_qwen_c64.yaml",
    HS_V0625_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0625_qwen_c64.yaml",
    HS_V0626_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0626_qwen_c64.yaml",
    HS_V0627_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0627_qwen_c64.yaml",
    HS_V0628_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0628_qwen_c64.yaml",
    HS_V0629_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v0629_qwen_c64.yaml",
    HS_V06210_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v06210_qwen_c64.yaml",
    HS_V06211_POLICY_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v06211_policy_qwen_c64.yaml",
    HS_V06211_PROMPT_VERSION: LOCAL_NDD_RESOURCES / "hs_v06211_qwen_c64.yaml",
}
PROMPT_VERSION_BY_POLICY = {
    "sarah": SARAH_MINHASH_PROMPT_VERSION,
    "hs": HS_MINHASH_PROMPT_VERSION,
    "hs-v061": HS_MINHASH_V061_PROMPT_VERSION,
    "hs-v062-dev-baseline": HS_V062_DEV_BASELINE_PROMPT_VERSION,
    "hs-v062-dev-gate": HS_V062_DEV_GATE_PROMPT_VERSION,
    "hs-v062": HS_V062_PROMPT_VERSION,
    "hs-v0621": HS_V0621_PROMPT_VERSION,
    "hs-v0622": HS_V0622_PROMPT_VERSION,
    "hs-v0623": HS_V0623_PROMPT_VERSION,
    "hs-v0624": HS_V0624_PROMPT_VERSION,
    "hs-v0625": HS_V0625_PROMPT_VERSION,
    "hs-v0626": HS_V0626_PROMPT_VERSION,
    "hs-v0627": HS_V0627_PROMPT_VERSION,
    "hs-v0628": HS_V0628_PROMPT_VERSION,
    "hs-v0629": HS_V0629_PROMPT_VERSION,
    "hs-v06210": HS_V06210_PROMPT_VERSION,
    "hs-v06211-policy": HS_V06211_POLICY_PROMPT_VERSION,
    "hs-v06211": HS_V06211_PROMPT_VERSION,
}
SCHEMA_VERSION_BY_PROMPT = {
    SARAH_MINHASH_PROMPT_VERSION: "dedup-judge-output-v0",
    HS_MINHASH_PROMPT_VERSION: "dedup-judge-output-v0",
    HS_MINHASH_V061_PROMPT_VERSION: "dedup-judge-output-v2",
    HS_V062_DEV_BASELINE_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V062_DEV_GATE_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V062_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0621_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0622_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0623_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0624_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0625_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0626_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0627_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0628_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V0629_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V06210_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V06211_POLICY_PROMPT_VERSION: "dedup-judge-output-v3",
    HS_V06211_PROMPT_VERSION: "dedup-judge-output-v3",
}
VISIBLE_PAYLOAD_VERSION_BY_PROMPT = {
    prompt_version: (
        "judge-visible-payload-v3"
        if prompt_version
        in {
            HS_V0626_PROMPT_VERSION,
            HS_V0627_PROMPT_VERSION,
            HS_V0628_PROMPT_VERSION,
            HS_V0629_PROMPT_VERSION,
            HS_V06210_PROMPT_VERSION,
            HS_V06211_POLICY_PROMPT_VERSION,
            HS_V06211_PROMPT_VERSION,
        }
        else "judge-visible-payload-v2"
    )
    for prompt_version in SCHEMA_VERSION_BY_PROMPT
}
POLICY_LABELS = {
    "sarah": "Sarah",
    "hs": "HS",
    "hs-v061": "HS V0.6.1",
    "hs-v062-dev-baseline": "HS V0.6.2 dev baseline",
    "hs-v062-dev-gate": "HS V0.6.2 dev anchor gate",
    "hs-v062": "HS V0.6.2",
    "hs-v0621": "HS V0.6.2.1",
    "hs-v0622": "HS V0.6.2.2 semantic ledger",
    "hs-v0623": "HS V0.6.2.3 reviewed boundary ledger",
    "hs-v0624": "HS V0.6.2.4 verified identity and surface ledger",
    "hs-v0625": "HS V0.6.2.5 boundary-delta ledger",
    "hs-v0626": "HS V0.6.2.6 deterministic span-evidence ledger",
    "hs-v0627": "HS V0.6.2.7 independently verified record binding",
    "hs-v0628": "HS V0.6.2.8 calibrated record-binding critic",
    "hs-v0629": "HS V0.6.2.9 asymmetric record-binding arbitration",
    "hs-v06210": "HS V0.6.2.10 evidence-scoped boundary arbitration",
    "hs-v06211-policy": "HS V0.6.2.11 policy-only development ablation",
    "hs-v06211": "HS V0.6.2.11 scoped policy and translation review",
}
DEFAULT_RUNNER_CONFIG = RUNNER_CONFIG_BY_PROMPT[SARAH_MINHASH_PROMPT_VERSION]
DEFAULT_RAY_TEMP_DIR = Path("/raid/hfang/ihb/qr")
DEFAULT_HUB_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_HUB_MODEL = "nvidia/qwen/qwen3.8-27b"
DEFAULT_LOGICAL_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3.8-27B-FP8"
DEFAULT_TOKENIZER_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
PRIMARY_DECISION_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
)
DESCRIPTIVE_TAXONOMY_FIELDS = (
    "relation_type",
    "material_difference",
)
LEGACY_SCOPE_FIELDS = ("fuzzy_scope",)
CORE_FIELDS = PRIMARY_DECISION_FIELDS + DESCRIPTIVE_TAXONOMY_FIELDS + LEGACY_SCOPE_FIELDS
MINHASH_DIAGNOSTIC_FIELDS = (
    "expected_minhash_action",
    "surface_evidence_sufficiency",
    "expected_minhash_outcome",
)


def _parquet() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for the dedup rejudge comparison"
        raise RuntimeError(msg) from exc
    return pa, pq


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "REJUDGE_INVALID_JSON", "expected a JSON object", path=str(path))
    return value


def _release_approval(path: Path, *, prompt_version: str) -> dict[str, Any]:
    value = _read_json(path)
    require(
        value.get("schema_version") == RELEASE_APPROVAL_SCHEMA
        and value.get("status") == "PASSED"
        and value.get("prompt_version") == prompt_version
        and value.get("holdout_evaluated_once") is True
        and value.get("gates", {}).get("passed") is True,
        "REJUDGE_RELEASE_APPROVAL_INVALID",
        "full V0.6.2 requires a passed, one-look holdout approval",
        path=str(path),
    )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DedupEvaluationError(
                    "REJUDGE_INVALID_JSONL", "invalid JSONL row", path=str(path), line=line_number
                ) from exc
            require(isinstance(row, dict), "REJUDGE_INVALID_JSONL", "JSONL row must be an object")
            rows.append(row)
    return rows


def _read_pair_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        rows = _read_jsonl(path)
    pair_ids = [str(row.get("canonical_pair_id", "")) for row in rows]
    require(all(pair_ids), "REJUDGE_PAIR_SELECTION_INVALID", "selected pair IDs cannot be empty")
    require(
        len(pair_ids) == len(set(pair_ids)),
        "REJUDGE_PAIR_SELECTION_INVALID",
        "selected pair IDs must be unique",
    )
    return pair_ids


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _resource_hashes(runner_config: Path) -> dict[str, str]:
    runner_config = runner_config.resolve()
    root = runner_config.parent.resolve()
    resources = {runner_config}
    for match in re.finditer(
        r"^\s*(?:system_prompt_path|prompt_path):\s*['\"]?([^'\"\s]+)['\"]?\s*$",
        runner_config.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    ):
        prompt_path = (root / match.group(1)).resolve()
        require(
            prompt_path.is_relative_to(root) and prompt_path.is_file(),
            "REJUDGE_SOURCE_MISSING",
            "runner references a missing or external prompt resource",
            path=str(prompt_path),
        )
        resources.add(prompt_path)
    compatibility_path = root / "cutlass_compat" / "sitecustomize.py"
    if compatibility_path.is_file():
        resources.add(compatibility_path)
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(resources)}


def _source_digest() -> str:
    judging_root = Path(__file__).resolve().parent / "judging"
    return sha256_json(
        {
            "runner": sha256_file(Path(__file__)),
            **{
                name: sha256_file(judging_root / filename)
                for name, filename in {
                    "request_relay": "request_relay.py",
                    "local_ndd_adapter": "local_ndd.py",
                    "boundary_critic": "boundary_critic.py",
                    "scoped_critic": "scoped_critic.py",
                    "payload": "payload.py",
                    "schema_router": "schema.py",
                    "schema_v3": "schema_v3.py",
                }.items()
            },
        }
    )


def _policy_name(prompt_version: str) -> str:
    for name, version in PROMPT_VERSION_BY_POLICY.items():
        if version == prompt_version:
            return name
    raise DedupEvaluationError(
        "REJUDGE_PROMPT_UNSUPPORTED", "unsupported rejudge prompt", prompt_version=prompt_version
    )


def _resolve_runner_config(prompt_version: str, runner_config: Path | None) -> Path:
    schema_version = SCHEMA_VERSION_BY_PROMPT.get(prompt_version)
    require(
        schema_version is not None and (prompt_version, schema_version) in LOCAL_NDD_JUDGE_CONTRACTS,
        "REJUDGE_PROMPT_UNSUPPORTED",
        "rejudge prompt must use a supported local_ndd contract",
        prompt_version=prompt_version,
    )
    resolved = (runner_config or RUNNER_CONFIG_BY_PROMPT[prompt_version]).resolve()
    builtin_versions = {path.resolve(): version for version, path in RUNNER_CONFIG_BY_PROMPT.items()}
    require(
        resolved not in builtin_versions or builtin_versions[resolved] == prompt_version,
        "REJUDGE_PROMPT_RUNNER_MISMATCH",
        "built-in runner does not match the selected prompt version",
        runner_config=str(resolved),
        prompt_version=prompt_version,
    )
    return resolved


def _new_run_id(prompt_version: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"dedup-{_policy_name(prompt_version)}-qwen-hub-{timestamp}-{uuid.uuid4().hex[:10]}"


def _judge_config(run_root: Path, manifest: dict[str, Any]) -> LocalNddJudgeConfig:
    settings = manifest["settings"]
    prompt_version = str(settings["prompt_version"])
    runner_config = _resolve_runner_config(prompt_version, Path(settings["runner_config"]))
    return LocalNddJudgeConfig(
        backend="local_ndd",
        model=settings["hub_model"],
        model_path=Path("/external/inference-hub"),
        runner_config=runner_config,
        ray_temp_dir=Path(settings["ray_temp_dir"]),
        checkpoint_root=run_root / "checkpoints",
        num_cpus=None,
        num_gpus=0,
        max_retries=int(settings["max_retries"]),
        max_visible_tokens=int(settings["max_visible_tokens"]),
        window_tokens=int(settings["window_tokens"]),
        window_overlap_tokens=int(settings["window_overlap_tokens"]),
        prompt_version=prompt_version,
        schema_version=SCHEMA_VERSION_BY_PROMPT[prompt_version],
        visible_payload_version=VISIBLE_PAYLOAD_VERSION_BY_PROMPT[prompt_version],
    )


def _contract_digest(run_root: Path, manifest: dict[str, Any]) -> str:
    judge = _judge_config(run_root, manifest)
    stable_judge = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(judge).items()
        if key not in {"checkpoint_root", "ray_temp_dir"}
    }
    return sha256_json(
        {
            "schema_version": "dedup-rejudge-contract-v1",
            "judge": stable_judge,
            "execution": {
                key: manifest["settings"][key]
                for key in (
                    "hub_base_url",
                    "hub_model",
                    "logical_model",
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "timeout_seconds",
                    "max_parallel_requests",
                )
            },
            "tokenizer": manifest["tokenizer"],
            "runner_resources": manifest["runner_resources"],
        }
    )


def prepare_run(
    *,
    source_run_root: Path = DEFAULT_SOURCE_RUN,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    runner_config: Path | None = None,
    prompt_version: str = SARAH_MINHASH_PROMPT_VERSION,
    ray_temp_dir: Path = DEFAULT_RAY_TEMP_DIR,
    block_size: int = 500,
    max_parallel_requests: int = 8,
    run_id: str | None = None,
    pair_ids_path: Path | None = None,
    release_approval_path: Path | None = None,
) -> Path:
    """Materialize Qwen-visible payloads for exactly the frozen V0 20k pair IDs."""

    pa, pq = _parquet()
    source_run_root = source_run_root.resolve()
    runner_config = _resolve_runner_config(prompt_version, runner_config)
    source_candidates = source_run_root / "data" / "candidate_pairs.parquet"
    source_provenance = source_run_root / "data" / "pair_provenance.parquet"
    source_corpus = source_run_root / "manifests" / "corpus_manifest.json"
    source_results = source_run_root / "data" / "judge_results.jsonl"
    source_errors = source_run_root / "logs" / "judge_errors.jsonl"
    source_metrics = source_run_root / "reports" / "metrics.json"
    source_comparisons = source_run_root / "data" / "pair_comparisons.parquet"
    source_outcomes = source_run_root / "data" / "document_outcomes.parquet"
    for path in (
        source_candidates,
        source_provenance,
        source_corpus,
        source_results,
        source_errors,
        source_metrics,
        source_comparisons,
        source_outcomes,
        runner_config,
    ):
        require(path.is_file(), "REJUDGE_SOURCE_MISSING", "required source artifact is missing", path=str(path))
    require(block_size > 0, "REJUDGE_BLOCK_SIZE_INVALID", "block size must be positive")
    require(
        max_parallel_requests > 0,
        "REJUDGE_MAX_PARALLEL_REQUESTS_INVALID",
        "max parallel requests must be positive",
    )
    release_approval = None
    if prompt_version in V062_RELEASE_PROMPT_VERSIONS and pair_ids_path is None:
        require(
            release_approval_path is not None,
            "REJUDGE_RELEASE_APPROVAL_REQUIRED",
            "the full 20,000-pair V0.6.2 run requires passed holdout approval",
        )
    if release_approval_path is not None:
        release_approval_path = release_approval_path.resolve()
        require(
            release_approval_path.is_file(),
            "REJUDGE_RELEASE_APPROVAL_INVALID",
            "release approval file is missing",
            path=str(release_approval_path),
        )
        release_approval = _release_approval(release_approval_path, prompt_version=prompt_version)

    candidate_table = pq.read_table(source_candidates)
    source_candidate_rows = candidate_table.to_pylist()
    source_candidate_ids = [str(row["canonical_pair_id"]) for row in source_candidate_rows]
    require(
        len(source_candidate_rows) == 20_000 and len(set(source_candidate_ids)) == 20_000,
        "REJUDGE_SOURCE_PAIR_COUNT_MISMATCH",
        "the reference run must contain exactly 20,000 unique candidate pairs",
        rows=len(source_candidate_rows),
        unique=len(set(source_candidate_ids)),
    )
    provenance = pq.read_table(source_provenance, columns=["canonical_pair_id", "track"]).to_pylist()
    tracks: dict[str, set[str]] = defaultdict(set)
    for row in provenance:
        tracks[str(row["canonical_pair_id"])].add(str(row["track"]))
    require(
        set(tracks) == set(source_candidate_ids)
        and Counter(next(iter(value)) for value in tracks.values() if len(value) == 1) == {"5a": 10_000, "5b": 10_000}
        and all(len(value) == 1 for value in tracks.values()),
        "REJUDGE_SOURCE_TRACK_MISMATCH",
        "the reference population must contain 10k single-track 5a and 10k single-track 5b pairs",
    )
    if pair_ids_path is None:
        candidates = source_candidate_rows
        candidate_ids = source_candidate_ids
        pair_selection = {"kind": "full_frozen_population", "rows": len(candidates)}
    else:
        pair_ids_path = pair_ids_path.resolve()
        require(
            pair_ids_path.is_file(),
            "REJUDGE_PAIR_SELECTION_MISSING",
            "pair selection file is missing",
            path=str(pair_ids_path),
        )
        requested_pair_ids = _read_pair_ids(pair_ids_path)
        requested_set = set(requested_pair_ids)
        unknown = sorted(requested_set - set(source_candidate_ids))
        require(
            not unknown,
            "REJUDGE_PAIR_SELECTION_INVALID",
            "pair selection contains IDs outside the frozen population",
            unknown_pair_ids=unknown[:20],
        )
        candidates = [row for row in source_candidate_rows if str(row["canonical_pair_id"]) in requested_set]
        candidate_ids = [str(row["canonical_pair_id"]) for row in candidates]
        require(
            len(candidates) == len(requested_pair_ids),
            "REJUDGE_PAIR_SELECTION_INVALID",
            "pair selection join is incomplete",
        )
        pair_selection = {
            "kind": "frozen_pair_id_subset",
            "path": str(pair_ids_path),
            "sha256": sha256_file(pair_ids_path),
            "rows": len(candidates),
        }

    run_id = run_id or _new_run_id(prompt_version)
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is not None,
        "REJUDGE_RUN_ID_INVALID",
        "run ID must contain only letters, numbers, dots, underscores, and hyphens",
        run_id=run_id,
    )
    run_root = runs_root.resolve() / run_id
    require(not run_root.exists(), "REJUDGE_RUN_EXISTS", "run root already exists", path=str(run_root))
    run_root.mkdir(parents=True)
    run_root.chmod(0o700)
    (run_root / "data").mkdir()
    (run_root / "logs").mkdir()

    endpoint_ids = sorted(
        {int(row["presented_doc_a"]) for row in candidates} | {int(row["presented_doc_b"]) for row in candidates}
    )
    documents = load_documents_by_ids(
        _read_json(source_corpus), endpoint_ids, columns=("text", "url", "timestamp", "language")
    )
    tokenizer = TokenCounter(
        TokenizerConfig(
            kind="huggingface",
            model_id=DEFAULT_TOKENIZER_MODEL,
            revision=DEFAULT_TOKENIZER_REVISION,
            cache_root=Path("/raid/hfang/dedup_eval_cache/huggingface"),
        )
    )
    payload_config = SimpleNamespace(
        schema_version=SCHEMA_VERSION_BY_PROMPT[prompt_version],
        visible_payload_version=VISIBLE_PAYLOAD_VERSION_BY_PROMPT[prompt_version],
        max_visible_tokens=20_000,
        window_tokens=4_096,
        window_overlap_tokens=512,
    )
    payload_rows = []
    rewritten_candidates = []
    for candidate in candidates:
        pair_id = str(candidate["canonical_pair_id"])
        payload, payload_hash = build_visible_payload(
            documents[int(candidate["presented_doc_a"])],
            documents[int(candidate["presented_doc_b"])],
            counter=tokenizer,
            config=payload_config,
        )
        assert_blind_payload(payload)
        rewritten_candidates.append(
            {**candidate, "evaluation_run_id": run_id, "judge_payload_hash": payload_hash, "judge_status": "pending"}
        )
        payload_rows.append(
            {
                "canonical_pair_id": pair_id,
                "judge_payload_hash": payload_hash,
                "payload": payload,
            }
        )
    candidate_destination = run_root / "data" / "candidate_pairs.parquet"
    pq.write_table(
        pa.Table.from_pylist(rewritten_candidates, schema=candidate_table.schema),
        candidate_destination,
        compression="zstd",
    )
    payload_destination = run_root / "data" / "judge_payloads.jsonl"
    _write_jsonl(payload_destination, payload_rows)

    blocks = []
    for index in range(0, len(rewritten_candidates), block_size):
        rows = rewritten_candidates[index : index + block_size]
        blocks.append(
            {
                "block_id": f"block-{index // block_size + 1:04d}",
                "start": index,
                "stop": index + len(rows),
                "pairs": len(rows),
                "membership_sha256": sha256_json(
                    [[row["canonical_pair_id"], row["judge_payload_hash"]] for row in rows]
                ),
            }
        )
    source_artifacts = {
        "candidate_pairs": str(source_candidates),
        "pair_provenance": str(source_provenance),
        "corpus_manifest": str(source_corpus),
        "judge_results": str(source_results),
        "judge_errors": str(source_errors),
        "metrics": str(source_metrics),
        "pair_comparisons": str(source_comparisons),
        "document_outcomes": str(source_outcomes),
    }
    manifest = {
        "schema_version": "dedup-rejudge-manifest-v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run_root": str(source_run_root),
        "source_artifacts": source_artifacts,
        "source_digests": {name: sha256_file(Path(path)) for name, path in source_artifacts.items()},
        "source_pair_count": len(candidates),
        "source_population_count": len(source_candidate_rows),
        "source_pair_ids_sha256": sha256_json(sorted(candidate_ids)),
        "pair_selection": pair_selection,
        "release_approval": (
            {
                "path": str(release_approval_path),
                "sha256": sha256_file(release_approval_path),
                "holdout_manifest_sha256": release_approval["holdout_manifest_sha256"],
            }
            if release_approval is not None and release_approval_path is not None
            else None
        ),
        "candidate_pairs_sha256": sha256_file(candidate_destination),
        "judge_payloads_sha256": sha256_file(payload_destination),
        "payload_membership_sha256": sha256_json(
            [[row["canonical_pair_id"], row["judge_payload_hash"]] for row in payload_rows]
        ),
        "tokenizer": tokenizer.contract(),
        "runner_resources": _resource_hashes(runner_config),
        "implementation_sha256": _source_digest(),
        "settings": {
            "hub_base_url": DEFAULT_HUB_BASE_URL,
            "hub_model": DEFAULT_HUB_MODEL,
            "logical_model": DEFAULT_LOGICAL_MODEL,
            "api_key_env": "NVIDIA_API_KEY",
            "runner_config": str(runner_config),
            "prompt_version": prompt_version,
            "ray_temp_dir": str(ray_temp_dir.resolve()),
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 4_096,
            "timeout_seconds": 600.0,
            "max_parallel_requests": max_parallel_requests,
            "max_retries": 2,
            "max_visible_tokens": 20_000,
            "window_tokens": 4_096,
            "window_overlap_tokens": 512,
            "block_size": block_size,
        },
        "blocks": blocks,
    }
    manifest["judge_contract_digest"] = _contract_digest(run_root, manifest)
    if release_approval is not None:
        require(
            release_approval["judge_contract_digest"] == manifest["judge_contract_digest"],
            "REJUDGE_RELEASE_APPROVAL_INVALID",
            "holdout approval belongs to a different Judge contract",
        )
    write_json_atomic(run_root / "run_manifest.json", manifest)
    _write_status(
        run_root / "progress.json",
        {"status": "prepared", "completed_blocks": 0, "total_blocks": len(blocks), "completed_pairs": 0},
    )
    return run_root


def _validate_run_root(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "run_manifest.json"
    require(manifest_path.is_file(), "REJUDGE_MANIFEST_MISSING", "run manifest is missing")
    manifest = _read_json(manifest_path)
    require(
        manifest.get("schema_version") == "dedup-rejudge-manifest-v1", "REJUDGE_MANIFEST_INVALID", "schema differs"
    )
    require(manifest.get("run_id") == run_root.name, "REJUDGE_MANIFEST_INVALID", "run ID differs")
    require(_source_digest() == manifest["implementation_sha256"], "REJUDGE_SOURCE_CHANGED", "runner source changed")
    runner_config = Path(manifest["settings"]["runner_config"])
    require(
        _resource_hashes(runner_config) == manifest["runner_resources"],
        "REJUDGE_SOURCE_CHANGED",
        "Judge resources changed",
    )
    for name, path in manifest["source_artifacts"].items():
        require(
            Path(path).is_file() and sha256_file(Path(path)) == manifest["source_digests"][name],
            "REJUDGE_SOURCE_CHANGED",
            "reference artifact changed",
            path=path,
        )
    require(
        sha256_file(run_root / "data" / "candidate_pairs.parquet") == manifest["candidate_pairs_sha256"]
        and sha256_file(run_root / "data" / "judge_payloads.jsonl") == manifest["judge_payloads_sha256"],
        "REJUDGE_WORKLOAD_CHANGED",
        "prepared Qwen workload changed",
    )
    require(
        _contract_digest(run_root, manifest) == manifest["judge_contract_digest"],
        "REJUDGE_CONTRACT_CHANGED",
        "Judge contract changed",
    )
    return manifest


class _BorrowedExternalRuntime(AbstractContextManager["_BorrowedExternalRuntime"]):
    def __init__(
        self,
        runtime: ExternalJudgeRuntime,
        settings: dict[str, Any],
        relay: RequestRelay,
        *,
        block_id: str,
        events_root: Path,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.relay = relay
        self.block_id = block_id
        self.events_root = events_root
        self.outer_attempt = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def run(self, **kwargs: Any) -> Any:
        self.outer_attempt += 1
        self.relay.set_context(
            RelayContext(
                block_id=self.block_id,
                outer_attempt=self.outer_attempt,
                events_path=self.events_root / f"outer-attempt-{self.outer_attempt:02d}.jsonl",
            )
        )
        return self.runtime.run(
            **kwargs,
            inference_parameter_overrides={
                "judge": {
                    "temperature": self.settings["temperature"],
                    "top_p": self.settings["top_p"],
                    "max_tokens": self.settings["max_output_tokens"],
                    "timeout": self.settings["timeout_seconds"],
                    "max_parallel_requests": self.settings["max_parallel_requests"],
                }
            },
        )


def _cache_by_pair(path: Path, *, contract_digest: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("canonical_pair_id"))
        require(pair_id not in by_pair, "REJUDGE_CACHE_DUPLICATE", "cache contains duplicate pair", pair_id=pair_id)
        require(
            row.get("judge_contract_digest") == contract_digest,
            "REJUDGE_CACHE_CONTRACT_MISMATCH",
            "cache row belongs to another Judge contract",
            pair_id=pair_id,
        )
        by_pair[pair_id] = row
    return by_pair


def _carry_forward_failed_attempts(result: dict[str, Any], failed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_id = str(result["canonical_pair_id"])
    prior_attempts = sum(
        int(row.get("attempts", 0)) for row in failed_rows if str(row.get("canonical_pair_id")) == pair_id
    )
    if prior_attempts == 0:
        return result
    return {
        **result,
        "attempts": int(result["attempts"]) + prior_attempts,
        "retried": True,
        "prior_failed_run_attempts": prior_attempts,
    }


def run_hub(run_root: Path) -> dict[str, Any]:
    """Run pending blocks through the selected NDD prompt policy on Inference Hub."""

    _, pq = _parquet()
    run_root = run_root.resolve()
    manifest = _validate_run_root(run_root)
    settings = manifest["settings"]
    api_key = os.environ.get(settings["api_key_env"], "").strip()
    require(api_key, "REJUDGE_API_KEY_MISSING", "Inference Hub credential is unavailable")
    candidates = pq.read_table(run_root / "data" / "candidate_pairs.parquet").to_pylist()
    payload_rows = _read_jsonl(run_root / "data" / "judge_payloads.jsonl")
    prepared = {str(row["canonical_pair_id"]): row["payload"] for row in payload_rows}
    require(
        len(candidates) == len(prepared) == manifest["source_pair_count"],
        "REJUDGE_WORKLOAD_CHANGED",
        "candidate and payload counts differ",
    )
    cache_path = run_root / "logs" / "judge_cache.jsonl"
    cache = _cache_by_pair(cache_path, contract_digest=manifest["judge_contract_digest"])
    judge_config = _judge_config(run_root, manifest)
    config = SimpleNamespace(judge=judge_config)
    completed_blocks = 0
    relay = RequestRelay(
        logical_model=settings["logical_model"],
        upstream_base_url=settings["hub_base_url"],
        upstream_model=settings["hub_model"],
        upstream_api_key=api_key,
        timeout_seconds=settings["timeout_seconds"],
        expected_generation_parameters={
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_output_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    relay.start()
    runtime = ExternalJudgeRuntime(
        settings["runner_config"],
        endpoint=relay.endpoint,
        provider_api_key="unused",  # pragma: allowlist secret
        served_model_overrides={"judge": settings["logical_model"]},
        ray_temp_dir=settings["ray_temp_dir"],
        num_cpus=None,
    )
    _write_status(
        run_root / "progress.json",
        {
            "status": "running",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "completed_blocks": sum(
                all(str(row["canonical_pair_id"]) in cache for row in candidates[block["start"] : block["stop"]])
                for block in manifest["blocks"]
            ),
            "total_blocks": len(manifest["blocks"]),
            "completed_pairs": len(cache),
        },
    )
    with runtime:
        preflight_path = run_root / "hub_preflight.json"
        if not preflight_path.is_file():
            pilot_rows = candidates[:10]
            pilot_terminal: list[dict[str, Any]] = []
            pilot_root = run_root / "preflight" / f"pilot-{uuid.uuid4().hex[:8]}"
            borrowed = _BorrowedExternalRuntime(
                runtime, settings, relay, block_id="pilot", events_root=pilot_root / "events"
            )
            run_local_ndd_pending(
                config,
                pending=pilot_rows,
                prepared={
                    str(row["canonical_pair_id"]): prepared[str(row["canonical_pair_id"])] for row in pilot_rows
                },
                contract_digest=manifest["judge_contract_digest"],
                contract_version="dedup-rejudge-contract-v1",
                work_root=pilot_root / "ndd",
                persist=pilot_terminal.append,
                runtime_factory=lambda *_args, _borrowed=borrowed, **_kwargs: _borrowed,
            )
            pilot_valid = sum(row["record_type"] == "result" for row in pilot_terminal)
            require(
                pilot_valid == len(pilot_rows),
                "REJUDGE_HUB_PREFLIGHT_FAILED",
                "Inference Hub pilot did not produce a valid selected-policy result for every pair",
                requested=len(pilot_rows),
                valid=pilot_valid,
                errors=len(pilot_terminal) - pilot_valid,
            )
            write_json_atomic(
                preflight_path,
                {
                    "schema_version": "dedup-rejudge-hub-preflight-v1",
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "judge_contract_digest": manifest["judge_contract_digest"],
                    "requested": len(pilot_rows),
                    "valid": pilot_valid,
                    "status": "pass",
                },
            )
        for block in manifest["blocks"]:
            block_rows = candidates[block["start"] : block["stop"]]
            require(
                sha256_json([[row["canonical_pair_id"], row["judge_payload_hash"]] for row in block_rows])
                == block["membership_sha256"],
                "REJUDGE_WORKLOAD_CHANGED",
                "block membership changed",
                block_id=block["block_id"],
            )
            pending = [row for row in block_rows if str(row["canonical_pair_id"]) not in cache]
            if pending:
                block_root = run_root / "runtime" / block["block_id"]
                attempt_number = len(list(block_root.glob("run-*"))) + 1 if block_root.exists() else 1
                work_root = block_root / f"run-{attempt_number:03d}"
                failed_path = block_root / "failed_attempts.jsonl"
                failed_history = _read_jsonl(failed_path)

                def persist(
                    result: dict[str, Any],
                    *,
                    _failed_path: Path = failed_path,
                    _failed_history: list[dict[str, Any]] = failed_history,
                ) -> None:
                    if result["record_type"] == "result":
                        result = _carry_forward_failed_attempts(result, _failed_history)
                        _append_jsonl(cache_path, result)
                        cache[str(result["canonical_pair_id"])] = result
                    else:
                        _append_jsonl(_failed_path, result)

                borrowed = _BorrowedExternalRuntime(
                    runtime, settings, relay, block_id=block["block_id"], events_root=work_root / "events"
                )
                run_local_ndd_pending(
                    config,
                    pending=pending,
                    prepared={
                        str(row["canonical_pair_id"]): prepared[str(row["canonical_pair_id"])] for row in pending
                    },
                    contract_digest=manifest["judge_contract_digest"],
                    contract_version="dedup-rejudge-contract-v1",
                    work_root=work_root,
                    persist=persist,
                    runtime_factory=lambda *_args, _borrowed=borrowed, **_kwargs: _borrowed,
                )
                missing = [
                    str(row["canonical_pair_id"]) for row in block_rows if str(row["canonical_pair_id"]) not in cache
                ]
                require(
                    not missing,
                    "REJUDGE_BLOCK_INCOMPLETE",
                    "a block exhausted retries; stop before submitting later blocks and retry the missing subset on resume",
                    block_id=block["block_id"],
                    missing_pairs=len(missing),
                )
            completed_blocks += 1
            _write_status(
                run_root / "progress.json",
                {
                    "status": "running",
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                    "completed_blocks": completed_blocks,
                    "total_blocks": len(manifest["blocks"]),
                    "completed_pairs": len(cache),
                    "valid_pairs": sum(row["record_type"] == "result" for row in cache.values()),
                    "terminal_errors": sum(row["record_type"] == "error" for row in cache.values()),
                },
            )
    relay.stop()
    require(len(cache) == len(candidates), "REJUDGE_ACCOUNTING_MISMATCH", "not every pair has a terminal record")
    ordered = [cache[str(row["canonical_pair_id"])] for row in candidates]
    results = [row for row in ordered if row["record_type"] == "result"]
    errors = [row for row in ordered if row["record_type"] == "error"]
    _write_jsonl(run_root / "data" / "judge_results.jsonl", results)
    _write_jsonl(run_root / "logs" / "judge_errors.jsonl", errors)
    complete = {
        "schema_version": "dedup-rejudge-complete-v1",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "requested": len(candidates),
        "valid": len(results),
        "errors": len(errors),
        "retried": sum(int(row["attempts"]) > 1 for row in ordered),
        "results_sha256": sha256_file(run_root / "data" / "judge_results.jsonl"),
        "errors_sha256": sha256_file(run_root / "logs" / "judge_errors.jsonl"),
    }
    write_json_atomic(run_root / "run_complete.json", complete)
    _write_status(run_root / "progress.json", {"status": "complete", **complete})
    return complete


def _by_pair(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["canonical_pair_id"]): row for row in rows}


def build_agreement_summary(
    baseline_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute label agreement and return pair-level disagreements without document text."""

    baseline = _by_pair(baseline_rows)
    new = _by_pair(new_rows)
    common_ids = sorted(set(baseline) & set(new))
    comparable_fields = list(PRIMARY_DECISION_FIELDS + DESCRIPTIVE_TAXONOMY_FIELDS)
    if not common_ids or all(
        field in baseline[pair_id] and field in new[pair_id] for pair_id in common_ids for field in LEGACY_SCOPE_FIELDS
    ):
        comparable_fields.extend(LEGACY_SCOPE_FIELDS)
    field_agreement = {
        field: sum(baseline[pair_id].get(field) == new[pair_id].get(field) for pair_id in common_ids) / len(common_ids)
        if common_ids
        else None
        for field in comparable_fields
    }
    all_core = [
        pair_id
        for pair_id in common_ids
        if all(baseline[pair_id].get(field) == new[pair_id].get(field) for field in comparable_fields)
    ]
    primary_agree = [
        pair_id
        for pair_id in common_ids
        if all(baseline[pair_id].get(field) == new[pair_id].get(field) for field in PRIMARY_DECISION_FIELDS)
    ]
    taxonomy_agree = [
        pair_id
        for pair_id in common_ids
        if all(baseline[pair_id].get(field) == new[pair_id].get(field) for field in DESCRIPTIVE_TAXONOMY_FIELDS)
    ]
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    disagreements = []
    for pair_id in common_ids:
        old = baseline[pair_id]
        current = new[pair_id]
        matrix[str(old.get("same_duplicate_group"))][str(current.get("same_duplicate_group"))] += 1
        changed = [field for field in comparable_fields if old.get(field) != current.get(field)]
        if changed:
            disagreements.append(
                {
                    "canonical_pair_id": pair_id,
                    "changed_fields": changed,
                    **{f"baseline_{field}": old.get(field) for field in comparable_fields},
                    **{f"new_{field}": current.get(field) for field in comparable_fields},
                    **{f"new_{field}": current.get(field) for field in MINHASH_DIAGNOSTIC_FIELDS},
                }
            )
    minhash_diagnostics = {
        field: dict(sorted(Counter(str(new[pair_id].get(field)) for pair_id in common_ids).items()))
        for field in MINHASH_DIAGNOSTIC_FIELDS
        if any(new[pair_id].get(field) is not None for pair_id in common_ids)
    }
    return (
        {
            "baseline_valid": len(baseline),
            "new_valid": len(new),
            "common_valid": len(common_ids),
            "all_core_fields_agreement": len(all_core) / len(common_ids) if common_ids else None,
            "all_core_fields_agree_pairs": len(all_core),
            "comparable_fields": comparable_fields,
            "primary_decision_agreement": len(primary_agree) / len(common_ids) if common_ids else None,
            "primary_decision_agree_pairs": len(primary_agree),
            "descriptive_taxonomy_agreement": len(taxonomy_agree) / len(common_ids) if common_ids else None,
            "descriptive_taxonomy_agree_pairs": len(taxonomy_agree),
            "legacy_fuzzy_scope_comparable": "fuzzy_scope" in comparable_fields,
            "field_agreement": field_agreement,
            "new_minhash_diagnostics": minhash_diagnostics,
            "same_duplicate_group_matrix": {key: dict(value) for key, value in sorted(matrix.items())},
            "disagreement_pairs": len(disagreements),
        },
        disagreements,
    )


def _outcome_agreement(baseline_path: Path, new_path: Path) -> dict[str, Any]:
    _, pq = _parquet()
    baseline = _by_pair(pq.read_table(baseline_path).to_pylist())
    new = _by_pair(pq.read_table(new_path).to_pylist())
    result = {}
    for track, field in (("5a", "removal_outcome"), ("5b", "cross_group_outcome")):
        common = [
            pair_id
            for pair_id in sorted(set(baseline) & set(new))
            if baseline[pair_id].get(field) is not None
            and new[pair_id].get(field) is not None
            and baseline[pair_id].get(field) != "judge_error"
            and new[pair_id].get(field) != "judge_error"
        ]
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        for pair_id in common:
            matrix[str(baseline[pair_id][field])][str(new[pair_id][field])] += 1
        agreed = sum(baseline[pair_id][field] == new[pair_id][field] for pair_id in common)
        result[track] = {
            "field": field,
            "common_resolved": len(common),
            "agreement": agreed / len(common) if common else None,
            "matrix": {key: dict(value) for key, value in sorted(matrix.items())},
        }
    return result


def _metric_delta(baseline: dict[str, Any], current: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    old: Any = baseline
    new: Any = current
    for key in path:
        old = old[key]
        new = new[key]
    return {"baseline": old, "new": new, "delta": new - old if old is not None and new is not None else None}


def summarize(run_root: Path) -> dict[str, Any]:
    """Build native metrics plus the selected Qwen-policy vs. V0/DeepSeek comparison."""

    pa, _ = _parquet()
    run_root = run_root.resolve()
    manifest = _validate_run_root(run_root)
    policy_name = _policy_name(manifest["settings"]["prompt_version"])
    policy_label = POLICY_LABELS[policy_name]
    complete = _read_json(run_root / "run_complete.json")
    source = {name: Path(path) for name, path in manifest["source_artifacts"].items()}
    new_results_path = run_root / "data" / "judge_results.jsonl"
    new_errors_path = run_root / "logs" / "judge_errors.jsonl"
    new_comparisons_path = run_root / "data" / "pair_comparisons.parquet"
    build_pair_comparisons(
        candidate_pairs_path=run_root / "data" / "candidate_pairs.parquet",
        pair_provenance_path=source["pair_provenance"],
        outcomes_path=Path(manifest["source_run_root"]) / "data" / "document_outcomes.parquet",
        judge_results_path=new_results_path,
        judge_errors_path=new_errors_path,
        destination=new_comparisons_path,
    )
    stage_counts = {
        "requested": complete["requested"],
        "valid": complete["valid"],
        "failed": complete["errors"],
        "retried": complete["retried"],
        "deterministically_repaired_results": 0,
        "deterministic_repair_events": 0,
    }
    new_metrics = compute_metrics(
        new_comparisons_path,
        requested_judge_pairs=int(complete["requested"]),
        metrics_destination=run_root / "metrics.json",
        slices_destination=run_root / "metrics_by_slice.csv",
        accounting_destination=run_root / "pipeline_accounting.csv",
        stage_markers=[
            {"step": 6, "name": f"rejudge_{policy_name}_qwen_hub", "status": "complete", "counts": stage_counts}
        ],
    )
    baseline_rows = _read_jsonl(source["judge_results"])
    new_rows = _read_jsonl(new_results_path)
    agreement, disagreements = build_agreement_summary(baseline_rows, new_rows)
    disagreement_path = run_root / "data" / "pair_disagreements.parquet"
    if disagreements:
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(disagreements), disagreement_path, compression="zstd")
    baseline_metrics = _read_json(source["metrics"])
    full_population = int(complete["requested"]) == int(manifest.get("source_population_count", 20_000))
    if full_population:
        headline_metric_deltas = {
            "judge_completion_rate": _metric_delta(baseline_metrics, new_metrics, ("judge", "completion_rate")),
            "removal_precision": _metric_delta(
                baseline_metrics, new_metrics, ("track_5a_removal_frame", "removal_precision")
            ),
            "wrong_removal_rate": _metric_delta(
                baseline_metrics, new_metrics, ("track_5a_removal_frame", "wrong_removal_rate")
            ),
            "cross_group_positive_yield": _metric_delta(
                baseline_metrics, new_metrics, ("track_5b_candidate_pool", "positive_yield")
            ),
        }
    else:
        headline_metric_deltas = {
            "judge_completion_rate": {
                "baseline": None,
                "new": new_metrics["judge"]["completion_rate"],
                "delta": None,
            },
            "removal_precision": {
                "baseline": None,
                "new": new_metrics["track_5a_removal_frame"]["removal_precision"],
                "delta": None,
            },
            "wrong_removal_rate": {
                "baseline": None,
                "new": new_metrics["track_5a_removal_frame"]["wrong_removal_rate"],
                "delta": None,
            },
            "cross_group_positive_yield": {
                "baseline": None,
                "new": new_metrics["track_5b_candidate_pool"]["positive_yield"],
                "delta": None,
            },
        }
    comparison = {
        "schema_version": "dedup-framework-comparison-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "baseline": {
            "run_id": Path(manifest["source_run_root"]).parent.name,
            "framework": "dedup-judge-v0",
            "model": "nvidia/deepseek-ai/deepseek-v4-pro",
        },
        "new": {
            "run_id": manifest["run_id"],
            "framework": manifest["settings"]["prompt_version"],
            "label": f"{policy_label}/Qwen",
            "model": manifest["settings"]["hub_model"],
        },
        "scope_note": (
            "This is a paired bundle comparison: framework, model, tokenizer, and visible-context policy changed together; "
            "the observed deltas cannot be attributed to any one component. Agreement with DeepSeek and SUT-derived "
            "outcomes are diagnostics only; human calibration gates version selection."
        ),
        "population": {
            "pairs": int(complete["requested"]),
            "track_5a_pairs": int(new_metrics["track_5a_removal_frame"]["requested"]),
            "track_5b_pairs": int(new_metrics["track_5b_candidate_pool"]["requested"]),
            "selection": manifest.get("pair_selection"),
        },
        "agreement": agreement,
        "outcome_agreement": _outcome_agreement(source["pair_comparisons"], new_comparisons_path),
        "headline_metric_deltas": headline_metric_deltas,
        "artifacts": {
            "new_metrics": str(run_root / "metrics.json"),
            "new_pair_comparisons": str(new_comparisons_path),
            "pair_disagreements": str(disagreement_path),
        },
    }
    write_json_atomic(run_root / "comparison.json", comparison)
    report = _render_report(comparison)
    (run_root / "RESULTS.md").write_text(report, encoding="utf-8")
    write_json_atomic(
        run_root / "comparison_complete.json",
        {
            "status": "complete",
            "comparison_sha256": sha256_file(run_root / "comparison.json"),
            "results_sha256": sha256_file(run_root / "RESULTS.md"),
        },
    )
    return comparison


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _render_report(comparison: dict[str, Any]) -> str:
    agreement = comparison["agreement"]
    deltas = comparison["headline_metric_deltas"]
    new_label = comparison["new"].get("label", comparison["new"]["framework"])
    population = comparison.get("population", {"pairs": 20_000, "track_5a_pairs": 10_000, "track_5b_pairs": 10_000})
    lines = [
        f"# {new_label} Inference Hub vs. V0/DeepSeek",
        "",
        f"This run rejudges {population['pairs']:,} pairs from the frozen V0 population: "
        f"{population['track_5a_pairs']:,} removal-track pairs and "
        f"{population['track_5b_pairs']:,} cross-group-track pairs.",
        "",
        f"- Baseline: `{comparison['baseline']['framework']}` + `{comparison['baseline']['model']}`",
        f"- New: `{comparison['new']['framework']}` + `{comparison['new']['model']}` on NVIDIA Inference Hub",
        f"- Common schema-valid pairs: {agreement['common_valid']:,}",
        f"- Primary decision tuple agrees: {_pct(agreement['primary_decision_agreement'])}",
        f"- Descriptive taxonomy tuple agrees: {_pct(agreement['descriptive_taxonomy_agreement'])}",
        f"- All comparable fields agree: {_pct(agreement['all_core_fields_agreement'])}",
        "",
        "## Headline metrics",
        "",
        f"| Metric | V0/DeepSeek | {new_label} | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("Judge completion", "judge_completion_rate"),
        ("Removal precision", "removal_precision"),
        ("Wrong-removal rate", "wrong_removal_rate"),
        ("Cross-group positive yield", "cross_group_positive_yield"),
    ):
        value = deltas[key]
        lines.append(f"| {label} | {_pct(value['baseline'])} | {_pct(value['new'])} | {_pct(value['delta'])} |")
    lines.extend(["", "## Agreement by axis", "", "| Field | Agreement |", "|---|---:|"])
    for field, value in agreement["field_agreement"].items():
        lines.append(f"| `{field}` | {_pct(value)} |")
    lines.extend(
        [
            "",
            "The three primary decisions are the human-calibrated release axis: `same_duplicate_group` and both "
            "replacement directions. Relation/material labels are descriptive. Legacy `fuzzy_scope` is compared only "
            "when both contracts use that same field. DeepSeek agreement, SUT outcomes, proxy scores, and "
            "version-specific diagnostics do not decide the winner.",
            "",
            "## Interpretation boundary",
            "",
            comparison["scope_note"],
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN)
    prepare.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    prepare.add_argument("--judge-policy", choices=tuple(PROMPT_VERSION_BY_POLICY), default="sarah")
    prepare.add_argument("--runner-config", type=Path)
    prepare.add_argument("--ray-temp-dir", type=Path, default=DEFAULT_RAY_TEMP_DIR)
    prepare.add_argument("--block-size", type=int, default=500)
    prepare.add_argument("--max-parallel-requests", type=int, default=8)
    prepare.add_argument("--run-id")
    prepare.add_argument("--pair-ids", type=Path)
    prepare.add_argument("--release-approval", type=Path)
    for command in ("run", "summarize"):
        child = subparsers.add_parser(command)
        child.add_argument("--run-root", type=Path, required=True)
    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN)
    all_parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    all_parser.add_argument("--judge-policy", choices=tuple(PROMPT_VERSION_BY_POLICY), default="sarah")
    all_parser.add_argument("--runner-config", type=Path)
    all_parser.add_argument("--ray-temp-dir", type=Path, default=DEFAULT_RAY_TEMP_DIR)
    all_parser.add_argument("--block-size", type=int, default=500)
    all_parser.add_argument("--max-parallel-requests", type=int, default=8)
    all_parser.add_argument("--run-id")
    all_parser.add_argument("--pair-ids", type=Path)
    all_parser.add_argument("--release-approval", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"prepare", "all"}:
            run_root = prepare_run(
                source_run_root=args.source_run_root,
                runs_root=args.runs_root,
                runner_config=args.runner_config,
                prompt_version=PROMPT_VERSION_BY_POLICY[args.judge_policy],
                ray_temp_dir=args.ray_temp_dir,
                block_size=args.block_size,
                max_parallel_requests=args.max_parallel_requests,
                run_id=args.run_id,
                pair_ids_path=args.pair_ids,
                release_approval_path=args.release_approval,
            )
            print(json.dumps({"status": "prepared", "run_root": str(run_root)}, sort_keys=True), flush=True)
            if args.command == "prepare":
                return 0
        else:
            run_root = args.run_root
        if args.command in {"run", "all"}:
            print(json.dumps(run_hub(run_root), sort_keys=True), flush=True)
            if args.command == "run":
                return 0
        print(json.dumps(summarize(run_root), sort_keys=True), flush=True)
        return 0
    except DedupEvaluationError as exc:
        print(json.dumps({"status": "error", "issue": asdict(exc.issue)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
