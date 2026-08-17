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

"""Ordered, resumable orchestration for the ten-step operational proposal."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.analysis.comparison import build_pair_comparisons
from eval.dedup.analysis.constraint_graph import build_constraint_graph
from eval.dedup.analysis.metrics import compute_metrics
from eval.dedup.config import EvaluationConfig, ProfileConfig, load_config
from eval.dedup.contracts import ARTIFACT_SCHEMA_VERSION, EVALUATION_MANIFEST_SCHEMA_VERSION
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.handoff.manifests import register_corpus_handoff, register_sut_handoff
from eval.dedup.judging.run import run_judging
from eval.dedup.pair_construction.anchors import sample_anchors
from eval.dedup.pair_construction.canonicalize import canonicalize_selected_pairs
from eval.dedup.pair_construction.outcomes import build_document_outcomes
from eval.dedup.pair_construction.removal_pairs import sample_removal_pairs
from eval.dedup.pair_construction.retrieval.lexical import build_minhash_cache
from eval.dedup.pair_construction.retrieval.selection import retrieve_and_select_cross_group_pairs
from eval.dedup.report import export_human_qa, pair_explorer_destination, publish_report
from eval.dedup.validation import (
    read_json,
    require,
    sha256_file,
    write_json_atomic,
)

STAGE_NAMES = {
    1: "register_corpus_handoff",
    2: "register_sut_handoff",
    3: "recover_document_outcomes",
    4: "sample_anchors",
    5: "construct_candidate_pairs",
    6: "run_blind_judge",
    7: "build_partial_constraint_graph",
    8: "compare_sut_and_judgments",
    9: "calculate_metrics",
    10: "publish_report",
}


@dataclass(frozen=True, slots=True)
class RunContext:
    config: EvaluationConfig
    profile: ProfileConfig
    evaluation_run_id: str
    run_root: Path

    @property
    def manifests(self) -> Path:
        return self.run_root / "manifests"

    @property
    def data(self) -> Path:
        return self.run_root / "data"

    @property
    def reports(self) -> Path:
        return self.run_root / "reports"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"


@dataclass(frozen=True, slots=True)
class StageExecution:
    outputs: tuple[str, ...]
    counts: dict[str, Any]


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _evaluation_source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix in {".py", ".json", ".txt", ".md"}
    ):
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git_worktree_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", "eval/dedup", "tests/eval/dedup"],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _new_run_id(profile: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"dedup-{profile}-{timestamp}-{uuid.uuid4().hex[:10]}"


def _marker_path(context: RunContext, step: int) -> Path:
    return context.logs / "stages" / f"step_{step:02d}.json"


def _input_lineage(context: RunContext, step: int) -> list[dict[str, Any]]:
    lineage = []
    for prior_step in range(1, step):
        path = _marker_path(context, prior_step)
        require(
            path.is_file(),
            "STAGE_LINEAGE_MISSING",
            "a prior stage marker is missing",
            step=step,
            prior_step=prior_step,
        )
        lineage.append(
            {
                "step": prior_step,
                "marker_path": str(path.relative_to(context.run_root)),
                "marker_sha256": sha256_file(path),
            }
        )
    if step == 10:
        import_manifest_path = context.manifests / "human_qa_import.json"
        if import_manifest_path.is_file():
            import_manifest = read_json(import_manifest_path)
            results_path = context.data / "human_qa_results.csv"
            require(results_path.is_file(), "HUMAN_QA_IMPORT_MISSING", "QA import manifest has no results artifact")
            require(
                sha256_file(results_path) == import_manifest["results_sha256"],
                "HUMAN_QA_IMPORT_CHECKSUM_MISMATCH",
                "imported QA results changed",
            )
            lineage.extend(
                [
                    {
                        "kind": "human_qa_import_manifest",
                        "path": str(import_manifest_path.relative_to(context.run_root)),
                        "sha256": sha256_file(import_manifest_path),
                    },
                    {
                        "kind": "human_qa_results",
                        "path": str(results_path.relative_to(context.run_root)),
                        "sha256": sha256_file(results_path),
                    },
                ]
            )
    return lineage


def _validate_marker(context: RunContext, step: int) -> dict[str, Any] | None:
    marker_path = _marker_path(context, step)
    if not marker_path.exists():
        return None
    marker = read_json(marker_path)
    require(
        marker["schema_version"] == "dedup-stage-marker-v1",
        "INVALID_STAGE_MARKER",
        "stage marker schema differs",
        step=step,
    )
    require(
        marker["evaluation_run_id"] == context.evaluation_run_id,
        "INVALID_STAGE_MARKER",
        "stage marker run ID differs",
        step=step,
    )
    require(
        marker["step"] == step and marker["name"] == STAGE_NAMES[step],
        "INVALID_STAGE_MARKER",
        "stage marker identity differs",
        step=step,
    )
    require(marker["status"] == "complete", "INVALID_STAGE_MARKER", "stage marker is not complete", step=step)
    require(
        marker["config_digest"] == context.config.digest, "RESUME_CONFIG_MISMATCH", "run config changed", step=step
    )
    require(
        marker.get("input_lineage") == _input_lineage(context, step),
        "STAGE_LINEAGE_MISMATCH",
        "a completed stage no longer matches its input lineage",
        step=step,
    )
    for artifact in marker["artifacts"]:
        path = context.run_root / artifact["path"]
        require(path.is_file(), "RESUME_ARTIFACT_MISSING", "completed stage artifact is missing", path=str(path))
        require(
            path.stat().st_size == artifact["size_bytes"] and sha256_file(path) == artifact["sha256"],
            "RESUME_ARTIFACT_CHECKSUM_MISMATCH",
            "completed stage artifact changed",
            path=str(path),
        )
    return marker


def _commit_stage_transaction(context: RunContext, work_root: Path, transaction: dict[str, Any]) -> dict[str, Any]:
    require(
        transaction["evaluation_run_id"] == context.evaluation_run_id,
        "INVALID_STAGE_TRANSACTION",
        "transaction run ID differs",
    )
    step = int(transaction["step"])
    require(
        transaction["config_digest"] == context.config.digest,
        "RESUME_CONFIG_MISMATCH",
        "transaction config differs",
        step=step,
    )
    require(
        transaction["input_lineage"] == _input_lineage(context, step),
        "STAGE_LINEAGE_MISMATCH",
        "transaction input lineage differs",
        step=step,
    )
    for artifact in transaction["artifacts"]:
        source = work_root / artifact["path"]
        destination = context.run_root / artifact["path"]
        if destination.exists():
            require(
                destination.is_file()
                and destination.stat().st_size == artifact["size_bytes"]
                and sha256_file(destination) == artifact["sha256"],
                "IMMUTABLE_ARTIFACT_COLLISION",
                "partially published stage artifact differs from its transaction",
                path=str(destination),
            )
            continue
        require(
            source.is_file(), "STAGE_OUTPUT_MISSING", "transaction output is missing", step=step, path=artifact["path"]
        )
        require(
            source.stat().st_size == artifact["size_bytes"] and sha256_file(source) == artifact["sha256"],
            "STAGE_OUTPUT_CHECKSUM_MISMATCH",
            "transaction output changed before publication",
            path=str(source),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    marker = {
        "schema_version": "dedup-stage-marker-v1",
        "evaluation_run_id": context.evaluation_run_id,
        "step": step,
        "name": STAGE_NAMES[step],
        "status": "complete",
        "config_digest": context.config.digest,
        "input_lineage": transaction["input_lineage"],
        "counts": transaction["counts"],
        "artifacts": transaction["artifacts"],
        "completed_at_utc": transaction["completed_at_utc"],
    }
    write_json_atomic(_marker_path(context, step), marker)
    shutil.rmtree(work_root)
    return marker


def _publish_stage(context: RunContext, step: int, work_root: Path, execution: StageExecution) -> dict[str, Any]:
    artifacts = []
    for relative in execution.outputs:
        source = work_root / relative
        require(
            source.is_file(),
            "STAGE_OUTPUT_MISSING",
            "stage did not produce a declared output",
            step=step,
            path=relative,
        )
        destination = context.run_root / relative
        require(
            not destination.exists(),
            "IMMUTABLE_ARTIFACT_COLLISION",
            "stage output already exists without a recoverable transaction",
            path=str(destination),
        )
        artifacts.append({"path": relative, "sha256": sha256_file(source), "size_bytes": source.stat().st_size})
    transaction = {
        "schema_version": "dedup-stage-transaction-v1",
        "evaluation_run_id": context.evaluation_run_id,
        "step": step,
        "config_digest": context.config.digest,
        "input_lineage": _input_lineage(context, step),
        "counts": execution.counts,
        "artifacts": artifacts,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(work_root / "stage_transaction.json", transaction)
    return _commit_stage_transaction(context, work_root, transaction)


def _execute_stage(context: RunContext, step: int, function: Callable[[Path], StageExecution]) -> dict[str, Any]:
    existing = _validate_marker(context, step)
    if existing is not None:
        return existing
    work_parent = context.run_root / ".work"
    recoverable = sorted(
        path for path in work_parent.glob(f"step_{step:02d}-*") if (path / "stage_transaction.json").is_file()
    )
    require(
        len(recoverable) <= 1,
        "MULTIPLE_STAGE_TRANSACTIONS",
        "more than one recoverable stage transaction exists",
        step=step,
    )
    if recoverable:
        return _commit_stage_transaction(context, recoverable[0], read_json(recoverable[0] / "stage_transaction.json"))
    work_root = work_parent / f"step_{step:02d}-{uuid.uuid4().hex}"
    work_root.mkdir(parents=True)
    execution = function(work_root)
    return _publish_stage(context, step, work_root, execution)


def _evaluation_manifest(
    context: RunContext,
    corpus: dict[str, Any],
    sut: dict[str, Any],
    tokenizer: TokenCounter,
    *,
    sut_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "evaluation_manifest_schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "evaluation_run_id": context.evaluation_run_id,
        "profile": context.profile.name,
        "formal_v0": context.profile.formal_v0,
        "corpus_manifest_uri": str(context.manifests / "corpus_manifest.json"),
        "corpus_manifest_sha256": sha256_file(context.manifests / "corpus_manifest.json"),
        "sut_run_id": sut["sut_run_id"],
        "sut_run_manifest_uri": str(context.manifests / "sut_run_manifest.json"),
        "sut_run_manifest_sha256": sha256_file(sut_manifest_path),
        "dataset_version": corpus["dataset_version"],
        "dataset_row_count": corpus["dataset_row_count"],
        "normalization_version": corpus["normalization_version"],
        "exact_dedup_applied_upstream": True,
        "exact_dedup_in_evaluation_scope": False,
        "upstream_reproducibility_status": sut["upstream_reproducibility_status"],
        "upstream_provenance_availability": sut["provenance_availability"],
        "embedding_artifact_uri": corpus["embedding"]["path"],
        "embedding_artifact_sha256": corpus["embedding"]["sha256"],
        "embedding_model": corpus["embedding"]["model"],
        "embedding_normalized": corpus["embedding"]["l2_normalized"],
        "tokenizer": tokenizer.contract(),
        "retrieval_config_and_pilot_cutoffs": {
            "uri": str(context.manifests / "retrieval_config.json"),
            "selection_rule": "pilot median in [20,50], closest to 35",
            "status": "frozen_by_step_5_before_candidate_selection",
        },
        **context.config.seeds,
        "canonical_pair_id_version": context.config.canonical_pair_id_version,
        "judge_model": context.config.judge.model,
        "judge_model_revision": context.config.judge.model,
        "prompt_version": context.config.judge.prompt_version,
        "judge_schema_version": context.config.judge.schema_version,
        "evaluation_code_revision": _git_revision(),
        "evaluation_code_worktree_dirty": _git_worktree_dirty(),
        "evaluation_source_tree_sha256": _evaluation_source_digest(),
        "python_version": sys.version,
        "output_root": str(context.run_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "immutable_after": "Step 5 candidate generation begins",
    }


def create_run(config: EvaluationConfig, profile_name: str, *, evaluation_run_id: str | None = None) -> RunContext:
    profile = config.profile(profile_name)
    require(
        config.retrieval.backend == "fixture_cpu"
        or not profile.formal_v0
        or config.judge.model == "nvidia/deepseek-ai/deepseek-v4-pro",
        "FORMAL_V0_JUDGE_MODEL_MISMATCH",
        "formal V0 requires the proposal judge model; V4 Flash is allowed only for NON-V0 smoke",
        judge_model=config.judge.model,
    )
    run_id = evaluation_run_id or _new_run_id(profile_name)
    run_root = config.output_root / run_id / "v0_run"
    require(not run_root.exists(), "RUN_ROOT_EXISTS", "new run root already exists", path=str(run_root))
    for path in (run_root / "manifests", run_root / "data", run_root / "reports", run_root / "logs" / "stages"):
        path.mkdir(parents=True, exist_ok=True)
    context = RunContext(config=config, profile=profile, evaluation_run_id=run_id, run_root=run_root)
    frozen_config_path = context.manifests / "config.frozen.json"
    write_json_atomic(frozen_config_path, config.raw)
    write_json_atomic(
        context.manifests / "config.snapshot.json",
        {
            "config_digest": config.digest,
            "frozen_config_sha256": sha256_file(frozen_config_path),
            "source_path": str(config.source_path),
            "profile": profile_name,
            "evaluation_run_id": run_id,
            "config": config.raw,
            "evaluation_source_tree_sha256": _evaluation_source_digest(),
        },
    )
    return context


def load_run(run_root: Path) -> RunContext:
    snapshot = read_json(run_root / "manifests" / "config.snapshot.json")
    frozen_config = run_root / "manifests" / "config.frozen.json"
    require(
        sha256_file(frozen_config) == snapshot["frozen_config_sha256"],
        "RESUME_CONFIG_MISMATCH",
        "frozen run config changed after run creation",
    )
    config = load_config(frozen_config, path_base=Path(snapshot["source_path"]).parent)
    require(
        config.digest == snapshot["config_digest"],
        "RESUME_CONFIG_MISMATCH",
        "frozen run config digest changed",
    )
    return RunContext(
        config=config,
        profile=config.profile(snapshot["profile"]),
        evaluation_run_id=snapshot["evaluation_run_id"],
        run_root=run_root.resolve(),
    )


def _stage_1(context: RunContext, work: Path) -> StageExecution:
    relative = "manifests/corpus_manifest.json"
    wrapper = register_corpus_handoff(context.config, work / relative)
    return StageExecution((relative,), {"rows": wrapper["dataset_row_count"], "shards": len(wrapper["shards"])})


def _stage_2(context: RunContext, work: Path) -> StageExecution:
    corpus = read_json(context.manifests / "corpus_manifest.json")
    work_manifests = work / "manifests"
    work_manifests.mkdir(parents=True, exist_ok=True)
    corpus_copy = work_manifests / "corpus_manifest.json"
    shutil.copyfile(context.manifests / "corpus_manifest.json", corpus_copy)
    sut_relative = "manifests/sut_run_manifest.json"
    sut_path = work / sut_relative
    sut = register_sut_handoff(context.config, corpus, sut_path)
    tokenizer = TokenCounter(context.config.tokenizer)
    evaluation_relative = "manifests/evaluation_manifest.json"
    write_json_atomic(
        work / evaluation_relative,
        _evaluation_manifest(context, corpus, sut, tokenizer, sut_manifest_path=sut_path),
    )
    corpus_copy.unlink()
    return StageExecution(
        (sut_relative, evaluation_relative),
        {
            "grouped_documents": sut["duplicate_groups"]["rows"],
            "groups": sut["duplicate_groups"]["groups"],
            "removals": sut["removal_ids"]["rows"],
        },
    )


def _tokenizer_for_run(context: RunContext) -> TokenCounter:
    manifest = read_json(context.manifests / "evaluation_manifest.json")
    frozen = replace(context.config.tokenizer, revision=manifest["tokenizer"]["resolved_revision"])
    tokenizer = TokenCounter(frozen)
    tokenizer.config = context.config.tokenizer
    require(tokenizer.contract() == manifest["tokenizer"], "TOKENIZER_CONTRACT_MISMATCH", "resolved tokenizer changed")
    return tokenizer


def _stage_3(context: RunContext, work: Path) -> StageExecution:
    relative = "data/document_outcomes.parquet"
    counts = build_document_outcomes(
        context.config,
        evaluation_manifest=read_json(context.manifests / "evaluation_manifest.json"),
        corpus_manifest=read_json(context.manifests / "corpus_manifest.json"),
        sut_manifest=read_json(context.manifests / "sut_run_manifest.json"),
        destination=work / relative,
        tokenizer=_tokenizer_for_run(context),
    )
    return StageExecution((relative,), counts)


def _stage_4(context: RunContext, work: Path) -> StageExecution:
    relative = "data/anchors.parquet"
    counts = sample_anchors(
        context.data / "document_outcomes.parquet",
        profile=context.profile,
        anchor_seed=context.config.seeds["anchor_seed"],
        destination=work / relative,
    )
    return StageExecution((relative,), counts)


def _stage_5(context: RunContext, work: Path) -> StageExecution:
    corpus = read_json(context.manifests / "corpus_manifest.json")
    signature_path, signature_manifest = build_minhash_cache(
        context.config,
        corpus_manifest=corpus,
        cache_dir=context.config.cache_root,
    )
    removal_relative = "data/removal_pairs_selected.parquet"
    removal_counts = sample_removal_pairs(
        context.data / "document_outcomes.parquet",
        profile=context.profile,
        pair_seed=context.config.seeds["pair_seed"],
        destination=work / removal_relative,
    )
    cross_relative = "data/cross_group_pairs_selected.parquet"
    retrieval_relative = "manifests/retrieval_config.json"
    cross_counts = retrieve_and_select_cross_group_pairs(
        context.config,
        profile=context.profile,
        corpus_manifest=corpus,
        outcomes_path=context.data / "document_outcomes.parquet",
        anchors_path=context.data / "anchors.parquet",
        signature_path=signature_path,
        signature_manifest=signature_manifest,
        destination=work / cross_relative,
        retrieval_config_destination=work / retrieval_relative,
    )
    candidate_relative = "data/candidate_pairs.parquet"
    provenance_relative = "data/pair_provenance.parquet"
    canonical_counts = canonicalize_selected_pairs(
        context.config,
        corpus_manifest=corpus,
        outcomes_path=context.data / "document_outcomes.parquet",
        removal_pairs_path=work / removal_relative,
        cross_group_pairs_path=work / cross_relative,
        tokenizer=_tokenizer_for_run(context),
        candidate_destination=work / candidate_relative,
        provenance_destination=work / provenance_relative,
    )
    return StageExecution(
        (removal_relative, cross_relative, candidate_relative, provenance_relative, retrieval_relative),
        {
            "minhash_cache_contract": signature_manifest["contract_digest"],
            **{f"removal_{key}": value for key, value in removal_counts.items()},
            **{f"cross_{key}": value for key, value in cross_counts.items()},
            **canonical_counts,
        },
    )


def _stage_6(context: RunContext, work: Path) -> StageExecution:
    payload_relative = "data/judge_payloads.jsonl"
    results_relative = "data/judge_results.jsonl"
    errors_relative = "logs/judge_errors.jsonl"
    judge_config_relative = "manifests/judge_config.json"
    counts = run_judging(
        context.config,
        corpus_manifest=read_json(context.manifests / "corpus_manifest.json"),
        candidate_pairs_path=context.data / "candidate_pairs.parquet",
        tokenizer=_tokenizer_for_run(context),
        cache_path=context.logs / "judge_cache.jsonl",
        payloads_destination=work / payload_relative,
        results_destination=work / results_relative,
        errors_destination=work / errors_relative,
        judge_config_destination=work / judge_config_relative,
    )
    qa_packet_relative = "data/human_qa_packet.jsonl"
    qa_labels_relative = "data/human_qa_labels.csv"
    qa_counts = export_human_qa(
        profile=context.profile,
        qa_seed=context.config.seeds["qa_seed"],
        payloads_path=work / payload_relative,
        provenance_path=context.data / "pair_provenance.parquet",
        packet_destination=work / qa_packet_relative,
        labels_destination=work / qa_labels_relative,
    )
    return StageExecution(
        (
            payload_relative,
            results_relative,
            errors_relative,
            judge_config_relative,
            qa_packet_relative,
            qa_labels_relative,
        ),
        {**counts, **qa_counts},
    )


def _stage_7(context: RunContext, work: Path) -> StageExecution:
    paths = {
        "must": "data/judged_must_links.parquet",
        "cannot": "data/judged_cannot_links.parquet",
        "components": "data/partial_reference_components.parquet",
        "conflicts": "data/constraint_conflicts.parquet",
    }
    counts = build_constraint_graph(
        context.data / "judge_results.jsonl",
        context.data / "candidate_pairs.parquet",
        must_links_destination=work / paths["must"],
        cannot_links_destination=work / paths["cannot"],
        components_destination=work / paths["components"],
        conflicts_destination=work / paths["conflicts"],
    )
    return StageExecution(tuple(paths.values()), counts)


def _stage_8(context: RunContext, work: Path) -> StageExecution:
    relative = "data/pair_comparisons.parquet"
    counts = build_pair_comparisons(
        candidate_pairs_path=context.data / "candidate_pairs.parquet",
        pair_provenance_path=context.data / "pair_provenance.parquet",
        outcomes_path=context.data / "document_outcomes.parquet",
        judge_results_path=context.data / "judge_results.jsonl",
        judge_errors_path=context.logs / "judge_errors.jsonl",
        destination=work / relative,
    )
    return StageExecution((relative,), counts)


def _stage_9(context: RunContext, work: Path) -> StageExecution:
    metrics_relative = "reports/metrics.json"
    slices_relative = "reports/metrics_by_slice.csv"
    accounting_relative = "reports/pipeline_accounting.csv"
    markers = [read_json(_marker_path(context, step)) for step in range(1, 9)]
    metrics = compute_metrics(
        context.data / "pair_comparisons.parquet",
        requested_judge_pairs=read_json(_marker_path(context, 6))["counts"]["requested"],
        metrics_destination=work / metrics_relative,
        slices_destination=work / slices_relative,
        accounting_destination=work / accounting_relative,
        stage_markers=markers,
    )
    return StageExecution(
        (metrics_relative, slices_relative, accounting_relative),
        {
            "judge_completion_rate": metrics["judge"]["completion_rate"],
            "removal_precision": metrics["track_5a_removal_frame"]["removal_precision"],
            "cross_group_positive_yield": metrics["track_5b_candidate_pool"]["positive_yield"],
        },
    )


def _stage_10(context: RunContext, work: Path) -> StageExecution:
    final_relative = "reports/final_report.md"
    recommendations_relative = "reports/recommendations.json"
    manifest_relative = "reports/report_generation_manifest.json"
    dashboard_relative = str(pair_explorer_destination(Path(final_relative)))
    result = publish_report(
        profile=context.profile,
        run_root=context.run_root,
        recommendation_judge=context.config.judge,
        final_destination=work / final_relative,
        recommendations_destination=work / recommendations_relative,
        manifest_destination=work / manifest_relative,
    )
    return StageExecution(
        (final_relative, recommendations_relative, manifest_relative, dashboard_relative),
        {**result, "pair_explorer": dashboard_relative},
    )


def _require_execution_source_contract(context: RunContext) -> None:
    snapshot = read_json(context.manifests / "config.snapshot.json")
    expected = snapshot.get("evaluation_source_tree_sha256")
    require(
        isinstance(expected, str) and expected == _evaluation_source_digest(),
        "RESUME_SOURCE_MISMATCH",
        "evaluation source tree differs from the run-creation snapshot",
        expected=expected,
    )


def run_pipeline(context: RunContext, *, through_step: int = 10) -> dict[str, Any]:
    require(1 <= through_step <= 10, "INVALID_STEP", "through_step must be between 1 and 10")
    markers = []
    _require_execution_source_contract(context)
    stage_functions: dict[int, Callable[[RunContext, Path], StageExecution]] = {
        1: _stage_1,
        2: _stage_2,
        3: _stage_3,
        4: _stage_4,
        5: _stage_5,
        6: _stage_6,
        7: _stage_7,
        8: _stage_8,
        9: _stage_9,
        10: _stage_10,
    }
    for step in range(1, through_step + 1):
        markers.append(_execute_stage(context, step, lambda work, fn=stage_functions[step]: fn(context, work)))
    return {"status": "complete", "run_root": str(context.run_root), "stages": markers}


def run_status(context: RunContext) -> dict[str, Any]:
    stages = []
    for step in range(1, 11):
        marker = _validate_marker(context, step)
        stages.append({"step": step, "name": STAGE_NAMES[step], "status": "complete" if marker else "pending"})
    return {
        "evaluation_run_id": context.evaluation_run_id,
        "profile": context.profile.name,
        "run_root": str(context.run_root),
        "stages": stages,
        "human_qa_results_available": (context.data / "human_qa_results.csv").is_file(),
    }


def validate_run(context: RunContext) -> dict[str, Any]:
    status = run_status(context)
    completed = sum(item["status"] == "complete" for item in status["stages"])
    if context.profile.formal_v0 and completed == 10:
        valid_rate = read_json(context.reports / "metrics.json")["judge"]["completion_rate"]
        require(
            valid_rate is not None and valid_rate >= 0.99,
            "JUDGE_COMPLETION_BELOW_ACCEPTANCE",
            "formal V0 valid rate is below 99%",
        )
    return {**status, "completed_steps": completed, "valid": completed == 10}
