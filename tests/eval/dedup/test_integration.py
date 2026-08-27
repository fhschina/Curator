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

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import eval.dedup.judging.local_ndd as local_ndd_module
import eval.dedup.run as run_module
from eval.dedup.cli import preflight
from eval.dedup.config import load_config
from eval.dedup.run import create_run, load_run, run_pipeline, validate_run
from eval.dedup.validation import DedupEvaluationError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _build_fixture_handoff(root: Path) -> tuple[Path, Path, Path]:
    handoff = root / "handoff"
    source_key = "fixture-source"
    text_path = handoff / "extracted_text" / "objects" / source_key / "text.parquet"
    text_path.parent.mkdir(parents=True)
    groups = {
        100: list(range(2)),
        101: list(range(2, 5)),
        102: list(range(5, 11)),
        103: list(range(11, 32)),
    }
    texts = []
    for doc_id in range(36):
        if doc_id < 32:
            texts.append("shared duplicate body with stable common language")
        else:
            texts.append(f"singleton document {doc_id} with unrelated material")
    table = pa.table(
        {
            "source_id": [source_key] * 36,
            "warc_path": ["crawl-data/fixture.warc.gz"] * 36,
            "warc_record_id": [f"warc-{doc_id}" for doc_id in range(36)],
            "warc_record_index": list(range(36)),
            "url": [f"HTTPS://Example.COM:443/doc/{doc_id}#fragment" for doc_id in range(36)],
            "timestamp": ["2026-08-12T00:00:00Z"] * 36,
            "language": ["en"] * 36,
            "text": texts,
        }
    )
    pq.write_table(table, text_path)
    dense = {
        "kind": "ExplicitDenseShardManifest",
        "schema_version": 1,
        "scale_semantics": {"corpus_level_claims_authorized": False},
        "shards": [
            {
                "source_key": source_key,
                "path": "/upstream/old/location/text.parquet",
                "start_id": 0,
                "end_id": 35,
                "rows": 36,
                "sha256": _sha256(text_path),
            }
        ],
    }
    manifest_path = handoff / "manifest" / "dense_10m_explicit.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(dense))
    singleton_manifest = handoff / "singletons" / "dataset_manifest.json"
    singleton_manifest.parent.mkdir(parents=True)
    singleton_manifest.write_text(json.dumps({"part_count": 1, "expected_singleton_documents": 4}))
    group_rows = [
        {"_curator_dedup_id": doc_id, "_duplicate_group_id": group_id}
        for group_id, members in groups.items()
        for doc_id in members
    ]
    removal_ids = [doc_id for members in groups.values() for doc_id in members[1:]]
    fuzzy = handoff / "fuzzy"
    fuzzy.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(group_rows), fuzzy / "duplicate_groups.parquet")
    pq.write_table(pa.table({"_curator_dedup_id": removal_ids}), fuzzy / "removal_ids.parquet")
    (fuzzy / "run_identity.json").write_text(
        json.dumps(
            {
                "run_id": "fixture-sut",
                "repository_revision": "fixture-revision",
                "config_digest": "fixture-config",
                "runtime_manifest_digest": "fixture-runtime",
                "manifest_digest": "fixture-manifest",
            }
        )
    )
    (fuzzy / "runtime_manifest.json").write_text(json.dumps({"schema_version": 1}))
    embedding_dir = handoff / "embeddings"
    embedding_dir.mkdir(parents=True)
    embeddings = np.zeros((36, 4), dtype=np.float32)
    embeddings[:, 0] = 1.0
    embeddings[32:, 0] = 0.7
    embeddings[32:, 1] = np.sqrt(1 - 0.7**2)
    embedding_path = embedding_dir / "embeddinggemma_300m.normalized.f32"
    embeddings.tofile(embedding_path)
    return handoff, embedding_path, text_path


def _write_fixture_config(root: Path, handoff: Path, embedding_path: Path, *, judge_backend: str = "stub") -> Path:
    value = {
        "schema_version": 1,
        "handoff_root": str(handoff),
        "output_root": str(root / "runs"),
        "cache_root": str(root / "cache"),
        "verify_checksums": True,
        "dataset": {
            "dataset_version": "fixture-36",
            "expected_rows": 36,
            "expected_grouped_documents": 32,
            "expected_groups": 4,
            "expected_removals": 28,
            "expected_singletons": 4,
            "expected_retained": 8,
            "embedding_rows": 36,
            "embedding_dimensions": 4,
            "embedding_dtype": "float32",
            "embedding_sha256": _sha256(embedding_path),
        },
        "tokenizer": {
            "kind": "whitespace",
            "model_id": "fixture-whitespace",
            "revision": "fixture-v1",
            "cache_root": str(root / "cache" / "tokenizer"),
        },
        "judge": {
            "backend": "stub",
            "base_url": "http://fixture.invalid/v1",
            "model": "fixture-judge",
            "api_key_env": "UNUSED_FIXTURE_KEY",
            "structured_output_mode": "json_schema",
            "thinking": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 1024,
            "concurrency": 2,
            "requests_per_minute": 60000,
            "timeout_seconds": 1.0,
            "max_retries": 2,
            "max_visible_tokens": 128,
            "window_tokens": 32,
            "window_overlap_tokens": 4,
            "prompt_version": "dedup-judge-v0",
            "schema_version": "dedup-judge-output-v0",
        },
        "retrieval": {
            "backend": "fixture_cpu",
            "minhash_seed": 42,
            "char_ngram_width": 3,
            "num_hashes": 8,
            "feature_ngram_width": 3,
            "lsh_grid": [[8, 1]],
            "pilot_target_min": 1,
            "pilot_target_max": 50,
            "pilot_target_center": 20,
            "top_k": 10,
            "signature_chunk_rows": 16,
            "semantic_chunk_rows": 16,
            "max_candidates_per_anchor": 100,
        },
        "seeds": {
            "pilot_seed": 26081200,
            "anchor_seed": 26081201,
            "pair_seed": 26081202,
            "judge_order_seed": 26081203,
            "qa_seed": 26081204,
        },
        "canonical_pair_id_version": "cp1",
        "profiles": {
            "smoke": {
                "anchor_quotas": {
                    "singleton": 2,
                    "size_2": 1,
                    "size_3_5": 1,
                    "size_6_20": 1,
                    "size_21_plus": 1,
                },
                "removal_pair_budget": 3,
                "cross_group_pair_budget": 5,
                "qa_pair_budget": 6,
                "minimum_diff_budget": 0,
                "formal_v0": False,
            },
            "full": {
                "anchor_quotas": {
                    "singleton": 2,
                    "size_2": 1,
                    "size_3_5": 1,
                    "size_6_20": 1,
                    "size_21_plus": 1,
                },
                "removal_pair_budget": 3,
                "cross_group_pair_budget": 5,
                "qa_pair_budget": 6,
                "minimum_diff_budget": 0,
                "formal_v0": True,
            },
        },
    }
    if judge_backend == "local_ndd":
        resources_root = root / "local_ndd_resources"
        resources_root.mkdir()
        runner_config = resources_root / "fixture.yaml"
        runner_config.write_text("models: []\nexecution: {stages: []}\n")
        model_path = root / "qwen-fixture"
        model_path.mkdir()
        value["judge"] = {
            "backend": "local_ndd",
            "model": "Qwen/Qwen3.8-27B",
            "model_path": str(model_path),
            "runner_config": str(runner_config),
            "ray_temp_dir": str(root / "ray"),
            "checkpoint_root": str(root / "checkpoints"),
            "num_cpus": 2,
            "num_gpus": 1,
            "max_retries": 2,
            "max_visible_tokens": 128,
            "window_tokens": 32,
            "window_overlap_tokens": 4,
            "prompt_version": "dedup-judge-sarah-minhash-v1",
            "schema_version": "dedup-judge-output-v0",
            "visible_payload_version": "judge-visible-payload-v2",
        }
        value["profiles"]["full"]["formal_v0"] = False
    elif judge_backend != "stub":
        raise ValueError(judge_backend)

    path = root / "fixture_config.json"
    path.write_text(json.dumps(value))
    return path


def test_fixture_pipeline_runs_all_ten_steps(tmp_path: Path, monkeypatch) -> None:  # noqa: PLR0915
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    handoff, embedding_path, _ = _build_fixture_handoff(tmp_path)
    config_path = _write_fixture_config(tmp_path, handoff, embedding_path)
    assert preflight(load_config(config_path), "smoke")["status"] == "passed"
    context = create_run(load_config(config_path), "smoke", evaluation_run_id="fixture-smoke")
    partial = run_pipeline(context, through_step=4)
    assert len(partial["stages"]) == 4

    original_commit = run_module._commit_stage_transaction

    def interrupt_after_first_publish(run_context, work_root, transaction):
        artifact = transaction["artifacts"][0]
        destination = run_context.run_root / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work_root / artifact["path"], destination)
        raise RuntimeError("simulated interruption after partial publication")

    monkeypatch.setattr(run_module, "_commit_stage_transaction", interrupt_after_first_publish)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_pipeline(context, through_step=5)
    monkeypatch.setattr(run_module, "_commit_stage_transaction", original_commit)
    partial = run_pipeline(context, through_step=5)
    assert len(partial["stages"]) == 5
    removal_rows = pq.read_table(context.data / "removal_pairs_selected.parquet").to_pylist()
    assert {row["frame_size"] for row in removal_rows} == {28}
    assert {row["selection_probability"] for row in removal_rows} == {3 / 28}
    provenance_table = pq.read_table(context.data / "pair_provenance.parquet")
    provenance = provenance_table.to_pylist()
    candidates = pq.read_table(context.data / "candidate_pairs.parquet").to_pylist()
    candidate_ids = [row["canonical_pair_id"] for row in candidates]
    assert {row["track"] for row in provenance} == {"5a", "5b"}
    assert len(candidate_ids) == len(set(candidate_ids))
    assert {row["canonical_pair_id"] for row in provenance} <= set(candidate_ids)
    assert {
        "retriever_bitmask",
        "lexical_rank",
        "semantic_rank",
        "jaccard",
        "cosine",
    } <= set(provenance_table.column_names)
    cross_provenance = [row for row in provenance if row["track"] == "5b"]
    assert cross_provenance
    assert all(row["retriever_bitmask"] in {"lexical_only", "semantic_only", "both"} for row in cross_provenance)
    marker = json.loads((context.logs / "stages" / "step_05.json").read_text())
    assert len(marker["input_lineage"]) == 4
    config_path.unlink()
    context = load_run(context.run_root)
    result = run_pipeline(context)
    assert result["status"] == "complete"
    validated = validate_run(context)
    assert validated["valid"]
    assert validated["completed_steps"] == 10
    comparisons = pq.read_table(context.data / "pair_comparisons.parquet").to_pylist()
    cross_comparisons = [row for row in comparisons if row["has_track_5b"]]
    assert cross_comparisons
    assert all(
        row["retriever_category"] in {"lexical_only", "semantic_only", "both_or_overlap"} for row in cross_comparisons
    )
    diagnostic_packet = context.data / "human_qa_diagnostic_packet.jsonl"
    diagnostic_labels = context.data / "human_qa_diagnostic_labels.csv"
    assert diagnostic_packet.is_file()
    assert diagnostic_labels.is_file()
    qa_dashboard = context.reports / "human_qa_dashboard.html"
    diagnostic_dashboard = context.reports / "human_qa_diagnostic_dashboard.html"
    assert qa_dashboard.is_file()
    assert not diagnostic_dashboard.exists()
    dashboard = qa_dashboard.read_text()
    assert "Human QA Review" in dashboard
    assert "Blind sample" in dashboard
    assert "Diagnostic set" in dashboard
    diagnostic_rows = [json.loads(line) for line in diagnostic_packet.read_text().splitlines()]
    assert len(diagnostic_rows) <= context.profile.qa_pair_budget
    assert all(set(row) == {"qa_pair_id", "judge_payload_hash", "visible_payload"} for row in diagnostic_rows)
    report = (context.reports / "final_report.md").read_text()
    assert "NON-V0 SMOKE" in report
    assert "not corpus recall" in report
    assert "### By predicted group size" in report
    assert "## 7. How to Inspect the Results" in report
    assert "Representative SUT-Judge removal examples" not in report
    assert "AI-generated Interpretation" not in report
    assert "## Appendix A — Pipeline and Judge Operations" in report
    pair_explorer = context.reports / "pair_explorer.html"
    assert pair_explorer.is_file()
    assert "Dedup Pair Explorer" in pair_explorer.read_text()


def _fixture_ndd_rubric() -> dict[str, dict[str, object]]:
    scores: dict[str, object] = {
        "same_duplicate_group": "yes",
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "exact",
        "material_difference": "none",
        "fuzzy_scope": "in_scope",
        "confidence": 0.98,
    }
    for field in (
        "reason_number_change",
        "reason_date_time_change",
        "reason_product_version_change",
        "reason_url_change",
        "reason_named_entity_change",
        "reason_negation_change",
        "reason_code_literal_change",
        "reason_code_output_change",
        "reason_insertion_deletion",
        "reason_boilerplate",
        "reason_parser_noise",
        "reason_language_mismatch",
        "reason_topic_only",
        "reason_insufficient_evidence",
        "reason_other_material",
    ):
        scores[field] = "no"
    return {name: {"score": score, "reasoning": f"private-{name}"} for name, score in scores.items()}


def test_fixture_local_ndd_runs_ten_steps_and_resumes_from_judge_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    handoff, embedding_path, _ = _build_fixture_handoff(tmp_path)
    config_path = _write_fixture_config(tmp_path, handoff, embedding_path, judge_backend="local_ndd")
    context = create_run(load_config(config_path), "smoke", evaluation_run_id="fixture-local-ndd")

    class FakeRuntime:
        starts = 0
        stops = 0
        batches: ClassVar[list[list[str]]] = []

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            type(self).starts += 1
            return self

        def __exit__(self, *args: object) -> None:
            del args
            type(self).stops += 1

        def run(self, **kwargs: object) -> None:
            input_rows = [json.loads(line) for line in Path(str(kwargs["input_path"])).read_text().splitlines()]
            type(self).batches.append([row["canonical_pair_id"] for row in input_rows])
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True, exist_ok=True)
            output_rows = [{**row, local_ndd_module.JUDGE_COLUMN: _fixture_ndd_rubric()} for row in input_rows]
            (output_path / "part.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in output_rows),
                encoding="utf-8",
            )

    monkeypatch.setattr(local_ndd_module, "_runtime_factory", lambda: FakeRuntime)
    run_pipeline(context, through_step=5)
    original_publish = run_module._publish_stage

    def interrupt_after_judge(run_context, step, work_root, execution):
        if step == 6:
            raise RuntimeError("simulated interruption after judge cache fsync")
        return original_publish(run_context, step, work_root, execution)

    monkeypatch.setattr(run_module, "_publish_stage", interrupt_after_judge)
    with pytest.raises(RuntimeError, match="after judge cache fsync"):
        run_pipeline(context, through_step=6)
    assert FakeRuntime.starts == FakeRuntime.stops == 1
    first_batch = FakeRuntime.batches[0]
    assert first_batch

    monkeypatch.setattr(run_module, "_publish_stage", original_publish)
    result = run_pipeline(context)

    assert result["status"] == "complete"
    assert FakeRuntime.starts == FakeRuntime.stops == 1
    assert validate_run(context)["valid"]
    payload_rows = [json.loads(line) for line in (context.data / "judge_payloads.jsonl").read_text().splitlines()]
    assert payload_rows
    assert all(row["payload"]["payload_schema_version"] == "judge-visible-payload-v2" for row in payload_rows)
    assert all(set(row["payload"]["document_a"]) == {"text"} for row in payload_rows)
    assert all(set(row["payload"]["document_b"]) == {"text"} for row in payload_rows)
    assert all("metadata" not in json.dumps(row["payload"]) for row in payload_rows)
    judge_rows = [json.loads(line) for line in (context.data / "judge_results.jsonl").read_text().splitlines()]
    assert len(judge_rows) == len(first_batch)
    assert all(row["schema_version"] == "dedup-judge-output-v0" for row in judge_rows)
    assert all("private-" not in json.dumps(row) for row in judge_rows)
    assert (context.data / "pair_comparisons.parquet").is_file()
    assert (context.reports / "final_report.md").is_file()
    assert (context.reports / "pair_explorer.html").is_file()


def test_run_execution_rejects_changed_source_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    handoff, embedding_path, _ = _build_fixture_handoff(tmp_path)
    config_path = _write_fixture_config(tmp_path, handoff, embedding_path)
    context = create_run(
        load_config(config_path),
        "smoke",
        evaluation_run_id="fixture-source-contract",
    )
    monkeypatch.setattr(run_module, "_evaluation_source_digest", lambda: "changed-source")

    with pytest.raises(DedupEvaluationError) as error:
        run_pipeline(context, through_step=1)

    assert error.value.issue.code == "RESUME_SOURCE_MISMATCH"
