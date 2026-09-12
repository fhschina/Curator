# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import csv
import io
import json

import pytest

from eval.dedup.analysis import exp5_dashboard as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic, write_text_atomic


def test_baseline_resolves_published_sarah_qwen_instead_of_source_v0(tmp_path):
    root = tmp_path / "sarah-qwen"
    write_text_atomic(root / "data/judge_results.jsonl", "{}\n")
    write_text_atomic(root / "logs/judge_errors.jsonl", "")
    write_json_atomic(
        root / "run_complete.json",
        {
            "requested": 1,
            "valid": 1,
            "errors": 0,
            "results_sha256": sha256_file(root / "data/judge_results.jsonl"),
            "errors_sha256": sha256_file(root / "logs/judge_errors.jsonl"),
        },
    )
    artifact = tmp_path / "comparison.json"
    write_json_atomic(artifact, {"version": "V0.5"})
    release = tmp_path / "release_manifest.json"
    write_json_atomic(
        release,
        {
            "version": "V0.5",
            "version_definition": {"framework": "Sarah MinHash judging framework"},
            "result_run_id": "sarah-qwen",
            "result_run_root": str(tmp_path / "sarah-qwen"),
            "source_v0_run_root": str(tmp_path / "v0-deepseek"),
            "frozen_pair_count": 1,
            "artifacts": {"comparison": {"path": str(artifact), "sha256": sha256_file(artifact)}},
        },
    )
    assert subject.baseline_root(release) == tmp_path / "sarah-qwen"
    (root / "data/judge_results.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(DedupEvaluationError, match="EXP5_V05_RESULTS_BINDING"):
        subject.baseline_root(release)


def test_mislabeled_v0_cannot_be_used_as_published_v05(tmp_path):
    release = tmp_path / "release_manifest.json"
    write_json_atomic(release, {"version": "V0", "version_definition": {"framework": "DeepSeek"}})
    with pytest.raises(DedupEvaluationError, match="EXP5_V05_RELEASE"):
        subject.baseline_root(release)


def test_comparison_does_not_score_failures_or_missing_baseline_as_agreement():
    yes = dict.fromkeys(subject.PRIMARY, "YES")
    no = dict.fromkeys(subject.PRIMARY, "NO")
    rows = subject.comparison_rows(
        [
            {"canonical_pair_id": "same", "status": "VALID", "public": yes},
            {"canonical_pair_id": "changed", "status": "VALID", "public": no},
            {"canonical_pair_id": "failed", "status": "ENGINEERING_FAILURE", "public": no},
            {"canonical_pair_id": "missing", "status": "VALID", "public": yes},
        ],
        {"same": yes, "changed": yes, "failed": no},
    )
    assert [r["primary_changed"] for r in rows] == [False, True, None, None]
    assert rows[2]["exp5_same_duplicate_group"] == "ENGINEERING_FAILURE"
    assert rows[3]["v05_status"] == "NO_VALID_OUTPUT"


def test_native_explorer_retained_and_partial_scope_explicit():
    summary = {"complete": False, "collected": 100, "population": 20000, "pending": 19900, "engineering_failures": 1}
    output = subject.explorer_html([], {}, summary)
    assert "Dedup Pair Explorer" in output
    assert "PARTIAL SNAPSHOT" in output
    assert "19,900 pending" in output
    assert "baselineComparison(r)" in output
    assert "UNAVAILABLE_MISSING_CONTRACT" in output
    assert 'id="language"' in output
    assert 'id="reason-grid"' in output
    assert "Version agreement is not accuracy" in output or "version agreement is not accuracy" in output


def test_final_sut_metrics_include_required_stage_and_keep_failures_in_denominator(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    first = {
        "token_length_ratio": 0.5,
        "predicted_group_size_low": 2,
        "judge_status": "valid",
        "same_duplicate_group": "YES",
        "judge_attempts": 1,
        "has_track_5a": True,
        "has_track_5b": False,
        "removal_outcome": "safe_removal",
        "cross_group_outcome": None,
        "removal_sampling_frame_size": 10,
        "removal_selection_probability": 0.1,
        "retriever_category": None,
    }
    second = {
        **first,
        "judge_status": "judge_error",
        "same_duplicate_group": None,
        "judge_attempts": 2,
        "has_track_5a": False,
        "has_track_5b": True,
        "removal_outcome": None,
        "cross_group_outcome": "judge_error",
        "retriever_category": "lexical_only",
    }
    path = tmp_path / "comparisons.parquet"
    pq.write_table(pa.Table.from_pylist([first, second]), path)
    metrics = subject.sut_metrics(path, tmp_path, prefix="exp5", requested=2, valid=1)
    assert metrics["judge"]["requested"] == 2
    assert metrics["judge"]["errors"] == 1
    assert metrics["judge"]["completion_rate"] == 0.5
    assert metrics["track_5a_removal_frame"]["safe"] == 1
    assert metrics["track_5b_candidate_pool"]["judge_errors"] == 1
    assert (tmp_path / "exp5_pipeline_accounting.csv").is_file()


def test_complete_report_real_parquet_metrics_calibration_and_views(tmp_path, monkeypatch):  # noqa: PLR0915 - real end-to-end artifact fixture
    import pyarrow as pa
    import pyarrow.parquet as pq

    from eval.dedup.analysis import checkpoint_preflight
    from eval.dedup.analysis import exp1_reproduction as historical

    root, source, baseline, reference = (tmp_path / name for name in ("run", "source", "sarah", "reference"))
    for base in (root, source, baseline):
        (base / "data").mkdir(parents=True)
    candidates, inputs, results, old, labels, outcomes = [], [], [], [], [], []
    for i in (1, 2):
        key, a, b = f"cp1_case{i}", 2 * i - 1, 2 * i
        payload = checkpoint_preflight.synthetic_payload("The same retained text.", "The same retained text.")
        payload["payload_schema_version"] = "judge-visible-payload-v3"
        candidates.append(
            {
                "canonical_pair_id": key,
                "canonical_pair_id_version": "cp1",
                "judge_payload_hash": "old-hash",
                "doc_id_low": a,
                "doc_id_high": b,
                "presented_doc_a": a,
                "presented_doc_b": b,
                "token_count_low": 10,
                "token_count_high": 10,
                "language_low": "en",
                "language_high": "en",
                "hostname_low": "a.example",
                "hostname_high": "a.example",
                "evaluation_run_id": "old",
            }
        )
        for doc in (a, b):
            outcomes.append(
                {
                    "doc_id": doc,
                    "predicted_group_id": 1 if i == 1 else -1,
                    "predicted_cluster_key": "group1" if i == 1 else f"single{doc}",
                    "predicted_group_size": 2 if i == 1 else 1,
                    "action": "REMOVE" if doc == 2 else "KEEP",
                    "final_keeper_id": 1 if i == 1 else doc,
                    "token_count": 10,
                    "hostname": "a.example",
                    "language": "en",
                    "url": "https://a.example/page",
                }
            )
        decision = {
            **subject.experiment.runtime.old.unresolved_judge_output_v3(),
            **dict.fromkeys(subject.PRIMARY, "YES"),
            "relation_type": "EXACT",
            "material_difference": "NONE",
            "primary_material_difference": "NONE",
            "confidence_tier": "HIGH",
            "reason_codes": [],
        }
        result = {
            "canonical_pair_id": key,
            "review_id": f"R{i}",
            "version": subject.experiment.runtime.VERSION,
            "status": "VALID" if i == 1 else "ENGINEERING_FAILURE",
            "error_code": "FIXTURE_FAILURE",
            "stages": [{"stage": "main-01-01"}],
            "main_attempts": [{"requests": 1}],
            "public": decision if i == 1 else subject.experiment.runtime.old.unresolved_judge_output_v3(),
        }
        results.append(result)
        inputs.append({"canonical_pair_id": key, "review_id": f"R{i}"})
        old.append(
            {
                **decision,
                "canonical_pair_id": key,
                "canonical_pair_id_version": "cp1",
                "judge_payload_hash": "old-hash",
                "attempts": 1,
            }
        )
        labels.append(
            {
                "canonical_pair_id": key,
                "review_id": f"R{i}",
                **{f"human_{k}": decision[k] for k in (*subject.PRIMARY, "relation_type", "material_difference")},
                "stratum_population_n": "10",
                "stratum_sample_n": "10",
            }
        )
        write_json_atomic(root / "inputs" / f"{key}.json", {**inputs[-1], "payload": payload})
        write_json_atomic(root / "results" / f"{key}.json", result)
    for base in (source, baseline):
        pq.write_table(pa.Table.from_pylist(candidates), base / "data/candidate_pairs.parquet")
    pq.write_table(pa.Table.from_pylist(outcomes), source / "data/document_outcomes.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "canonical_pair_id": "cp1_case1",
                    "track": "5a",
                    "ordered_keeper_id": 1,
                    "ordered_removed_id": 2,
                    "frame_size": 10,
                    "selection_probability": 0.1,
                    "retriever_bitmask": None,
                },
                {
                    "canonical_pair_id": "cp1_case2",
                    "track": "5b",
                    "ordered_keeper_id": None,
                    "ordered_removed_id": None,
                    "frame_size": None,
                    "selection_probability": None,
                    "retriever_bitmask": "lexical_only",
                },
            ]
        ),
        source / "data/pair_provenance.parquet",
    )
    write_json_atomic(source / "manifests/sut_run_manifest.json", {})
    write_text_atomic(baseline / "data/judge_results.jsonl", "".join(json.dumps(r) + "\n" for r in old))
    write_text_atomic(baseline / "logs/judge_errors.jsonl", "")
    write_json_atomic(
        baseline / "run_complete.json",
        {
            "requested": 2,
            "valid": 2,
            "errors": 0,
            "results_sha256": sha256_file(baseline / "data/judge_results.jsonl"),
            "errors_sha256": sha256_file(baseline / "logs/judge_errors.jsonl"),
        },
    )
    release = tmp_path / "release.json"
    write_json_atomic(
        release,
        {
            "version": "V0.5",
            "version_definition": {"framework": "Sarah MinHash judging framework"},
            "result_run_id": baseline.name,
            "result_run_root": str(baseline),
            "frozen_pair_count": 2,
            "artifacts": {},
        },
    )
    monkeypatch.setattr(subject, "V05_RELEASE", release)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(labels[0]))
    writer.writeheader()
    writer.writerows(labels)
    write_text_atomic(reference / "reference_revised_1000.csv", buffer.getvalue())
    write_json_atomic(reference / "comparison_exclusions.json", {"excluded_canonical_pair_ids": []})
    monkeypatch.setattr(historical, "REBENCHMARK", reference)
    write_json_atomic(root / "panel_index.json", inputs)
    manifest = {"population": 2, "source_run": str(source), "sources": {}, "artifacts": {}}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(
        root / "complete.json", {"artifacts": {str(p): sha256_file(p) for p in (root / "results").glob("*.json")}}
    )
    destination = root / "presentation/final"
    built = subject.build(root, destination)
    assert built["complete"] is True
    assert built["engineering_failures"] == 1
    summary = subject.experiment.read(destination / "reports/comparison.json")
    assert summary["judge_conditioned_sut_metrics"]["judge"]["errors"] == 1
    assert summary["baseline_judge_conditioned_sut_metrics"]["judge"]["errors"] == 0
    assert summary["development_calibration"]["exp5"]["weighted"]["duplicate_recall"] == 0.5
    assert summary["development_calibration"]["v05"]["weighted"]["duplicate_recall"] == 1
    assert summary["development_calibration"]["exp5_empirical_confidence_tiers"]["HIGH"]["rows"] == 1
    assert (destination / "reports/comparison.html").is_file()
    assert "v0.5 vs Exp5" in (destination / "reports/pair_explorer_exp5.html").read_text()
    snapshot = subject.experiment.read(destination / "snapshot.json")
    assert all(sha256_file(subject.Path(p)) == digest for p, digest in snapshot["files"].items())
