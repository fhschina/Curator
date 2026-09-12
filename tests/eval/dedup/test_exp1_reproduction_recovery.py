# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from eval.dedup.analysis import exp1_reproduction_recovery as subject
from eval.dedup.analysis import exp1_reproduction_runtime as runtime
from eval.dedup.validation import sha256_file, sha256_json, write_json_atomic, write_text_atomic
from tests.eval.dedup.test_critic_retention_v3 import fixture as coverage_fixture
from tests.eval.dedup.test_critic_subject_binding import fixture as subject_fixture
from tests.eval.dedup.test_exp1_reproduction_runtime import fixtures, server


def case():
    payload = fixtures.synthetic_payload("Same policy.", "Same policy.")
    payload["payload_schema_version"] = "judge-visible-payload-v3"
    row = {"canonical_pair_id": "pair", "review_id": "R1", "payload": payload}
    body = runtime.body(runtime.main_messages(payload))
    frozen = {"canonical_pair_id": "pair", "body": body, "request_sha256": sha256_json(body)}
    return row, frozen, fixtures.synthetic_raw(payload, deltas=("none", "none"))


def test_saved_response_replays_without_http_and_never_overwrites(tmp_path):
    row, frozen, raw = case()
    with server([raw]) as (endpoint, bodies):
        expected = runtime.collect(tmp_path, "pair-main-01-01", frozen["body"], endpoint)
        files = {p: sha256_file(p) for p in tmp_path.rglob("*.json")}
        collector = subject.Collector(tmp_path, tmp_path / "session")
        with subject.use_collector(collector):
            result = runtime.execute_case(tmp_path, row, endpoint, frozen, runtime.coverage_renderer())
        assert len(bodies) == 1
    assert result["status"] == "VALID"
    assert collector.replayed == 1
    assert collector.submitted == 0
    assert all(sha256_file(p) == digest for p, digest in files.items())
    assert subject.read(tmp_path / "responses/pair-main-01-01.json") == expected


def test_pending_request_resends_identical_body_and_preserves_request_file(tmp_path):
    _, frozen, raw = case()
    request = tmp_path / "requests/pair-main-01-01.json"
    write_json_atomic(request, {k: frozen[k] for k in ("body", "request_sha256")})
    before = request.stat().st_mtime_ns
    collector = subject.Collector(tmp_path, tmp_path / "session")
    with server([raw]) as (endpoint, bodies):
        receipt = collector(tmp_path, "pair-main-01-01", frozen["body"], endpoint)
        assert collector(tmp_path, "pair-main-01-01", frozen["body"], endpoint) == receipt
    assert bodies == [frozen["body"]]
    assert request.stat().st_mtime_ns == before
    assert collector.retransmitted == collector.submitted == collector.replayed == 1
    assert subject.events(tmp_path / "session/collection_events.jsonl")[0]["interrupted_request_retransmission"]


def test_checkpoint_mismatch_escapes_native_retry_and_does_not_create_result(tmp_path):
    row, frozen, _ = case()
    damaged = deepcopy(frozen["body"])
    damaged["messages"] = []
    write_json_atomic(
        tmp_path / "requests/pair-main-01-01.json", {"body": damaged, "request_sha256": sha256_json(damaged)}
    )
    with (
        subject.use_collector(subject.Collector(tmp_path, tmp_path / "session")),
        pytest.raises(subject.CheckpointIntegrityError),
    ):
        runtime.execute_case(tmp_path, row, "http://127.0.0.1:1/v1", frozen, runtime.coverage_renderer())
    assert not (tmp_path / "results/pair.json").exists()
    assert len(list((tmp_path / "requests").glob("*.json"))) == 1


def test_saved_failure_is_not_retried_by_recovery(tmp_path):
    _, frozen, _ = case()
    key = "pair-coverage"
    write_json_atomic(tmp_path / "requests" / (key + ".json"), {k: frozen[k] for k in ("body", "request_sha256")})
    receipt = {
        "request_sha256": frozen["request_sha256"],
        "status": "FAILURE",
        "error_code": "EXP1_HTTP",
        "http_status": 400,
    }
    write_json_atomic(tmp_path / "responses" / (key + ".json"), receipt)
    collector = subject.Collector(tmp_path, tmp_path / "session")
    assert collector(tmp_path, key, frozen["body"], "http://127.0.0.1:1/v1") == receipt
    assert collector.submitted == 0


def test_resume_native_format_correction_matches_clean_pipeline(tmp_path):
    row, frozen, raw = case()
    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    with server(["invalid json", raw]) as (endpoint, bodies):
        expected = runtime.execute_case(clean, row, endpoint, frozen, runtime.coverage_renderer())
    assert len(bodies) == 2
    assert expected["status"] == "VALID"
    for folder in ("requests", "responses"):
        write_json_atomic(
            resumed / folder / "pair-main-01-01.json", subject.read(clean / folder / "pair-main-01-01.json")
        )
    collector = subject.Collector(resumed, resumed / "session")
    with server([raw]) as (endpoint, bodies), subject.use_collector(collector):
        actual = runtime.execute_case(resumed, row, endpoint, frozen, runtime.coverage_renderer())
    assert len(bodies) == 1
    assert bodies[0] == subject.read(clean / "requests/pair-main-01-02.json")["body"]
    assert actual == expected


def test_completed_pairs_are_skipped_and_partial_case_is_reconstructed(tmp_path):
    row, frozen, raw = case()
    write_json_atomic(tmp_path / "panel_private.json", [row])
    write_json_atomic(tmp_path / "main_requests.json", [frozen])
    with server([raw]) as (endpoint, bodies):
        runtime.collect(tmp_path, "pair-main-01-01", frozen["body"], endpoint)
        assert subject.inventory(tmp_path)["partial_review_ids"] == ["R1"]
        relay = SimpleNamespace(endpoint=endpoint, _circuit_reason=None)
        collector = subject.Collector(tmp_path, tmp_path / "session")
        subject.collect_pending(tmp_path, tmp_path / "session", relay, collector)
        original_result = (tmp_path / "results/pair.json").read_bytes()
        subject.collect_pending(tmp_path, tmp_path / "session2", relay, collector)
        assert (tmp_path / "results/pair.json").read_bytes() == original_result
        assert len(bodies) == 1
    assert subject.inventory(tmp_path)["completed_pair_ids"] == ["pair"]


@pytest.mark.parametrize("damage", ["receipt_hash", "orphan_response", "result_binding", "unknown_request"])
def test_inventory_rejects_corruption(tmp_path, damage):
    row, frozen, raw = case()
    write_json_atomic(tmp_path / "panel_private.json", [row])
    with server([raw]) as (endpoint, _):
        result = runtime.execute_case(tmp_path, row, endpoint, frozen, runtime.coverage_renderer())
    if damage == "receipt_hash":
        # Damage only the fixture, not the append-only production writer.
        path = tmp_path / "responses/pair-main-01-01.json"
        value = subject.read(path)
        value["request_sha256"] = "bad"
        path.write_text(json.dumps(value))
    elif damage == "orphan_response":
        write_json_atomic(tmp_path / "responses/orphan.json", {})
    elif damage == "result_binding":
        result["review_id"] = "other"
        (tmp_path / "results/pair.json").write_text(json.dumps(result))
    else:
        write_json_atomic(
            tmp_path / "requests/unknown-main-01-01.json", {k: frozen[k] for k in ("body", "request_sha256")}
        )
    with pytest.raises(subject.CheckpointIntegrityError):
        subject.inventory(tmp_path)


def test_exclusive_lock_rejects_second_process_and_releases_after_exit(tmp_path):
    code = "from pathlib import Path; from eval.dedup.analysis.exp1_reproduction_recovery import exclusive; import sys\nwith exclusive(Path(sys.argv[1])): pass"
    with subject.exclusive(tmp_path):
        other = subprocess.run(  # noqa: S603 - fixed local lock probe
            [sys.executable, "-c", code, str(tmp_path)], check=False, capture_output=True, text=True
        )
        assert other.returncode != 0
        assert "ALREADY_RUNNING" in other.stderr
    other = subprocess.run(  # noqa: S603 - fixed local lock probe
        [sys.executable, "-c", code, str(tmp_path)], check=False, capture_output=True, text=True
    )
    assert other.returncode == 0


def test_status_detects_dead_or_reused_process_and_exit_record(tmp_path):
    session = tmp_path / "recovery/sessions/001"
    write_json_atomic(
        session / "started.json", {"pid": os.getpid(), "process_start": subject.process_start(os.getpid())}
    )
    assert subject.status(tmp_path)["running"]
    write_json_atomic(session / "exit.json", {"status": "STOPPED"})
    assert not subject.status(tmp_path)["running"]
    other = tmp_path / "recovery/sessions/002"
    write_json_atomic(other / "started.json", {"pid": os.getpid(), "process_start": "not-current-process"})
    assert not subject.status(tmp_path)["running"]


def test_partial_coverage_resume_preserves_full_chain_and_original_audit(tmp_path):
    payload, _, proposal = subject_fixture()
    payload["payload_schema_version"] = "judge-visible-payload-v3"
    raw = fixtures.synthetic_raw(
        payload,
        profiles=("non_main_only", "non_main_only"),
        deltas=("universal_ui_or_repetition", "universal_ui_or_repetition"),
        basis="verified_equivalent_non_main_message",
    )
    _, _, coverage = coverage_fixture(payload["document_a"]["text"], payload["document_b"]["text"])
    verification = {
        "a_subject_kind": "NAMED_ACTUAL_TARGET",
        "b_subject_kind": "NAMED_ACTUAL_TARGET",
        "comparison": "SUPPORTED_DIFFERENT_NAMED_TARGETS",
        "explanation": "Distinct named liability parties.",
    }
    row = {"canonical_pair_id": "pair", "review_id": "R1", "payload": payload}
    body = runtime.body(runtime.main_messages(payload))
    frozen = {"canonical_pair_id": "pair", "body": body, "request_sha256": sha256_json(body)}
    clean, recovered = tmp_path / "clean", tmp_path / "recovered"
    with server([raw, json.dumps(coverage), json.dumps(proposal), json.dumps(verification)]) as (endpoint, _):
        expected = runtime.execute_case(clean, row, endpoint, frozen, runtime.coverage_renderer())
    assert expected["status"] == "VALID"
    for folder, name in (
        ("requests", "pair-main-01-01.json"),
        ("responses", "pair-main-01-01.json"),
        ("requests", "pair-coverage.json"),
    ):
        write_json_atomic(recovered / folder / name, subject.read(clean / folder / name))
    write_json_atomic(recovered / "main_requests.json", [frozen])
    write_json_atomic(recovered / "manifest.json", {"sources": {}, "artifacts": {}})
    collector = subject.Collector(recovered, recovered / "session")
    with (
        server([json.dumps(coverage), json.dumps(proposal), json.dumps(verification)]) as (endpoint, bodies),
        subject.use_collector(collector),
    ):
        actual = runtime.execute_case(recovered, row, endpoint, frozen, runtime.coverage_renderer())
    assert actual == expected
    assert len(bodies) == 3
    assert collector.retransmitted == collector.replayed == 1
    audit = subject.original.audit(recovered, [row], [actual])
    assert audit["valid_pipeline_replays"] == 1
    assert audit["bound_calls"] == 4


def test_local_end_to_end_recovery_exports_original_scores_and_preserves_freeze(tmp_path, monkeypatch):
    import csv
    import io

    row, frozen, raw = case()
    write_json_atomic(tmp_path / "panel_private.json", [row])
    write_json_atomic(tmp_path / "main_requests.json", [frozen])
    public = runtime.common.critic.main_decision(raw, row["payload"])
    primary = subject.original.benchmark.primary(public)
    labels = [
        {
            "canonical_pair_id": "pair",
            "review_id": "R1",
            "stratum_population_n": 10,
            "stratum_sample_n": 10,
            "human_reason_code": "identity_slot",
            "comparison_scoring_eligibility": "INCLUDED",
            **{"human_" + k: v for k, v in primary.items()},
        }
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
    writer.writeheader()
    writer.writerows(labels)
    write_text_atomic(tmp_path / "reference_revised_1000.csv", stream.getvalue())
    write_json_atomic(tmp_path / "comparison_exclusions.json", {"excluded_canonical_pair_ids": []})
    write_json_atomic(
        tmp_path / "historical_predictions.json",
        {v: [{"canonical_pair_id": "pair", **primary}] for v in subject.original.HISTORICAL},
    )
    write_text_atomic(tmp_path / "transport_events.jsonl", "")
    write_json_atomic(tmp_path / "started.json", {})
    monkeypatch.setenv("NVIDIA_API_KEY", "local-test-only")
    with server([raw]) as (endpoint, bodies):
        manifest = {
            "sources": {},
            "artifacts": {},
            "endpoint": endpoint,
            "model": runtime.LOGICAL_MODEL,
            "max_external_attempts": 20,
        }
        manifest["contract_digest"] = sha256_json(manifest)
        write_json_atomic(tmp_path / "manifest.json", manifest)
        initial_hash = sha256_file(tmp_path / "manifest.json")
        subject.prepare(tmp_path)
        subject.run(tmp_path, tmp_path / "absent.env", tmp_path / "recovery/sessions/001")
    assert len(bodies) == 1
    assert sha256_file(tmp_path / "manifest.json") == initial_hash
    subject.verify(tmp_path)
    report = subject.read(tmp_path / "comparison.json")
    assert report["offline_audit"]["valid_pipeline_replays"] == 1
    assert report["engineering"]["external_attempts"] == 1
    assert report["scores"][subject.original.FRESH]["weighted"]["duplicate_precision"] == 1
    assert subject.read(tmp_path / "complete.json")["population"] == 1
    assert subject.status(tmp_path)["exit"]["status"] == "COMPLETE"
    assert not subject.status(tmp_path)["running"]
