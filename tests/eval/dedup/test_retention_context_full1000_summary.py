# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import pytest

from eval.dedup.analysis import retention_context_full1000_summary as subject
from eval.dedup.validation import DedupEvaluationError, write_json_atomic
from tests.eval.dedup.test_retention_context_v1 import payload, review


def test_successive_progress_is_append_only_and_first_snapshot_is_never_overwritten(tmp_path):
    subject.record_progress(tmp_path, 1, 1000)
    first = (tmp_path / "progress_snapshots/0001.json").read_bytes()
    subject.record_progress(tmp_path, 2, 1000)
    assert (tmp_path / "progress_snapshots/0001.json").read_bytes() == first
    assert len(list((tmp_path / "progress_snapshots").glob("*.json"))) == 2
    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        subject.record_progress(tmp_path, 1, 999)


def test_offline_audit_replays_own_main_and_critic_without_inference_or_explanation_override():
    p = payload("This claim is false: Service free.", "Service free.")
    adapter = subject.experiment.pilot.candidate
    raw = review(p)
    critic = review(p, conflict="NEGATION_CONFLICT")
    main = adapter.adapt_main(raw, p)
    final = adapter.apply_critic(main, p, critic)[0]
    row = {"payload": p}
    result = {"components": {"main": main, "coverage": final}, "public": final}
    values = {"main": raw, "coverage": critic}
    before = deepcopy((row, result, values))
    assert subject.replay_valid(row, result, values.__getitem__) == final
    assert (row, result, values) == before
    result["public"] = main
    with pytest.raises(DedupEvaluationError, match="CONTEXT_SUMMARY_FINAL"):
        subject.replay_valid(row, result, values.__getitem__)


def test_summary_does_not_score_an_unbound_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        subject.audit(tmp_path)


def test_immutable_progress_collision_does_not_cancel_already_submitted_worker_results(tmp_path):
    def collect():
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write_json_atomic, tmp_path / "results" / f"{i}.json", {"id": i}) for i in range(8)]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                write_json_atomic(tmp_path / "legacy_progress.json", {"completed": count})

    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        collect()
    assert len(list((tmp_path / "results").glob("*.json"))) == 8
