# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from eval.dedup.analysis import exp5_presentation_finalize as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic, write_text_atomic


def fixture(root, *, final=False):
    snapshot = root / "presentation/snapshot"
    write_json_atomic(root / "manifest.json", {"population": 20000, "contract_digest": "test-contract"})
    write_json_atomic(
        snapshot / "reports/comparison.json",
        {
            "population": 20000,
            "collected": 20000 if final else 10,
            "complete": final,
        },
    )
    for name in ("pair_explorer_exp5.html", "comparison.html", "v05_exp5_pairs.csv", "RESULTS.md"):
        write_text_atomic(snapshot / "reports" / name, name)
    files = {str(p): sha256_file(p) for p in (snapshot / "reports").iterdir()}
    write_json_atomic(
        snapshot / "snapshot.json",
        {
            "source_contract_digest": "test-contract",
            "preview": not final,
            "result_bindings": {},
            "files": files,
        },
    )
    return snapshot


def test_preview_publish_is_atomic_and_old_snapshot_is_preserved(tmp_path):
    snapshot = fixture(tmp_path)
    old = tmp_path / "presentation/older"
    old.mkdir()
    (tmp_path / "presentation/current").symlink_to(old)
    result = subject.publish(tmp_path, snapshot)
    assert result["complete"] is False
    assert (tmp_path / "presentation/current").resolve() == snapshot
    assert old.is_dir()
    assert subject.publish(tmp_path, snapshot) == result


def test_modified_artifact_is_not_published(tmp_path):
    snapshot = fixture(tmp_path)
    (snapshot / "reports/comparison.html").write_text("modified")
    with pytest.raises(DedupEvaluationError, match="EXP5_PUBLISH_BINDING"):
        subject.publish(tmp_path, snapshot)
    assert not (tmp_path / "presentation/current").exists()


def test_final_requires_audited_full_population(tmp_path):
    snapshot = fixture(tmp_path, final=True)
    write_json_atomic(
        tmp_path / "complete.json",
        {
            "population": 19999,
            "fresh_full20k": False,
            "contract_digest": "test-contract",
            "artifacts": {},
        },
    )
    with pytest.raises(DedupEvaluationError, match="EXP5_PUBLISH_COMPLETION"):
        subject.publish(tmp_path, snapshot)


def test_existing_user_directory_is_never_replaced(tmp_path):
    snapshot = fixture(tmp_path)
    (tmp_path / "presentation/current").mkdir()
    with pytest.raises(DedupEvaluationError, match="EXP5_CURRENT_TYPE"):
        subject.publish(tmp_path, snapshot)


def test_finalizer_stops_if_collector_is_not_live_and_does_not_manufacture_results(tmp_path):
    manifest = {"population": 20000, "sources": {}, "artifacts": {}}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(tmp_path / "manifest.json", manifest)
    session = tmp_path / "presentation/finalizer/session"
    with pytest.raises(DedupEvaluationError, match="EXP5_COLLECTION_STOPPED"):
        subject.watch(tmp_path, session, poll_seconds=0.001)
    assert subject.experiment.read(session / "exit.json")["error_code"] == "EXP5_COLLECTION_STOPPED"
    assert not (tmp_path / "complete.json").exists()
    assert not (tmp_path / "presentation/final-v1").exists()
