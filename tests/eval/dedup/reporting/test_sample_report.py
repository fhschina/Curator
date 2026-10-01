from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from eval.dedup.core.validation import sha256_file
from eval.dedup.runtime import preparation, release, state
from tests.eval.dedup.handoff.test_paths import make_inputs
from tests.eval.dedup.runtime import test_preparation as preparation_tests
from tests.eval.dedup.runtime.test_preparation import offline_collector

small_config = preparation_tests.small_config


@pytest.mark.usefixtures("small_config")
def test_audit_reports_only_frozen_documents_and_rebuilds_without_calls(tmp_path: Path) -> None:
    root = tmp_path / "run"
    preparation.prepare(root, input_paths=make_inputs(tmp_path / "input"))
    manifest = release.verify(root)
    with state.use_collector(offline_collector(root)):
        for row in release.read(root / "panel_index.json"):
            release.execute(root, row, "fixture://judge", release.contract.coverage_renderer())
    report = release.audit(root)
    assert report["population"] == 12
    assert report["sample_report"]["statistics_scope"] == "frozen_pairs"
    assert report["sample_report"]["documents"] == manifest["sample_document_count"] < 24
    assert (root / "complete.json").exists()
    ledger = {
        str(path): sha256_file(path)
        for folder in ("requests", "responses", "results")
        for path in (root / folder).glob("*.json")
    }
    hashes = report["sample_report"]["artifacts"]
    background = root / "background_tokens/arbitrary-progress"
    background.mkdir(parents=True)
    (background / "worker.json").write_text('{"counted": 99999999}')
    (background / "part.parquet").write_bytes(b"unusable partial background data")
    shutil.rmtree(root / "reports")
    assert release.audit(root) == report
    assert all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    assert all(sha256_file(Path(path)) == digest for path, digest in ledger.items())
    release.verify(root)
