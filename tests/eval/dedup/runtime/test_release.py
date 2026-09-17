from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.dedup.core.validation import DedupEvaluationError, sha256_file, sha256_json
from eval.dedup.runtime import contract, release, state


def _manifest(*, current: bool) -> dict:
    value = {
        "version": "v0.7.1" if current else "v0.7",
        "runtime_version": "v0.7",
        "generation": contract.GENERATION,
        "old_answers_reused": False,
    }
    if current:
        value.update(tool_version="v0.7.1", judge_contract_version="v0.7")
    value["contract_digest"] = sha256_json(value)
    return value


@pytest.mark.parametrize("current", [False, True])
def test_verify_accepts_current_and_legacy_v07_manifest(tmp_path: Path, current: bool) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(current=current)), encoding="utf-8")

    assert release.verify(tmp_path)["runtime_version"] == "v0.7"


def test_legacy_smoke_audit_never_creates_a_completion_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "smoke_panel.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        release,
        "replay_results",
        lambda _root, _rows: {
            "population": 0,
            "statuses": {},
            "stages": {},
            "semantic_unresolved": 0,
            "main_retry_pairs": 0,
            "repairs": {},
            "saved_calls_replayed": 0,
            "replayed_keys": [],
            "artifacts": {},
            "offline_replay_identical": True,
        },
    )

    with pytest.raises(state.CheckpointIntegrityError):
        release.check_smoke(
            tmp_path,
            {"smoke_size": 0, "required_smoke_stages": [], "contract_digest": "legacy"},
            persist_missing=False,
        )
    assert not (tmp_path / "smoke_complete.json").exists()


def test_smoke_only_root_cannot_continue_as_full_run() -> None:
    manifest = {"mode": release.SMOKE_ONLY_MODE}

    release.require_execution_mode(manifest, smoke_only=True)
    with pytest.raises(DedupEvaluationError, match="DEDUP_SMOKE_ONLY_ROOT"):
        release.require_execution_mode(manifest, smoke_only=False)


def test_relocated_source_file_retains_checksum_validation(tmp_path: Path) -> None:
    relative = Path("inputs/pair.json")
    target = tmp_path / relative
    target.parent.mkdir()
    target.write_text('{"text": "frozen"}', encoding="utf-8")
    digest = sha256_file(target)
    bindings = release._index_bound_digests(
        {"/previous/machine/run/inputs/pair.json": digest, "inputs/pair.json": digest}
    )

    assert release._verify_bound_file(tmp_path, relative, bindings, error_code="SOURCE_CHANGED") == target

    target.write_text('{"text": "changed"}', encoding="utf-8")
    with pytest.raises(DedupEvaluationError, match="SOURCE_CHANGED"):
        release._verify_bound_file(tmp_path, relative, bindings, error_code="SOURCE_CHANGED")


@pytest.mark.parametrize(
    "bindings",
    [
        {},
        {"/previous/not-inputs/pair.json": "a"},
        {"/first/inputs/pair.json": "a", "/second/inputs/pair.json": "b"},
    ],
)
def test_source_digest_requires_an_unambiguous_path_suffix(bindings: dict[str, str]) -> None:
    with pytest.raises(DedupEvaluationError, match="SOURCE_BINDING"):
        release._bound_digest(
            release._index_bound_digests(bindings), Path("inputs/pair.json"), error_code="SOURCE_BINDING"
        )


@pytest.mark.skipif(not os.environ.get("CURATOR_V07_VERIFIED_RUN"), reason="archived v0.7 bundle not configured")
def test_archived_full20k_replays_identically() -> None:
    root = Path(os.environ["CURATOR_V07_VERIFIED_RUN"])
    manifest = release.verify(root)
    report = release.replay_results(root, release.read(root / "panel_index.json"))

    assert manifest["contract_digest"] == "bdd2cf795479a7a6a9d225ff51c7f29f4cf8c1089bab30c75a65546814a530b6"
    assert report["population"] == 20_000
    assert report["statuses"] == {"VALID": 20_000}
    assert report["saved_calls_replayed"] == 24_105
    assert report["offline_replay_identical"] is True
