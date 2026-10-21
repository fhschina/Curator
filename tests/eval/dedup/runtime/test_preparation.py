from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from eval.dedup.core.config import DatasetConfig, EvaluationConfig

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from eval.dedup.core.config import ProfileConfig
from eval.dedup.core.validation import DedupEvaluationError, sha256_json, write_json_atomic
from eval.dedup.runtime import backends, contract, preparation, release, state
from tests.eval.dedup.handoff.test_paths import make_inputs


@pytest.fixture
def small_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    original = preparation.preparation_config

    def create(root: Path, dataset: DatasetConfig) -> EvaluationConfig:
        config = original(root, dataset)
        profile = ProfileConfig(
            "full", {"singleton": 2, "size_2": 2, "size_3_5": 0, "size_6_20": 0, "size_21_plus": 0}, 4, 8, 0, 0, False
        )
        return replace(
            config,
            tokenizer=replace(config.tokenizer, kind="whitespace"),
            retrieval=replace(
                config.retrieval,
                backend="fixture_cpu",
                num_hashes=4,
                char_ngram_width=3,
                lsh_grid=((1, 1),),
                pilot_target_min=0,
                pilot_target_max=100,
                top_k=6,
            ),
            profiles={"full": profile},
        )

    monkeypatch.setattr(preparation, "preparation_config", create)


@pytest.mark.usefixtures("small_config")
def test_prepare_generates_new_pairs_and_text_without_source_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CURATOR_V07_SOURCE_RUN", str(tmp_path / "must-not-be-read"))
    runs = []
    removal_populations = []
    for prefix, combined in (("first-corpus", False), ("second-corpus", True)):
        paths = make_inputs(tmp_path / prefix, prefix=prefix, combined=combined)
        if combined:
            for path in [*paths["groups"].glob("*.parquet"), paths["removals"]]:
                data = pq.read_table(path).to_pydict()
                data["_curator_dedup_id"] = [(value + 12) % 24 for value in data["_curator_dedup_id"]]
                pq.write_table(pa.table(data), path)
        root = tmp_path / (prefix + "-run")
        argv = ["prepare", "--root", str(root)]
        for name, path in paths.items():
            argv.extend(["--" + name, str(path)])
        assert backends.main(argv) == 0
        manifest = release.verify(root)
        assert manifest["population"] == 12
        assert manifest["required_smoke_stages"] == ["main"]
        assert manifest["dataset"]["expected_rows"] == 24
        assert not manifest["old_answers_reused"]
        assert "source_run" not in manifest
        pairs = pq.read_table(root / "preparation/data/candidate_pairs.parquet").to_pylist()
        by_key = {row["canonical_pair_id"]: row for row in pairs}
        rows = release.read(root / "panel_index.json")
        removal_populations.append({row["canonical_pair_id"] for row in rows if row["track"] == "5a"})
        assert {row["track"] for row in rows} == {"5a", "5b"}
        for row in rows:
            value = release.read(root / "inputs" / (row["canonical_pair_id"] + ".json"))
            pair = by_key[row["canonical_pair_id"]]
            for side in ("a", "b"):
                assert (
                    value["payload"][f"document_{side}"]["text"]
                    == f"{prefix} example article about shared subject and document {pair[f'presented_doc_{side}']}"
                )
        runs.append(manifest)
    assert runs[0]["input_manifest_sha256"] != runs[1]["input_manifest_sha256"]
    assert runs[0]["dataset"]["dataset_version"] != runs[1]["dataset"]["dataset_version"]
    assert removal_populations[0].isdisjoint(removal_populations[1])
    payload_file = next((root / "inputs").glob("*.json"))
    payload_file.write_text("{}")
    with pytest.raises(DedupEvaluationError, match="INPUT_ARTIFACT_CHANGED"):
        release.verify(root)


@pytest.mark.parametrize("sizes", [(30, 30), (2, 30), (2, 3)])
def test_smoke_is_deterministic_balanced_and_bounded(sizes: tuple[int, int]) -> None:
    rows = [
        {"canonical_pair_id": f"pair-{track}-{index}", "review_id": str(index), "track": track}
        for track, count in zip(("5a", "5b"), sizes, strict=True)
        for index in range(count)
    ]
    selected = preparation.population_smoke(rows)
    assert selected == preparation.population_smoke(list(reversed(rows)))
    assert len(selected) == min(24, sum(sizes))
    assert len({row["canonical_pair_id"] for row in selected}) == len(selected)
    if sizes == (30, 30):
        assert sum(row["track"] == "5a" for row in selected) == 12


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["--documents", "new"], "INPUT_PATHS_REQUIRED"),
        (
            ["--documents", "d", "--groups", "g", "--removals", "r", "--embeddings", "e", "--source-run", "old"],
            "INPUT_PATHS_CONFLICT",
        ),
    ],
)
def test_cli_never_falls_back_to_old_data(
    tmp_path: Path,
    arguments: list[str],
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CURATOR_V07_SOURCE_RUN", str(tmp_path / "old"))
    root = tmp_path / "run"
    assert backends.main(["prepare", "--root", str(root), *arguments]) == 1
    assert json.loads(capsys.readouterr().out)["error_code"] == error
    assert not root.exists()


@pytest.mark.usefixtures("small_config")
def test_new_smoke_replays_without_requiring_untriggered_critics(tmp_path: Path) -> None:
    paths = make_inputs(tmp_path / "input")
    root = tmp_path / "run"
    preparation.prepare(root, input_paths=paths)
    manifest = release.verify(root)
    collect = offline_collector(root)
    with state.use_collector(collect):
        for row in release.read(root / "smoke_panel.json"):
            result = release.execute(root, row, "fixture://judge", contract.coverage_renderer())
            assert result["status"] == "VALID"
    report = release.check_smoke(root, manifest)
    assert report["passed"]
    assert report["offline_replay_identical"]
    assert report["stages"] == {"main": 12}
    assert report["saved_calls_replayed"] == 12


def offline_collector(root: Path):
    scores = yaml.safe_load(contract.MAIN_CONFIG.read_text())["execution"]["stages"][0]["judges"][0]["scores"]
    raw = {}
    for score in scores:
        options = score["options"]
        value = (
            "unresolved"
            if "unresolved" in options
            else "unreadable"
            if "unreadable" in options
            else next(iter(options))
        )
        raw[score["name"]] = {"score": value, "reasoning": f"Insufficient evidence to establish {score['name']}."}

    def collect(destination: Path, key: str, request: dict, endpoint: str) -> dict:
        assert destination == root
        assert endpoint == "fixture://judge"
        digest = sha256_json(request)
        write_json_atomic(root / "requests" / (key + ".json"), {"body": request, "request_sha256": digest})
        receipt = {
            "status": "RECEIVED",
            "request_sha256": digest,
            "raw_response": {
                "id": key,
                "object": "chat.completion",
                "created": 0,
                "model": contract.LOGICAL_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "```json\n" + json.dumps(raw) + "\n```"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        }
        write_json_atomic(root / "responses" / (key + ".json"), receipt)
        return receipt

    return collect


@pytest.mark.usefixtures("small_config")
def test_smoke_only_freezes_only_its_endpoint_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def smoke(rows: list[dict]) -> list[dict]:
        return [next(row for row in rows if row["track"] == track) for track in ("5a", "5b")]

    monkeypatch.setattr(preparation, "population_smoke", smoke)
    root = tmp_path / "run"
    preparation.prepare(root, input_paths=make_inputs(tmp_path / "input"), smoke_only=True)
    manifest = release.verify(root)
    assert manifest["population"] == 2
    assert manifest["source_population"] == 12
    keys = [row["canonical_pair_id"] for row in release.read(root / "panel_index.json")]
    pairs = pq.read_table(
        root / "preparation/data/candidate_pairs.parquet", filters=[("canonical_pair_id", "in", keys)]
    ).to_pylist()
    expected = {row[side] for row in pairs for side in ("doc_id_low", "doc_id_high")}
    assert set(pq.read_table(root / "sample_documents.parquet")["doc_id"].to_pylist()) == expected
    assert manifest["sample_document_count"] == len(expected)
    with state.use_collector(offline_collector(root)):
        for row in release.read(root / "panel_index.json"):
            release.execute(root, row, "fixture://judge", contract.coverage_renderer())
    assert release.audit(root)["sample_report"]["documents"] == len(expected)
