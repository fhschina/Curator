# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Freeze fresh Judge inputs from the four explicitly supplied data paths."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from eval.dedup.core.config import EvaluationConfig
from eval.dedup.core.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.handoff.paths import adapt_inputs, parquet_files
from eval.dedup.judging.payload import assert_blind_payload, build_visible_payload
from eval.dedup.pair_construction.prepare import build_population, preparation_config
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION, TOOL_VERSION, contract, release, state


def population_smoke(rows: list[dict]) -> list[dict]:
    """Select up to 24 fresh pairs, preferring twelve from each track."""
    ordered = sorted(
        rows, key=lambda row: (sha256_json([26081204, row["canonical_pair_id"]]), row["canonical_pair_id"])
    )
    selected = [row for track in ("5a", "5b") for row in [r for r in ordered if r["track"] == track][:12]]
    keys = {row["canonical_pair_id"] for row in selected}
    selected.extend(row for row in ordered if row["canonical_pair_id"] not in keys)
    return [{**row, "sampling_reason": "deterministic_new_population_engineering_smoke"} for row in selected[:24]]


def prepare(root: Path, *, input_paths: dict[str, Path], smoke_only: bool = False) -> dict:
    require(not root.exists(), "V07_FRESH_ROOT", "new data requires a fresh run root")
    dataset, corpus, sut, inventory = adapt_inputs(root / "preparation", **input_paths)
    print(
        json.dumps(
            {
                "prepared_stage": "inputs",
                "documents": dataset.expected_rows,
                "dimensions": dataset.embedding_dimensions,
            }
        ),
        flush=True,
    )
    config = preparation_config(root, dataset)
    tokenizer = TokenCounter(config.tokenizer)
    summary = build_population(config, corpus, sut, tokenizer)
    return freeze_population(
        root,
        config=config,
        corpus=corpus,
        inventory=inventory,
        tokenizer=tokenizer,
        summary=summary,
        smoke_only=smoke_only,
    )


def freeze_population(
    root: Path,
    *,
    config: EvaluationConfig,
    corpus: dict,
    inventory: dict,
    tokenizer: TokenCounter,
    summary: dict,
    smoke_only: bool = False,
) -> dict:
    import pyarrow.parquet as pq

    candidates = pq.read_table(config.output_root / "candidate_pairs.parquet").to_pylist()
    provenance = pq.read_table(
        config.output_root / "pair_provenance.parquet", columns=["canonical_pair_id", "track"]
    ).to_pylist()
    tracks = {}
    for event in provenance:
        key, track = event["canonical_pair_id"], event["track"]
        require(
            tracks.setdefault(key, track) == track,
            "PAIR_TRACK_COLLISION",
            "a pair cannot be both within and across predicted groups",
        )
    rows = [
        {
            "canonical_pair_id": candidate["canonical_pair_id"],
            "review_id": f"P{index:05d}",
            "track": tracks[candidate["canonical_pair_id"]],
        }
        for index, candidate in enumerate(candidates, 1)
    ]
    require(
        rows and len(rows) == len({row["canonical_pair_id"] for row in rows}),
        "INVALID_POPULATION",
        "new population must contain unique pairs",
    )
    smoke = population_smoke(rows)
    source_population = len(rows)
    if smoke_only:
        smoke_keys = {row["canonical_pair_id"] for row in smoke}
        rows = [row for row in rows if row["canonical_pair_id"] in smoke_keys]
    by_key = {row["canonical_pair_id"]: row for row in candidates}
    endpoint_ids = sorted(
        {
            int(by_key[row["canonical_pair_id"]][side])
            for row in rows
            for side in ("presented_doc_a", "presented_doc_b")
        }
    )
    sample_path = root / "sample_documents.parquet"
    sample = pq.read_table(config.output_root / "document_outcomes.parquet", filters=[("doc_id", "in", endpoint_ids)])
    require(sample.num_rows == len(endpoint_ids), "SAMPLE_DOCUMENT_JOIN", "one exact outcome per frozen endpoint")
    pq.write_table(sample, sample_path, compression="zstd")
    counts = {int(row["doc_id"]): int(row["token_count"]) for row in sample.to_pylist()}
    documents = load_documents_by_ids(corpus, endpoint_ids, columns=("text",))
    artifacts = {}
    for index, row in enumerate(rows, 1):
        key = row["canonical_pair_id"]
        candidate = by_key[key]
        payload, payload_hash = build_visible_payload(
            documents[int(candidate["presented_doc_a"])],
            documents[int(candidate["presented_doc_b"])],
            counter=tokenizer,
            config=config.judge,
            token_counts=(counts[int(candidate["presented_doc_a"])], counts[int(candidate["presented_doc_b"])]),
        )
        assert_blind_payload(payload)
        require(
            payload_hash == candidate["judge_payload_hash"],
            "PAYLOAD_CHANGED",
            "payload must match the newly constructed pair",
        )
        request = contract.body(contract.main_messages(payload))
        for folder, value in (
            ("inputs", {**row, "payload": payload}),
            ("main_requests", {"body": request, "request_sha256": sha256_json(request)}),
        ):
            path = root / folder / (key + ".json")
            write_json_atomic(path, value)
            artifacts[str(path)] = sha256_file(path)
        if index % 1000 == 0:
            print(json.dumps({"prepared": index}), flush=True)
    for kind, source_path in inventory["paths"].items():
        require(
            {str(path) for path in parquet_files(Path(source_path))} == set(inventory["shards"][kind]),
            "INPUT_CHANGED",
            "input shards changed during pair construction",
            input=kind,
        )
    for source_path, stat in inventory["files"].items():
        current = Path(source_path).stat()
        require(
            (current.st_size, current.st_mtime_ns) == (stat["size_bytes"], stat["mtime_ns"]),
            "INPUT_CHANGED",
            "input files changed during pair construction",
            path=source_path,
        )
    write_json_atomic(root / "panel_index.json", rows)
    write_json_atomic(root / "smoke_panel.json", smoke)
    write_text_atomic(root / "protocol.md", release.PROTOCOL.read_text())
    for path in (
        sample_path,
        root / "panel_index.json",
        root / "smoke_panel.json",
        root / "protocol.md",
        *sorted((root / "preparation").glob("*.json")),
        *sorted(config.output_root.glob("*.json")),
        config.output_root / "candidate_pairs.parquet",
        config.output_root / "pair_provenance.parquet",
        config.output_root / "anchors.parquet",
    ):
        artifacts[str(path)] = sha256_file(path)
    source_root = Path(__file__).resolve().parents[1]
    sources = release._current_sources(
        extras=[
            *sorted(source_root.rglob("*.py")),
            *sorted((release.RELEASE / "prompts").glob("*")),
            release.PROTOCOL,
            release.RELEASE_MANIFEST,
            *sorted((source_root.parents[1] / "nemo_curator/eval/llm_judge").glob("*.py")),
        ]
    )
    manifest = {
        "version": TOOL_VERSION,
        "runtime_version": JUDGE_CONTRACT_VERSION,
        "tool_version": TOOL_VERSION,
        "judge_contract_version": JUDGE_CONTRACT_VERSION,
        "mode": release.SMOKE_ONLY_MODE if smoke_only else release.FULL_MODE,
        "input_mode": "parquet_paths",
        "preparation_flow": "selected_documents_v1",
        "statistics_scope": "frozen_pairs",
        "sample_documents": "sample_documents.parquet",
        "sample_document_count": len(endpoint_ids),
        "artifact_root": str(root),
        "backend": "hub",
        "at_utc": state.now(),
        "distribution": "FHSCHINA_FORK_DEDUP_EVAL_ONLY",
        "population": len(rows),
        "source_population": source_population,
        "smoke_size": len(smoke),
        "required_smoke_stages": ["main"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root.parents[1], text=True
        ).strip(),
        "input_manifest": str(root / "preparation/inputs_manifest.json"),
        "input_manifest_sha256": sha256_file(root / "preparation/inputs_manifest.json"),
        "dataset": config.raw["dataset"],
        "tokenizer": tokenizer.contract(),
        "preparation": summary,
        "smoke_panel_source": "new_population",
        "smoke_panel_source_sha256": sha256_file(root / "smoke_panel.json"),
        "development_payload_matches": 0,
        "old_answers_reused": False,
        "smoke_included_in_population": True,
        "independent_holdout_passed": False,
        "release_eligible": False,
        "minhash_diagnostics": "RECORDED_IN_PREPARATION",
        "model": config.judge.model,
        "endpoint": config.judge.base_url,
        "generation": contract.GENERATION,
        "workers": 2,
        "min_interval_seconds": 2,
        "max_external_attempts": 120000,
        "sources": sources,
        "source_digest": sha256_json(sources),
        "contract_lineage": {"judge_contract_version": JUDGE_CONTRACT_VERSION},
        "artifacts": artifacts,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key not in {"sources", "artifacts"}}
