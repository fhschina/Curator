# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Deterministic construction of the paired hosting benchmark workload."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from eval.dedup.config import TokenizerConfig
from eval.dedup.handoff.corpus import TokenCounter, load_documents_by_ids
from eval.dedup.judging.payload import assert_blind_payload, build_visible_payload
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

from .config import HostingBenchmarkConfig

TRACKS = ("5a", "5b")
QUINTILES = 5


def assign_quintiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign stable within-track length quintiles."""

    result = []
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
    for track in TRACKS:
        ordered = sorted(by_track[track], key=lambda row: (row["stratification_tokens"], row["canonical_pair_id"]))
        require(ordered, "HOSTING_WORKLOAD_EMPTY_TRACK", "hosting workload track is empty", track=track)
        for index, row in enumerate(ordered):
            quintile = min(QUINTILES - 1, index * QUINTILES // len(ordered)) + 1
            result.append({**row, "length_quintile": quintile, "stratum": f"{track}:q{quintile}"})
    return result


def allocate_blocks(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    concurrencies: tuple[int, ...],
    repeats: int,
    pairs_per_block: int,
    warmup: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allocate disjoint, balanced warmup and measured blocks."""

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assign_quintiles(rows):
        strata[row["stratum"]].append(row)
    expected_strata = {f"{track}:q{quintile}" for track in TRACKS for quintile in range(1, QUINTILES + 1)}
    require(set(strata) == expected_strata, "HOSTING_WORKLOAD_STRATA_MISMATCH", "workload strata differ")
    block_count = len(concurrencies) * repeats
    require(
        pairs_per_block % len(expected_strata) == 0 and warmup % len(expected_strata) == 0,
        "HOSTING_WORKLOAD_NOT_BALANCED",
        "block and warmup sizes must divide evenly across strata",
    )
    per_block = pairs_per_block // len(expected_strata)
    per_warmup = warmup // len(expected_strata)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not a security primitive
    for key in sorted(strata):
        rng.shuffle(strata[key])
        require(
            len(strata[key]) >= per_warmup + block_count * per_block,
            "HOSTING_WORKLOAD_STRATUM_TOO_SMALL",
            "a workload stratum cannot satisfy the frozen allocation",
            stratum=key,
            available=len(strata[key]),
            required=per_warmup + block_count * per_block,
        )
    offsets = dict.fromkeys(strata, 0)

    def take(key: str, count: int) -> list[dict[str, Any]]:
        start = offsets[key]
        offsets[key] += count
        return strata[key][start : start + count]

    warmup_rows = [row for key in sorted(strata) for row in take(key, per_warmup)]
    blocks = []
    for concurrency in concurrencies:
        for repeat in range(1, repeats + 1):
            rows_for_block = [row for key in sorted(strata) for row in take(key, per_block)]
            rng.shuffle(rows_for_block)
            blocks.append(
                {
                    "block_id": f"c{concurrency:02d}-r{repeat:02d}",
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "rows": rows_for_block,
                }
            )
    random.Random(seed ^ 0x5A17).shuffle(blocks)  # noqa: S311 - deterministic temporal balancing
    for block_index, block in enumerate(blocks):
        block["endpoint_order"] = ["local", "hub"] if block_index % 2 == 0 else ["hub", "local"]
    require(
        len({row["canonical_pair_id"] for row in warmup_rows}) == len(warmup_rows),
        "HOSTING_WORKLOAD_DUPLICATE",
        "warmup workload contains duplicate pairs",
    )
    measured_ids = [row["canonical_pair_id"] for block in blocks for row in block["rows"]]
    require(
        len(set(measured_ids)) == len(measured_ids)
        and not set(measured_ids) & {row["canonical_pair_id"] for row in warmup_rows},
        "HOSTING_WORKLOAD_DUPLICATE",
        "measured workload is not disjoint",
    )
    return warmup_rows, blocks


def _dependencies() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required to prepare the hosting benchmark"
        raise RuntimeError(msg) from exc
    return pq


def _prompt_assets(config: HostingBenchmarkConfig) -> tuple[str, str, str, dict[str, str]]:
    runner = yaml.safe_load(config.runner_config.read_text(encoding="utf-8"))
    judge = runner["execution"]["stages"][0]["judges"][0]
    root = config.runner_config.parent
    system_path = root / judge["system_prompt_path"]
    pair_path = root / judge["prompt_path"]
    system = system_path.read_text(encoding="utf-8")
    pair = pair_path.read_text(encoding="utf-8")
    rubrics = json.dumps(judge["scores"], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    hashes = {
        "runner_config": sha256_file(config.runner_config),
        "system_prompt": sha256_file(system_path),
        "pair_prompt": sha256_file(pair_path),
        "rubrics": sha256_json(judge["scores"]),
    }
    return system, pair, rubrics, hashes


def _git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not status.strip(), "HOSTING_SOURCE_DIRTY", "commit or remove repository changes before preparing a run")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def provision_model_checkpoint(config: HostingBenchmarkConfig) -> dict[str, Any]:
    """Materialize the exact local checkpoint revision and leave an auditable marker."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        msg = "huggingface_hub is required to provision the frozen local model"
        raise RuntimeError(msg) from exc
    config.model.local_model_path.mkdir(parents=True, exist_ok=True)
    resolved_path = snapshot_download(
        repo_id=config.model.local_model_id,
        revision=config.model.local_model_revision,
        local_dir=config.model.local_model_path,
        cache_dir=config.payload.tokenizer_cache_root,
    )
    marker = {
        "schema_version": "hosting-model-provision-v1",
        "model_id": config.model.local_model_id,
        "revision": config.model.local_model_revision,
        "resolved_path": str(Path(resolved_path).resolve()),
        "provisioned_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(config.model.local_model_path / ".hosting-benchmark-revision.json", marker)
    return marker


def prepare_workload(config: HostingBenchmarkConfig) -> Path:
    """Create a private immutable run root and its paired request blocks."""

    pq = _dependencies()
    candidate_path = config.source_run_root / "data" / "candidate_pairs.parquet"
    provenance_path = config.source_run_root / "data" / "pair_provenance.parquet"
    corpus_path = config.source_run_root / "manifests" / "corpus_manifest.json"
    for path in (candidate_path, provenance_path, corpus_path, config.runner_config):
        require(path.is_file(), "HOSTING_SOURCE_MISSING", "hosting benchmark source is missing", path=str(path))
    git_commit = _git_commit()
    storage_root = config.runs_root
    while not storage_root.exists():
        storage_root = storage_root.parent
    free_bytes = shutil.disk_usage(storage_root).free
    require(
        free_bytes >= 100 * 1024**3,
        "HOSTING_STORAGE_LOW",
        "benchmark filesystem has less than 100 GiB free before checkpoint provisioning",
        free_bytes=free_bytes,
    )
    candidates = pq.read_table(candidate_path).to_pylist()
    provenance = pq.read_table(provenance_path, columns=["canonical_pair_id", "track"]).to_pylist()
    tracks: dict[str, set[str]] = defaultdict(set)
    for row in provenance:
        tracks[str(row["canonical_pair_id"])].add(str(row["track"]))
    eligible = [row for row in candidates if tracks.get(str(row["canonical_pair_id"])) in ({"5a"}, {"5b"})]
    require(eligible, "HOSTING_WORKLOAD_EMPTY", "no single-track candidate pairs are available")
    endpoint_ids = sorted(
        {int(row["presented_doc_a"]) for row in eligible} | {int(row["presented_doc_b"]) for row in eligible}
    )
    corpus_manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = load_documents_by_ids(corpus_manifest, endpoint_ids, columns=("text", "url", "timestamp", "language"))
    tokenizer = TokenCounter(
        TokenizerConfig(
            kind="huggingface",
            model_id=config.payload.tokenizer_model_id,
            revision=config.payload.tokenizer_revision,
            cache_root=config.payload.tokenizer_cache_root,
        )
    )
    payload_config = SimpleNamespace(
        schema_version="dedup-judge-output-v0",
        visible_payload_version="judge-visible-payload-v2",
        max_visible_tokens=config.payload.max_visible_tokens,
        window_tokens=config.payload.window_tokens,
        window_overlap_tokens=config.payload.window_overlap_tokens,
    )
    system_prompt, pair_prompt, rubrics, resource_hashes = _prompt_assets(config)
    template = Environment(
        undefined=StrictUndefined,
        autoescape=False,  # noqa: S701 - this renders an LLM prompt, not HTML
    ).from_string(pair_prompt)
    prepared = []
    for candidate in eligible:
        pair_id = str(candidate["canonical_pair_id"])
        payload, payload_hash = build_visible_payload(
            documents[int(candidate["presented_doc_a"])],
            documents[int(candidate["presented_doc_b"])],
            counter=tokenizer,
            config=payload_config,
        )
        assert_blind_payload(payload)
        rendered_pair = template.render(payload=payload, repair_feedback=None)
        stratification_tokens = tokenizer.count_many([system_prompt, rendered_pair, rubrics])
        prepared.append(
            {
                "canonical_pair_id": pair_id,
                "track": next(iter(tracks[pair_id])),
                "stratification_tokens": sum(stratification_tokens),
                "candidate": {**candidate, "judge_payload_hash": payload_hash},
                "payload": payload,
            }
        )
    warmup, blocks = allocate_blocks(
        prepared,
        seed=config.workload.seed,
        concurrencies=config.workload.concurrencies,
        repeats=config.workload.repeats,
        pairs_per_block=config.workload.pairs_per_block,
        warmup=config.workload.warmup_pairs,
    )
    model_provision = provision_model_checkpoint(config)
    source_digests = {
        "candidate_pairs": sha256_file(candidate_path),
        "pair_provenance": sha256_file(provenance_path),
        "corpus_manifest": sha256_file(corpus_path),
    }
    tokenizer_contract = tokenizer.contract()
    workload_digest = sha256_json(
        {
            "warmup": [[row["canonical_pair_id"], row["candidate"]["judge_payload_hash"]] for row in warmup],
            "blocks": [
                [
                    block["block_id"],
                    [[row["canonical_pair_id"], row["candidate"]["judge_payload_hash"]] for row in block["rows"]],
                ]
                for block in blocks
            ],
        }
    )
    identity_digest = sha256_json(
        {
            "config": config.digest,
            "git_commit": git_commit,
            "judge_resources": resource_hashes,
            "source": source_digests,
            "tokenizer": tokenizer_contract,
            "workload": workload_digest,
        }
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"qwen38-hosting-{timestamp}-{identity_digest[:12]}"
    config.runs_root.mkdir(parents=True, exist_ok=True)
    config.runs_root.chmod(0o700)
    run_root = config.runs_root / run_id
    require(not run_root.exists(), "HOSTING_RUN_EXISTS", "benchmark run root already exists", path=str(run_root))
    blocks_root = run_root / "workload" / "blocks"
    blocks_root.mkdir(parents=True)
    run_root.chmod(0o700)

    def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
        write_text_atomic(
            path,
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n" for row in rows),
        )
        return sha256_file(path)

    warmup_path = run_root / "workload" / "warmup.jsonl"
    warmup_digest = write_rows(warmup_path, warmup)
    block_manifest = []
    for block in blocks:
        path = blocks_root / f"{block['block_id']}.jsonl"
        digest = write_rows(path, block["rows"])
        block_manifest.append(
            {key: block[key] for key in ("block_id", "concurrency", "repeat", "endpoint_order")}
            | {
                "path": str(path.relative_to(run_root)),
                "sha256": digest,
                "pairs": len(block["rows"]),
            }
        )
    manifest = {
        "schema_version": "hosting-benchmark-manifest-v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "identity_digest": identity_digest,
        "workload_digest": workload_digest,
        "source_run_root": str(config.source_run_root),
        "source_digests": source_digests,
        "git_commit": git_commit,
        "config_path": str(config.source_path),
        "config_digest": config.digest,
        "tokenizer": tokenizer_contract,
        "judge_resources": resource_hashes,
        "model": {
            "logical_model": config.model.logical_model,
            "local_model_id": config.model.local_model_id,
            "local_model_revision": config.model.local_model_revision,
            "hub_model": config.model.hub_model,
            "remote_revision_disclosed": False,
            "local_provision": model_provision,
        },
        "workload": {
            "seed": config.workload.seed,
            "warmup_path": str(warmup_path.relative_to(run_root)),
            "warmup_sha256": warmup_digest,
            "warmup_pairs": len(warmup),
            "blocks": block_manifest,
            "measured_pairs_per_endpoint": sum(block["pairs"] for block in block_manifest),
        },
    }
    write_json_atomic(run_root / "benchmark_manifest.json", manifest)
    write_json_atomic(run_root / "benchmark_config.json", json.loads(config.source_path.read_text(encoding="utf-8")))
    return run_root
