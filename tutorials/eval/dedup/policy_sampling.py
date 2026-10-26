# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample fresh pairs from the existing dense manifest and completed fuzzy outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from policy_cases import digest, discussion
from policy_eval import full_population, read_json, write_json, write_rows


def fetch_documents(root: Path, ids: set[int]) -> tuple[dict[int, str], dict[str, str]]:
    """Resolve global IDs as manifest start_id + physical row index, without renumbering."""
    manifest = read_json(root / "manifest/dense_10m_explicit.json")
    shards = manifest["shards"]
    starts = np.array([shard["start_id"] for shard in shards])
    wanted = defaultdict(list)
    for key in sorted(ids):
        index = int(np.searchsorted(starts, key, side="right") - 1)
        if index < 0 or key > shards[index]["end_id"]:
            msg = f"Document outside dense manifest: {key}"
            raise ValueError(msg)
        wanted[index].append(key)
    documents, hashes = {}, {}
    for index, keys in wanted.items():
        shard = shards[index]
        path = root / "extracted_text/objects" / shard["source_key"] / "text.parquet"
        checksum = digest(path.read_bytes())
        if checksum != shard["sha256"]:
            msg = f"Corpus shard hash mismatch: {path}"
            raise ValueError(msg)
        hashes[str(path)] = checksum
        table = pq.read_table(path, columns=["text"])
        if table.num_rows != shard["rows"]:
            msg = f"Corpus shard row count mismatch: {path}"
            raise ValueError(msg)
        for key in keys:
            documents[key] = table["text"][key - shard["start_id"]].as_py()
    return documents, hashes


def additional_exposures(sources: list[Path]) -> tuple[list[dict], dict[str, str]]:
    rows, hashes = [], {}
    for source in sources:
        paths = sorted((source / "inputs").glob("*.json"))
        if not paths:
            msg = f"No historical input snapshots found in {source}"
            raise ValueError(msg)
        for path in paths:
            content = path.read_bytes()
            hashes[str(path)] = digest(content)
            payload = json.loads(content)["payload"]
            rows.append(
                {
                    "pair_id": path.stem,
                    "text_a": payload["document_a"].get("text"),
                    "text_b": payload["document_b"].get("text"),
                }
            )
    return rows, hashes


def sample(args: argparse.Namespace) -> dict:
    root = args.corpus_root
    labels = pq.read_table(root / "fuzzy/duplicate_groups.parquet").to_pandas()
    removed = pq.read_table(root / "fuzzy/removal_ids.parquet")["_curator_dedup_id"].to_numpy()
    ids = labels["_curator_dedup_id"].to_numpy()
    groups = labels["_duplicate_group_id"].to_numpy()
    order = np.argsort(ids)
    sorted_ids, sorted_groups = ids[order], groups[order]

    def group(key: int) -> int | None:
        index = int(np.searchsorted(sorted_ids, key))
        return int(sorted_groups[index]) if index < len(sorted_ids) and sorted_ids[index] == key else None

    old, metadata = full_population(args.full_source, args.provenance)
    discussed, votes = discussion(read_json(args.discussion_snapshot), args.runs_root)
    excluded = [
        {**row, **meta, "group_id_a": group(int(meta["doc_id_a"])), "group_id_b": group(int(meta["doc_id_b"]))}
        for row, meta in zip(old, metadata, strict=True)
    ]
    excluded.extend({**row, **vote} for row, vote in zip(discussed, votes, strict=True))
    additional, exposure_hashes = additional_exposures(args.additional_exposure_run)
    excluded.extend(additional)
    used_docs = {int(row[key]) for row in excluded for key in ("doc_id_a", "doc_id_b") if row.get(key) is not None}
    used_groups = {row[key] for row in excluded for key in ("group_id_a", "group_id_b") if row.get(key) is not None}
    is_removed = np.isin(ids, removed)
    keepers = dict(zip(groups[~is_removed].tolist(), ids[~is_removed].tolist(), strict=True))
    rng = np.random.default_rng(42)
    candidates, chosen_groups = [], set()
    for index in rng.permutation(len(removed)):
        b = int(removed[index])
        g = group(b)
        if g is None or g in used_groups or g in chosen_groups or g not in keepers:
            continue
        a = int(keepers[g])
        if a in used_docs or b in used_docs:
            continue
        candidates.append(
            {
                "pair_id": f"fresh-removal-{a}-{b}",
                "pair_type": "keeper_removed",
                "doc_id_a": str(a),
                "doc_id_b": str(b),
                "group_id_a": g,
                "group_id_b": g,
                "keeper_direction": "a_can_replace_b",
            }
        )
        chosen_groups.add(g)
        if len(candidates) == args.pool_per_type:
            break
    total = read_json(root / "manifest/dense_10m_explicit.json")["total_rows"]
    for _ in range(args.pool_per_type):
        while True:
            a, b = (int(value) for value in rng.integers(0, total, size=2))
            ga, gb = group(a), group(b)
            if a != b and not {a, b} & used_docs and not {ga, gb} & used_groups and (ga is None or ga != gb):
                break
        candidates.append(
            {
                "pair_id": f"fresh-cross-{a}-{b}",
                "pair_type": "cross_group_sample",
                "doc_id_a": str(a),
                "doc_id_b": str(b),
                "group_id_a": ga,
                "group_id_b": gb,
            }
        )
    documents, hashes = fetch_documents(
        root, {int(row[key]) for row in candidates for key in ("doc_id_a", "doc_id_b")}
    )
    for row in candidates:
        row.update(text_a=documents[int(row["doc_id_a"])], text_b=documents[int(row["doc_id_b"])])
    args.output.mkdir(parents=True, exist_ok=False)
    write_rows(args.output / "candidates.jsonl", candidates)
    write_rows(args.output / "exclusions.jsonl", excluded)
    receipt = {
        "seed": 42,
        "candidate_pairs": len(candidates),
        "excluded_pairs": len(excluded),
        "exclusion_scope": "All frozen 10M 20K pairs (including the historical 1000 development cases) and 150 discussion cases",
        "historical_candidate_scope": "Existing Deciffusion studies select from the same excluded 10M 20K population",
        "additional_exposure_inputs": exposure_hashes,
        "fuzzy_groups_sha256": digest((root / "fuzzy/duplicate_groups.parquet").read_bytes()),
        "removal_ids_sha256": digest((root / "fuzzy/removal_ids.parquet").read_bytes()),
        "shards": hashes,
    }
    write_json(args.output / "sampling_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus-root", "full-source", "provenance", "discussion-snapshot", "runs-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pool-per-type", type=int, default=200)
    parser.add_argument("--additional-exposure-run", type=Path, action="append", default=[])
    result = sample(parser.parse_args())
    print({key: value for key, value in result.items() if key != "shards"})


if __name__ == "__main__":
    main()
