# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Generate the static Pair Explorer for a Judge v0.7 result bundle."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from eval.dedup.core.validation import require, write_text_atomic
from eval.dedup.reporting import explorer

VERSION = "v0.7"
PAIR_EXPLORER_NAME = "pair_explorer.html"


def _group_size_bucket(size: int) -> str:
    if size == 1:
        return "singleton"
    if size == 2:
        return "size_2"
    if size <= 5:
        return "size_3_5"
    if size <= 20:
        return "size_6_20"
    return "size_21_plus"


def _ratio_bucket(ratio: float) -> str:
    if ratio < 0.25:
        return "0-0.25"
    if ratio < 0.5:
        return "0.25-0.5"
    if ratio < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def _comparison_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["report_group_size_bucket"] = _group_size_bucket(int(row["predicted_group_size_low"]))
        row["report_ratio_bucket"] = _ratio_bucket(float(row["token_length_ratio"]))
    return rows


def render_pair_explorer(records: list[dict[str, Any]], contexts: dict[str, dict[str, Any]]) -> str:
    """Render v0.7 data with the repository's native Pair Explorer UI."""

    output = explorer.pair_explorer_html(
        evaluation_run_id=VERSION,
        records=records,
        group_contexts=contexts,
    )
    replacements = {
        "<title>Dedup Pair Explorer</title>": "<title>Internal Judge v0.7 Pair Explorer</title>",
        "<h1>Dedup Pair Explorer</h1>": "<h1>Internal Judge v0.7 Pair Explorer</h1>",
        "search=[r.pair_id,": "search=[r.review_id,r.pair_id,",
    }
    for original, replacement in replacements.items():
        require(output.count(original) == 1, "V07_EXPLORER_TEMPLATE", "known native template integration point")
        output = output.replace(original, replacement, 1)
    return output


def _link(source: Path, destination: Path) -> None:
    require(source.is_file(), "V07_EXPLORER_SOURCE", "required source artifact", path=str(source))
    destination.symlink_to(source.resolve())


def build(*, release_root: Path, context_root: Path, destination: Path) -> dict[str, Any]:
    """Join v0.7 verdicts with the unchanged SUT and document context."""

    with tempfile.TemporaryDirectory(prefix="v07-pair-explorer-") as temporary:
        stage = Path(temporary)
        for folder in ("data", "logs", "manifests"):
            (stage / folder).mkdir()
        for relative, source in (
            ("data/judge_results.jsonl", release_root / "data/judge_results.jsonl"),
            ("logs/judge_errors.jsonl", release_root / "logs/judge_errors.jsonl"),
            ("data/judge_payloads.jsonl", context_root / "data/judge_payloads.jsonl"),
            ("data/pair_provenance.parquet", context_root / "data/pair_provenance.parquet"),
            ("data/document_outcomes.parquet", context_root / "data/document_outcomes.parquet"),
            ("manifests/sut_run_manifest.json", context_root / "manifests/sut_run_manifest.json"),
        ):
            _link(source, stage / relative)

        rows = _comparison_rows(release_root / "data/pair_comparisons.parquet")
        records = explorer.build_pair_explorer_records(stage, rows)
        index = {
            row["canonical_pair_id"]: row
            for row in json.loads((release_root / "reports/pair_index.json").read_text(encoding="utf-8"))
        }
        require(
            len(records) == len(index) == 20_000 and {row["pair_id"] for row in records} == set(index),
            "V07_EXPLORER_POPULATION",
            "exact 20K Pair Explorer join",
        )
        for record in records:
            record["review_id"] = index[record["pair_id"]]["review_id"]
        contexts = explorer.attach_group_context(stage, records)

    rendered = render_pair_explorer(records, contexts)
    write_text_atomic(destination, rendered)
    return {
        "version": VERSION,
        "pairs": len(records),
        "groups": len(contexts),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build(
                release_root=args.release_root.resolve(),
                context_root=args.context_root.resolve(),
                destination=args.destination.resolve(),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
