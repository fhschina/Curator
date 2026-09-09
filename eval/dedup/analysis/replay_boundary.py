# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline scoped-adapter replay with source checks; never a judge cache or promotion result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from eval.dedup.analysis.judge_calibration import evaluate_predictions
from eval.dedup.contracts import canonical_json_bytes
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.validation import require, sha256_file, write_json_atomic


def replay_run(source_root: Path, labels_path: Path) -> dict[str, Any]:
    results_path = source_root / "data" / "judge_results.jsonl"
    payloads_path = source_root / "data" / "judge_payloads.jsonl"
    complete = json.loads((source_root / "run_complete.json").read_text())
    manifest = json.loads((source_root / "run_manifest.json").read_text())
    require(
        sha256_file(results_path) == complete["results_sha256"],
        "REPLAY_RESULT_CHANGED",
        "source result digest mismatch",
    )
    require(
        sha256_file(payloads_path) == manifest["judge_payloads_sha256"],
        "REPLAY_PAYLOAD_CHANGED",
        "source payload digest mismatch",
    )
    original = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    payloads = {
        row["canonical_pair_id"]: row["payload"]
        for line in payloads_path.read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }
    raw_paths = sorted(source_root.glob("runtime/**/attempt_*/output/*.jsonl"))
    raw = [json.loads(line) for path in raw_paths for line in path.read_text().splitlines() if line.strip()]
    by_pair = {row["canonical_pair_id"]: row for row in raw}
    require(
        len(by_pair) == len(raw) == len(original),
        "REPLAY_RAW_MEMBERSHIP",
        "raw attempts must map one-to-one to source results",
    )
    predictions = []
    for row in original:
        pair_id = row["canonical_pair_id"]
        require(pair_id in by_pair and pair_id in payloads, "REPLAY_PAIR_MISSING", "missing source pair")
        saved = by_pair[pair_id]
        main, critic = saved["qwen_dedup_semantic_judge"], saved["qwen_dedup_record_binding_critic"]
        digest = hashlib.sha256(
            canonical_json_bytes({"semantic_judge": main, "record_binding_critic": critic})
        ).hexdigest()
        require(
            digest == row["provider_response_sha256"],
            "REPLAY_PROVIDER_CHANGED",
            "raw response does not match source result",
        )
        prediction = adapt_ndd_judge_output(
            main,
            "dedup-judge-output-v3",
            payload=payloads[pair_id],
            record_binding_critic=critic,
            record_binding_policy="v5-replay",
        )
        predictions.append({"canonical_pair_id": pair_id, **prediction})
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        labels = [row for row in csv.DictReader(handle) if row["canonical_pair_id"] in by_pair]
    require(len(labels) == len(original), "REPLAY_LABEL_JOIN", "incomplete reference labels")
    judging = Path(__file__).resolve().parents[1] / "judging"
    return {
        "schema_version": "dedup-v06211-offline-boundary-replay-v1",
        "eligible_for_promotion": False,
        "source_run_root": str(source_root.resolve()),
        "source_prompt_version": manifest["settings"]["prompt_version"],
        "source_digests": {str(p): sha256_file(p) for p in [results_path, payloads_path, labels_path, *raw_paths]},
        "implementation_digests": {
            str(p): sha256_file(p) for p in [Path(__file__), judging / "local_ndd.py", judging / "scoped_critic.py"]
        },
        "interpretation": "Counterfactual adapter replay of recorded V0.6.2.10 scores, with no new LLM calls or cache reuse; not a live V0.6.2.11 result.",
        "original": evaluate_predictions(labels, original),
        "replay": evaluate_predictions(labels, predictions),
        "replay_results": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "REPLAY_OUTPUT_EXISTS", "do not overwrite a frozen replay artifact")
    summary = replay_run(args.source_run_root, args.labels)
    write_json_atomic(args.output, summary)
    print(
        json.dumps(
            {
                name: {
                    key: summary[name][key]
                    for key in ("weighted", "over_group", "containment_over_group", "under_group")
                }
                for name in ("original", "replay")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
