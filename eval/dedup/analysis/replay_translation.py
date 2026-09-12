# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Isolated V0.6.2.12 translation-route replay; never a live candidate score or cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.dedup.analysis.development_diagnostic import (
    _index,
    classify_primary_error,
    evaluate_primary_guards,
    load_run,
    matched_raw_outputs,
    replay_v0629_arbitration,
)
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, evaluate_predictions
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, write_json_atomic


def replay_run(root: Path, labels_path: Path) -> dict[str, Any]:
    manifest, _, original, payloads = load_run(root)
    require(
        manifest["settings"]["prompt_version"] == "dedup-judge-hs-v0.6.2.9"
        and manifest["pair_selection"]["sha256"] == sha256_file(labels_path),
        "TRANSLATION_REPLAY_SOURCE",
        "requires frozen V0.6.2.9 outputs and their unchanged reference",
    )
    labels = _read_csv(labels_path)
    references, by_payload = _index(labels, "labels"), _index(payloads, "payloads")
    require(set(references) == set(by_payload), "TRANSLATION_REPLAY_MEMBERSHIP", "reference membership differs")
    attribution = replay_v0629_arbitration(root, labels, original, payloads)
    raw, raw_digests = matched_raw_outputs(root, original)
    predictions, changes = [], []
    for row in original:
        pair_id = row["canonical_pair_id"]
        main, critic = raw[pair_id]
        payload = by_payload[pair_id]["payload"]
        result = adapt_ndd_judge_output(
            main,
            "dedup-judge-output-v3",
            payload=payload,
            record_binding_critic=critic,
            record_binding_policy="v6-route",
        )
        validate_evidence_offsets(result, payload)
        predictions.append({"canonical_pair_id": pair_id, **result})
        changed_fields = [key for key, value in result.items() if row[key] != value]
        if changed_fields:
            label = references[pair_id]
            changes.append(
                {
                    "review_id": label["review_id"],
                    "canonical_pair_id": pair_id,
                    "changed_fields": changed_fields,
                    "original_error": classify_primary_error(label, row),
                    "replay_error": classify_primary_error(label, result),
                    "original": {key: row[key] for key in PRIMARY_FIELDS},
                    "replay": {key: result[key] for key in PRIMARY_FIELDS},
                }
            )
    by_prediction = _index(predictions, "replay")
    guards = [
        {
            **r,
            "preserved": all(by_prediction[r["canonical_pair_id"]][key] == r["final"][key] for key in PRIMARY_FIELDS),
        }
        for r in attribution["changed_primary_rows"]
        if r["effect"] == "MAIN_WRONG_CRITIC_REPAIRED"
    ]
    return {
        "schema_version": "dedup-v06212-translation-route-replay-v1",
        "eligible_for_promotion": False,
        "interpretation": "Only the deterministic translation route changed. All main and critic responses are saved V0.6.2.9 responses; the revised V0.6.2.12 prompts and retained-conflict proof were NOT evaluated. No labels changed, no new model calls, no cache migration.",
        "source_run_root": str(root.resolve()),
        "source_digests": {
            str(path): sha256_file(path)
            for path in [
                labels_path,
                root / "run_manifest.json",
                root / "run_complete.json",
                root / "data/judge_results.jsonl",
                root / "data/judge_payloads.jsonl",
            ]
        },
        "raw_source_digests": raw_digests,
        "implementation_digests": {
            str(path): sha256_file(path)
            for path in [
                Path(__file__),
                Path(__file__).parents[1] / "judging/local_ndd.py",
                Path(__file__).with_name("development_diagnostic.py"),
            ]
        },
        "historical_public_replay_exact": len(original),
        "original": evaluate_predictions(labels, original),
        "replay": evaluate_predictions(labels, predictions),
        "changed_rows": changes,
        "critic_repaired_negative_guards": guards,
        "all_critic_repaired_negatives_preserved": all(row["preserved"] for row in guards),
        "guard_checks": evaluate_primary_guards(guards, predictions) if guards else {"expected": 0, "passed": False},
        "replay_predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.resolve() != args.predictions.resolve()
        and not any(p.exists() for p in (args.output, args.predictions)),
        "TRANSLATION_REPLAY_OUTPUT_EXISTS",
        "use distinct new offline artifact paths",
    )
    summary = replay_run(args.source_run_root, args.labels)
    write_json_atomic(
        args.predictions, {"eligible_for_promotion": False, "predictions": summary.pop("replay_predictions")}
    )
    summary["predictions_artifact"] = {"path": str(args.predictions), "sha256": sha256_file(args.predictions)}
    write_json_atomic(args.output, summary)
    print(summary["replay"]["weighted"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
