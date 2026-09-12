# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Read-only replay and full-denominator accounting for a completed paced diagnostic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis import paced_development as experiment
from eval.dedup.analysis.development_diagnostic import classify_primary_error, load_run
from eval.dedup.analysis.judge_calibration import _weight, evaluate_development_gates, evaluate_predictions
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.validation import require, sha256_file, write_json_atomic

HISTORICAL = {
    "v0.6.2.9": Path("/raid/hfang/ihb/runs/v0.6.2.9-full-development-diagnostic"),
    "v0.6.2.12": experiment.FULL_ROOT,
}


def operation_audit(root: Path) -> dict:
    called, corrected = set(), set()
    stages = {}
    for stage in ("main", "critic"):
        folder = root / stage
        stage_called, stage_corrected, assistant_count = set(), set(), 0
        attempts = sorted(folder.glob("attempt_*"))
        for number, attempt in enumerate(attempts):
            ids = {r["canonical_pair_id"] for r in experiment.common._jsonl(attempt / "input.jsonl")}
            stage_called.update(ids)
            if number:
                stage_corrected.update(ids)
            for row in experiment.common._read_output_rows(attempt / "output"):
                count = sum(m.get("role") == "assistant" for m in row.get(COLUMN + "__trace", []))
                assistant_count += count
                if count > 1:
                    stage_corrected.add(row["canonical_pair_id"])
        receipts = experiment.transport_receipts(folder)
        successes = receipts["upstream_http_statuses"].get("200", 0)
        stages[stage] = {
            "called_pairs": len(stage_called),
            "model_corrected_pairs": len(stage_corrected),
            "called_pair_model_correction_rate": len(stage_corrected) / len(stage_called) if stage_called else None,
            "saved_assistant_messages": assistant_count,
            "successful_upstream_receipts": successes,
            "unattributed_successful_responses": successes - assistant_count,
            "transport": receipts,
        }
        called.update(stage_called)
        corrected.update(stage_corrected)
    return {
        "stages": stages,
        "unique_called_pairs": len(called),
        "unique_model_corrected_pairs": len(corrected),
        "called_pair_model_correction_rate": len(corrected) / len(called) if called else 0.0,
        "corrected_pair_ids": sorted(corrected),
        "all_successes_have_saved_assistant_count": all(
            s["unattributed_successful_responses"] == 0 for s in stages.values()
        ),
        "interpretation": "Transport retries are separate. Assistant/HTTP count agreement is not request-body hash reconstruction.",
    }


def replay(root: Path, name: str, inputs: list[dict]) -> dict:
    folder = root / name
    main, main_ops = experiment.PacedArm("main").replay_cell(folder, "main", "coverage", inputs, {})
    valid_ids = {r["canonical_pair_id"] for r in main}
    packets, mains = experiment.semantic.critic_inputs(
        [p for p in inputs if p["canonical_pair_id"] in valid_ids], main
    )
    final, final_ops = experiment.PacedArm("critic").replay_cell(folder, "critic", "coverage", packets, mains)
    main_terminal = json.loads((folder / "main/terminal_pair_ids.json").read_text())
    final_terminal = main_terminal + json.loads((folder / "critic/terminal_pair_ids.json").read_text())
    scoring = {
        "main": experiment.accounted_predictions(inputs, main, main_terminal),
        "final": experiment.accounted_predictions(inputs, final, final_terminal),
    }
    for stage, rows in scoring.items():
        require(
            rows == experiment.common._jsonl(folder / f"{stage}_scoring_only.jsonl"),
            "PACED_SCORING_CHANGED",
            "scoring-only outputs must match strict raw replay and terminal IDs",
        )
    return {
        "requested": len(inputs),
        "main_valid": main_ops["valid"],
        "final_valid": final_ops["valid"],
        "terminal_ids": final_terminal,
        "scoring": scoring,
        "operations": operation_audit(folder),
    }


def weighted_error_distribution(comparison: dict) -> dict:
    weights = Counter()
    by_reason = {}
    for row in comparison["errors"]:
        weights[row["error"]] += row["weight"]
        reason = str(row["reason"])
        by_reason.setdefault(reason, Counter())[row["error"]] += row["weight"]
    return {"by_error": dict(weights), "by_reference_reason": {k: dict(v) for k, v in sorted(by_reason.items())}}


def error_axes(labels: list[dict], predictions: list[dict]) -> dict:
    references = labels_by_id(labels)
    actual = experiment.common._index(predictions, "predictions")
    require(references.keys() == actual.keys(), "PACED_ERROR_MEMBERSHIP", "complete identical memberships required")
    axes = {name: {} for name in ("reference_reason", "predicted_relation", "predicted_basis", "predicted_overlap")}
    for pid, prediction in actual.items():
        label = references[pid]
        error = classify_primary_error(label, prediction)
        if error == "CORRECT":
            continue
        values = {
            "reference_reason": label.get("human_reason_code"),
            "predicted_relation": prediction.get("relation_type"),
            "predicted_basis": prediction.get("coverage_response", {}).get("shared_basis"),
            "predicted_overlap": prediction.get("dominant_overlap_source"),
        }
        for axis, value in values.items():
            bucket = axes[axis].setdefault(str(value), {}).setdefault(error, {"count": 0, "weight": 0.0})
            bucket["count"] += 1
            bucket["weight"] += _weight(label)
    return axes


def audit(root: Path) -> dict:
    m = experiment.validate(root)
    require(
        (root / "assessment.json").is_file() and not (root / "stopped.json").exists(),
        "PACED_AUDIT_INCOMPLETE",
        "completed non-fatal schedule required for a full-development score",
    )
    inputs = experiment.common._jsonl(root / "input_full.jsonl")
    labels = json.loads((root / "labels_full_private.json").read_text())
    require(len(inputs) == len(labels) == 1000, "PACED_AUDIT_MEMBERSHIP", "all original 1000 pairs required")
    full = replay(root, "full", inputs)
    comparison = experiment.compare(root / "full", labels)
    predictions = full.pop("scoring")
    require(
        all(
            not r.get("metric_only_missing_output")
            or labels_by_id(labels)[r["canonical_pair_id"]]["human_same_duplicate_group"] != "UNRESOLVED"
            for r in predictions["final"]
        ),
        "PACED_MISSING_REFERENCE_ABSTENTION",
        "missing output must never gain primary credit from an unresolved reference",
    )
    baseline, input_index = {}, experiment.common._index(inputs, "inputs")
    for version, path in HISTORICAL.items():
        _, _, old_predictions, payloads = load_run(path)
        old_input = experiment.common._index(payloads, version)
        require(
            old_input.keys() == input_index.keys()
            and all(p["judge_payload_hash"] == input_index[pid]["judge_payload_hash"] for pid, p in old_input.items()),
            "PACED_HISTORICAL_PAYLOAD",
            "same original pair membership and payload required",
        )
        baseline[version] = {
            "metrics": evaluate_predictions(labels, old_predictions),
            "run_root": str(path),
            "interpretation": "Historical same-input comparison, not a simultaneous controlled experiment.",
        }
    gates = evaluate_development_gates(
        candidate=comparison["final_metrics"],
        baseline=baseline["v0.6.2.9"]["metrics"],
        schema_completion_rate=full["final_valid"] / 1000,
        retry_rate=full["operations"]["called_pair_model_correction_rate"],
    )
    gates["checks"]["zero_terminal_errors"] = not full["terminal_ids"]
    gates["checks"]["no_unattributed_successful_responses"] = full["operations"][
        "all_successes_have_saved_assistant_count"
    ]
    gates["passed"] = all(gates["checks"].values())
    return {
        "schema_version": "dedup-paced-result-audit-v1",
        "semantic_version": m["version"],
        "execution_contract_digest": m["contract_digest"],
        "full_development": full,
        "comparison": comparison,
        "weighted_errors": weighted_error_distribution(comparison),
        "error_axes": {stage: error_axes(labels, rows) for stage, rows in predictions.items()},
        "historical": baseline,
        "development_gates": gates,
        "eligible_for_release": False,
        "reference_changed": False,
        "reference_status": m["reference_status"],
        "all_schedule_transport": experiment.transport_receipts(root),
        "audit_source_sha256": sha256_file(Path(__file__)),
        "artifacts": {str(p): sha256_file(p) for p in sorted(root.rglob("*.json*")) if p.is_file()},
    }


def labels_by_id(labels: list[dict]) -> dict:
    return experiment.common._index(labels, "labels")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "weighted": result["comparison"]["final_metrics"]["weighted"],
                "gates": result["development_gates"],
                "eligible_for_release": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
