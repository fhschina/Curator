# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""All-denominator cause weights from assessment-bound, prediction-aware reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.analysis.judge_calibration import _weight
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def cell_ledger(rows: list[dict], cell: dict, reviews: dict) -> list[dict]:
    originals = {row["review_id"]: row for row in rows}
    predictions = {row["review_id"]: row for row in cell["pairs"]}
    require(
        len(originals) == len(rows)
        and len(predictions) == len(cell["pairs"])
        and originals.keys() == predictions.keys(),
        "CRITIC_CAUSE_MEMBERSHIP",
        "no subset scoring or duplicate rows",
    )
    ledger = []
    for rid, row in originals.items():
        observed = predictions[rid]
        error = (
            "ENGINEERING_MISSING_OUTPUT" if observed["status"] == "ENGINEERING_FAILURE" else observed["draft_error"]
        )
        if error == "CORRECT":
            cause = "NO_OBSERVED_REFERENCE_DISAGREEMENT"
        else:
            require(
                rid in reviews and isinstance(reviews[rid].get("cause"), str) and bool(reviews[rid]["cause"]),
                "CRITIC_CAUSE_REVIEW",
                "every observed mismatch needs an explicit cause, not an inferred default",
            )
            cause = "ENGINEERING_FAILURE" if observed["status"] == "ENGINEERING_FAILURE" else reviews[rid]["cause"]
        ledger.append(
            {
                "review_id": rid,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": _weight(row["draft_label"]),
                "error_type": error,
                "cause": cause,
                "reference": selected.base.previous.reference.primary(row["draft_label"], "human_"),
                "final": observed["primary"],
                "status": observed["status"],
            }
        )
    return ledger


def build(root: Path, source: Path) -> dict:
    root, source = root.resolve(), source.resolve()
    require(not root.exists(), "CRITIC_CAUSE_ROOT", "new cause-audit root required")
    sources = selected.base.previous.reference.verify_freeze(source / "manifest.json")
    assessment = json.loads((source / "assessment.json").read_text())
    review = json.loads((source / "review_complete.json").read_text())
    require(
        review.get("all_observed_disagreements_reviewed") is True
        and review.get("assessment_sha256") == sha256_file(source / "assessment.json"),
        "CRITIC_CAUSE_BINDING",
        "review must bind the complete actual assessment",
    )
    rows = json.loads((source / "panel_private.json").read_text())
    result = {
        "population": len(rows),
        "reference_changed": False,
        "online_model_calls": 0,
        "review_provenance": review["review_provenance"],
        "reference_status": assessment["reference_status"],
        "cells": {},
        "limitation": "Prediction-aware cause attribution, not independent gold, label correction, repair gains or a score with disputed cases removed.",
    }
    for name, cell in assessment["cells"].items():
        ledger = cell_ledger(rows, cell, review["cases"])
        accounting = selected.base.previous.historical.cause_accounting(ledger)
        metrics = cell["scores"]["partial_draft"]["weighted"]
        for metric, contribution in (
            ("duplicate_precision", "precision_loss_contribution_pp"),
            ("duplicate_recall", "recall_loss_contribution_pp"),
        ):
            if metrics[metric] is not None:
                total = sum(c[contribution] or 0 for c in accounting["by_cause"].values())
                require(
                    abs(total - 100 * (1 - metrics[metric])) < 1e-7,
                    "CRITIC_CAUSE_RECONCILIATION",
                    "cause weights must reconcile to the unchanged full score",
                )
        result["cells"][name] = {"scores_unchanged": metrics, "accounting": accounting, "ledger": ledger}
    write_json_atomic(root / "cause_accounting.json", result)
    for path in (Path(__file__).resolve(), source / "assessment.json", source / "review_complete.json"):
        sources[str(path)] = sha256_file(path)
    manifest = {
        "sources": sources,
        "artifacts": {str(root / "cause_accounting.json"): sha256_file(root / "cause_accounting.json")},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return {"population": len(rows), "cells": list(result["cells"]), "contract_digest": manifest["contract_digest"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.source)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
