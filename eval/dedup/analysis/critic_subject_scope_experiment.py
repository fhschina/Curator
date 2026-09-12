# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exact offline action-scope replay; the model responses and all labels stay fixed."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_subject_experiment as trial
from eval.dedup.judging import critic_subject_scope as scope
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def replay(root: Path, source: Path) -> dict:
    root, source = root.resolve(), source.resolve()
    require(not root.exists(), "SUBJECT_SCOPE_ROOT", "new replay root required")
    sources = trial.selected.base.previous.reference.verify_freeze(source / "manifest.json")
    prior = json.loads((source / "assessment.json").read_text())
    old = json.loads((source / "manifest.json").read_text())
    if old.get("stage") == "full_specialist_component":
        require(
            old.get("declared_final_action_contract") == scope.CONTRACT,
            "SUBJECT_SCOPE_DECLARATION",
            "full pipeline must freeze final scope before any component results",
        )
    else:
        review = json.loads((source / "review_complete.json").read_text())
        errors = {p["review_id"] for c in prior["cells"].values() for p in c["pairs"] if p["requires_cause_review"]}
        require(
            review.get("assessment_sha256") == sha256_file(source / "assessment.json")
            and review.get("all_observed_disagreements_reviewed") is True
            and errors <= set(review.get("cases", {}))
            and review.get("next_stage") == "OFFLINE_SUBJECT_ACTION_SCOPE_ONLY",
            "SUBJECT_SCOPE_REVIEW",
            "review all previous errors before narrowing authority",
        )
        sources[str(source / "review_complete.json")] = sha256_file(source / "review_complete.json")
    rows = json.loads((source / "panel_private.json").read_text())
    result = deepcopy(prior)
    result.update(
        stage="full_pipeline_final" if len(rows) == 1000 else "subject_scope_offline96",
        online_model_calls=0,
        action_contract=scope.CONTRACT,
        scope_changed_only=True,
    )
    receipts = {}
    for path in (source / "responses").glob("*.json"):
        value = json.loads(path.read_text())
        receipts[(value["repeat"], value["canonical_pair_id"])] = value
        sources[str(path)] = sha256_file(path)
    for repeat in range(1, old["repeat_count"] + 1):
        cell = result["cells"][f"{repeat}/candidate"]
        predictions = []
        for row, observed in zip(rows, cell["pairs"], strict=True):
            require(
                row["review_id"] == observed["review_id"],
                "SUBJECT_SCOPE_MEMBERSHIP",
                "all original rows and original order required",
            )
            public = row["coverage_base"][str(repeat)]
            receipt = receipts.get((repeat, row["canonical_pair_id"]))
            if observed["status"] == "ENGINEERING_FAILURE":
                public = trial.subject.veto.unresolved_judge_output_v3()
            elif scope.route(public, row["payload"]) == "REVIEW_BILATERAL_SUBJECTS":
                require(
                    receipt is not None and receipt["status"] == "VALID",
                    "SUBJECT_SCOPE_RECEIPT",
                    "valid original specialist output required",
                )
                raw = receipt["raw_response"]["choices"][0]
                require(
                    raw["finish_reason"] == "stop" and raw["message"]["content"] == receipt["assistant_content"],
                    "SUBJECT_SCOPE_RAW",
                    "complete original response binding",
                )
                value = trial.selected.base.previous.strict_json(receipt["assistant_content"])
                original, _ = trial.subject.apply_review(public, row["payload"], value)
                require(
                    original == receipt["public"],
                    "SUBJECT_SCOPE_PRIOR_REPLAY",
                    "validate original adapter before changing authorized action",
                )
                public, observed["rule"] = scope.apply_review(public, row["payload"], value)
            observed["primary"] = trial.selected.base.previous.reference.primary(public)
            observed["draft_error"] = trial.selected.base.previous.reference.classify_primary_error(
                row["draft_label"], public
            )
            observed["requires_cause_review"] = (
                observed["draft_error"] != "CORRECT" or observed["status"] == "ENGINEERING_FAILURE"
            )
            predictions.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    **public,
                    **({"metric_only_missing_output": True} if observed["status"] == "ENGINEERING_FAILURE" else {}),
                }
            )
        cell["scores"] = {
            name: trial.selected.base.previous.reference.score([r[field] for r in rows], predictions)
            for name, field in (("historical", "historical_label"), ("partial_draft", "draft_label"))
        }
    result["candidate_repeat_disagreements"] = (
        [
            a["review_id"]
            for a, b in zip(
                result["cells"]["1/candidate"]["pairs"], result["cells"]["2/candidate"]["pairs"], strict=True
            )
            if a["primary"] != b["primary"] or a["status"] != b["status"]
        ]
        if old["repeat_count"] == 2
        else []
    )
    result["new_errors_from_previously_correct"] = {
        str(rep): [
            a["review_id"]
            for a, b in zip(
                result["cells"][f"{rep}/saved_coverage"]["pairs"],
                result["cells"][f"{rep}/candidate"]["pairs"],
                strict=True,
            )
            if a["draft_error"] == "CORRECT" and b["draft_error"] != "CORRECT"
        ]
        for rep in range(1, old["repeat_count"] + 1)
    }
    guards = (*trial.paired.TARGETS, *trial.paired.GUARDS, "H0017")
    result["clear_guard_failures"] = {
        str(rep): [
            p["review_id"]
            for p in result["cells"][f"{rep}/candidate"]["pairs"]
            if p["review_id"] in guards and p["draft_error"] != "CORRECT"
        ]
        for rep in range(1, old["repeat_count"] + 1)
    }
    for path in (
        Path(__file__).resolve(),
        Path(scope.__file__).resolve(),
        Path(__file__).with_name("critic_subject_scope_v2.md"),
        source / "assessment.json",
        source / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "assessment.json", result)
    manifest = {
        "contract_version": scope.CONTRACT,
        "stage": result["stage"],
        "population": len(rows),
        "repeat_count": old["repeat_count"],
        "online_model_calls": 0,
        "source_run": str(source),
        "sources": sources,
        "artifacts": {
            str(root / name): sha256_file(root / name) for name in ("panel_private.json", "assessment.json")
        },
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "contract_digest": manifest["contract_digest"],
            "external_attempts": 0,
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.root, args.source)
    print(json.dumps({k: v for k, v in result.items() if k not in ("cells", "response_artifacts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
