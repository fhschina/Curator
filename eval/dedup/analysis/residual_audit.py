# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reproducible, reference-frozen residual audits; never a release approval."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, evaluate_predictions
from eval.dedup.validation import require, sha256_file, write_json_atomic, write_text_atomic

PROTECTED_IDS = frozenset({"H0038", "H0347", "H0453", "H0679", "H0748"})
CONTAINMENT_IDS = frozenset({"H0878", "H0333", "H0653", "H0017"})
_STABLE_SETTINGS = (
    "hub_model",
    "logical_model",
    "temperature",
    "top_p",
    "max_output_tokens",
    "max_parallel_requests",
    "max_visible_tokens",
    "window_tokens",
    "window_overlap_tokens",
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_residual(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    operations: dict[str, Any],
    *,
    protected_ids: frozenset[str],
    containment_ids: frozenset[str],
    negative_ids: frozenset[str],
) -> dict[str, Any]:
    """Count abstentions/errors as misses and require every frozen guard ID to be present."""
    by_pair = {row["canonical_pair_id"]: row for row in predictions}
    label_ids = {row["canonical_pair_id"] for row in labels}
    review_ids = {row["review_id"] for row in labels}
    require(
        len(by_pair) == len(predictions) and len(label_ids) == len(review_ids) == len(labels),
        "RESIDUAL_DUPLICATE_ID",
        "residual labels and predictions must have unique pair and review IDs",
    )
    require(set(by_pair) <= label_ids, "RESIDUAL_EXTRA_PREDICTION", "prediction outside the frozen residual")
    require(
        protected_ids | containment_ids | negative_ids <= review_ids,
        "RESIDUAL_GUARD_MISSING",
        "all frozen guard IDs must be represented",
    )
    require(
        int(operations["requested"]) == len(labels) and int(operations["valid"]) == len(predictions),
        "RESIDUAL_ACCOUNTING_MISMATCH",
        "run accounting differs from residual membership or result count",
    )
    joined, rows = {}, []
    for label in labels:
        prediction = by_pair.get(
            label["canonical_pair_id"],
            {
                "canonical_pair_id": label["canonical_pair_id"],
                **dict.fromkeys((*PRIMARY_FIELDS, "relation_type", "material_difference"), "UNRESOLVED"),
                "confidence_tier": "LOW",
            },
        )
        joined[label["review_id"]] = (label, prediction)
        rows.append(prediction)
    metrics = evaluate_predictions(labels, rows)

    def correct(review_id: str) -> bool:
        label, prediction = joined[review_id]
        return all(label[f"human_{field}"] == prediction[field] for field in PRIMARY_FIELDS)

    protected_group = sorted(r for r in protected_ids if joined[r][1]["same_duplicate_group"] == "YES")
    protected_exact = sorted(r for r in protected_ids if correct(r))
    containment_exact = sorted(r for r in containment_ids if correct(r))
    negative_violations = sorted(r for r in negative_ids if joined[r][1]["same_duplicate_group"] != "NO")
    errors = {
        "over_group": sorted(
            r
            for r, (label, p) in joined.items()
            if label["human_same_duplicate_group"] == "NO" and p["same_duplicate_group"] == "YES"
        ),
        "under_group": sorted(
            r
            for r, (label, p) in joined.items()
            if label["human_same_duplicate_group"] == "YES" and p["same_duplicate_group"] != "YES"
        ),
        "direction_only": sorted(
            r
            for r, (label, p) in joined.items()
            if label["human_same_duplicate_group"] == p["same_duplicate_group"] and not correct(r)
        ),
        "unresolved": sorted(r for r, (_, p) in joined.items() if p["same_duplicate_group"] == "UNRESOLVED"),
        "missing_results": sorted(r for r, (label, _) in joined.items() if label["canonical_pair_id"] not in by_pair),
    }
    requested = len(labels)
    require(requested > 0, "RESIDUAL_EMPTY", "the residual must be nonempty")
    checks = {
        "over_group": metrics["over_group"] <= 32,
        "containment_over_group": metrics["containment_over_group"] <= 15,
        "under_group": metrics["under_group"] <= 5,
        "protected_duplicates": len(protected_group) == len(protected_ids),
        "protected_primary_tuples": len(protected_exact) == len(protected_ids),
        "containment_direction_exact": len(containment_exact) == len(containment_ids),
        "diagnostic_negatives": not negative_violations,
        "schema_completion": len(predictions) == requested,
        "terminal_errors": int(operations["errors"]) == 0,
        "retry_rate": int(operations["retried"]) / requested <= 0.01,
    }
    return {
        **metrics,
        "operations": operations,
        "error_ids": errors,
        "protected_group_ids": protected_group,
        "protected_exact_ids": protected_exact,
        "containment_exact_ids": containment_exact,
        "diagnostic_negative_violations": negative_violations,
        "group_outcomes": dict(Counter(p["same_duplicate_group"] for p in rows)),
        "relation_outcomes": dict(Counter(p["relation_type"] for p in rows)),
        "boundary_outcomes": dict(
            Counter(code for p in rows for code in p.get("reason_codes", []) if code.startswith("BOUNDARY_"))
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "primary_by_review_id": {r: {field: p[field] for field in PRIMARY_FIELDS} for r, (_, p) in joined.items()},
    }


def compare_residual_runs(
    *,
    labels_path: Path,
    subset_path: Path,
    negative_review_paths: list[Path],
    baseline_root: Path,
    candidate_root: Path,
    expected_pairs: int = 127,
    gate_profile: str = "v06210",
) -> dict[str, Any]:
    subset = _csv(subset_path)
    pair_ids = {row["canonical_pair_id"] for row in subset}
    require(
        len(pair_ids) == len(subset) == expected_pairs, "RESIDUAL_SUBSET_CHANGED", "unexpected residual membership"
    )
    labels = [row for row in _csv(labels_path) if row["canonical_pair_id"] in pair_ids]
    require(len(labels) == expected_pairs, "RESIDUAL_LABEL_JOIN", "incomplete residual reference labels")
    negative_ids = frozenset(
        row["review_id"] for path in negative_review_paths for row in _csv(path) if row["same_duplicate_group"] == "NO"
    )
    require(len(negative_ids) == 23, "RESIDUAL_NEGATIVE_SET_CHANGED", "expected the original 23 diagnostic negatives")
    manifests = [json.loads((root / "run_manifest.json").read_text()) for root in (baseline_root, candidate_root)]
    baseline_manifest, candidate_manifest = manifests
    require(
        all(
            baseline_manifest[key] == candidate_manifest[key]
            for key in ("source_pair_ids_sha256", "payload_membership_sha256")
        )
        and all(baseline_manifest["settings"][key] == candidate_manifest["settings"][key] for key in _STABLE_SETTINGS),
        "RESIDUAL_EXECUTION_MISMATCH",
        "comparison requires the same model, settings, pairs and visible payloads",
    )
    summaries, provenance = {}, {}
    for name, root, manifest in zip(
        ("baseline", "candidate"), (baseline_root, candidate_root), manifests, strict=True
    ):
        results_path = root / "data" / "judge_results.jsonl"
        complete = json.loads((root / "run_complete.json").read_text())
        require(
            sha256_file(results_path) == complete["results_sha256"],
            "RESIDUAL_RESULT_CHANGED",
            "result digest mismatch",
        )
        predictions = _jsonl(results_path)
        summaries[name] = summarize_residual(
            labels,
            predictions,
            complete,
            protected_ids=PROTECTED_IDS,
            containment_ids=CONTAINMENT_IDS,
            negative_ids=negative_ids,
        )
        provenance[name] = {
            "run_root": str(root.resolve()),
            "prompt_version": manifest["settings"]["prompt_version"],
            "judge_contract_digest": manifest["judge_contract_digest"],
            "manifest_sha256": sha256_file(root / "run_manifest.json"),
            "results_sha256": sha256_file(results_path),
        }
    summary = {
        "schema_version": "dedup-residual-audit-v1",
        **summaries,
        "provenance": provenance,
        "reference_digests": {str(p): sha256_file(p) for p in (labels_path, subset_path, *negative_review_paths)},
        "evaluator_digests": {
            name: sha256_file(Path(__file__).with_name(name)) for name in ("residual_audit.py", "judge_calibration.py")
        },
        "passed": summaries["candidate"]["passed"],
        "interpretation": "Error-enriched development reference, including prior prediction-aware AI adjudications; not an independent human holdout or population estimate.",
        "selection_rule": "A failed residual gate forbids full-development/holdout progression; passing is not release approval.",
    }
    require(gate_profile in {"v06210", "v06211"}, "RESIDUAL_GATE_PROFILE_INVALID", "unknown residual gates")
    if gate_profile == "v06211":
        apply_v06211_gates(summary)
    return summary


def apply_v06211_gates(summary: dict[str, Any]) -> None:
    """Apply the predeclared non-regression gates without changing V0.6.2.10's profile."""
    baseline, candidate = summary["baseline"], summary["candidate"]
    checks = candidate["checks"]
    checks.update(
        {
            "over_group": candidate["over_group"] <= 20,
            "containment_over_group": candidate["containment_over_group"] <= 1,
            "under_group": candidate["under_group"] <= 4,
            "unresolved": len(candidate["error_ids"]["unresolved"]) <= 5,
        }
    )
    for field in ("duplicate_precision", "duplicate_recall", "primary_decision_exact"):
        new, old = candidate["weighted"][field], baseline["weighted"][field]
        checks[f"weighted_{field}_nonregression"] = new is not None and old is not None and new >= old - 1e-12
    candidate["passed"] = summary["passed"] = all(checks.values())
    summary["schema_version"] = "dedup-residual-audit-v2"
    summary["gate_profile"] = "v06211"
    summary["gate_limits"] = {
        "over_group": 20,
        "containment_over_group": 1,
        "under_group": 4,
        "unresolved": 5,
        "weighted_metrics_baseline": {
            field: baseline["weighted"][field]
            for field in ("duplicate_precision", "duplicate_recall", "primary_decision_exact")
        },
    }


def render_residual_report(summary: dict[str, Any]) -> str:
    old, new = summary["baseline"], summary["candidate"]
    lines = [
        f"# {summary['provenance']['candidate']['prompt_version']} residual audit",
        "",
        f"Residual gates: **{'PASS' if summary['passed'] else 'FAIL'}**. {summary['selection_rule']}",
        "",
        summary["interpretation"],
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
    ]
    for weighted in ("unweighted", "weighted"):
        for field in ("duplicate_precision", "duplicate_recall", "primary_decision_exact", "taxonomy_exact"):
            values = [f"{s[weighted][field]:.2%}" if s[weighted][field] is not None else "n/a" for s in (old, new)]
            lines.append(f"| {weighted} {field} | {' | '.join(values)} |")
    for field in ("over_group", "containment_over_group", "under_group"):
        lines.append(f"| {field} | {old[field]} | {new[field]} |")
    lines += [
        "",
        "UNRESOLVED reference duplicates count as recall misses. Historical taxonomy is diagnostic only;",
        "legacy translation MINOR is not a primary-selection criterion. Weighting does not remove residual-selection bias.",
        "",
        "## Frozen gates",
        "",
        "| Gate | Candidate |",
        "|---|---|",
    ]
    lines.extend(f"| {key} | {'pass' if passed else 'FAIL'} |" for key, passed in new["checks"].items())
    lines += [
        "",
        "## Confidence calibration (candidate)",
        "",
        "| Tier | Rows | Primary exact | Weighted primary exact |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {tier} | {row['rows']} | {row['primary_decision_accuracy']:.2%} | {row['weighted_primary_decision_accuracy']:.2%} |"
        for tier, row in new["confidence_tiers"].items()
    )
    lines += [
        "",
        "These are empirical accuracies on this slice, not calibrated probabilities.",
        "",
        "## Residual errors",
        "",
    ]
    lines.extend(f"- {name}: {', '.join(ids) or 'none'}" for name, ids in new["error_ids"].items())
    lines += [
        f"- diagnostic negative violations: {', '.join(new['diagnostic_negative_violations']) or 'none'}",
        "",
        "## Operations and provenance",
        "",
    ]
    for name in ("baseline", "candidate"):
        ops = summary[name]["operations"]
        lines.append(
            f"- {name}: {ops['valid']}/{ops['requested']} valid, {ops['errors']} terminal errors, {ops['retried']} retried pairs; `{summary['provenance'][name]['run_root']}`."
        )
    lines += [
        "",
        "The companion JSON records per-review primary tuples, cohort/relation matrices, boundary outcomes, all gates",
        "and source/result/evaluator digests. No holdout or full-population results are claimed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--negative-review", type=Path, action="append", required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-profile", choices=("v06210", "v06211"), default="v06210")
    args = parser.parse_args()
    summary = compare_residual_runs(
        labels_path=args.labels,
        subset_path=args.subset,
        negative_review_paths=args.negative_review,
        baseline_root=args.baseline_run_root,
        candidate_root=args.candidate_run_root,
        gate_profile=args.gate_profile,
    )
    write_json_atomic(args.output, summary)
    write_text_atomic(args.report, render_residual_report(summary))
    print(json.dumps({"passed": summary["passed"], "checks": summary["candidate"]["checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
