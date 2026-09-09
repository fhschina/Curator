# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Human-calibrated metrics and release gates for versioned dedup judges."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from eval.dedup.validation import require, write_json_atomic

PRIMARY_FIELDS: Final = ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
TAXONOMY_FIELDS: Final = ("relation_type", "material_difference")
SEMANTIC_LEDGER_REASON_PREFIXES: Final = (
    "CONTENT_PROFILE_A",
    "CONTENT_PROFILE_B",
    "SHARED_CONTENT_BASIS",
    "HARD_CONFLICT",
    "DIFFERENCE_LOCATION",
    "RECORD_ALIGNMENT",
    "NON_MAIN_DIFFERENCE",
    "TRANSLATION_STATUS",
    "RECORD_IDENTITY_SUPPORT",
    "OVERLAP_SCOPE",
    "SURFACE_DELTA_TYPE",
    "BOUNDARY_DELTA_CLASS",
    "SEMANTIC_LEDGER_RULE",
)


@dataclass(frozen=True, slots=True)
class DevelopmentGateThresholds:
    weighted_precision: float = 0.75
    weighted_recall: float = 0.75
    weighted_primary_exact: float = 0.79
    max_over_group: int = 66
    max_cohort_regression: float = 0.03
    schema_completion: float = 1.0
    max_retry_rate: float = 0.01


@dataclass(frozen=True, slots=True)
class HoldoutGateThresholds:
    representative_noninferiority: float = -0.03
    precision: float = 0.75
    recall: float = 0.75
    difficult_over_group_reduction: float = 0.30
    max_containment_miss_growth: float = 0.10


DEFAULT_DEVELOPMENT_THRESHOLDS: Final = DevelopmentGateThresholds()
DEFAULT_HOLDOUT_THRESHOLDS: Final = HoldoutGateThresholds()


def prefixed_predictions(rows: Iterable[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Project prediction fields embedded in a human-label CSV."""

    fields = (*PRIMARY_FIELDS, *TAXONOMY_FIELDS)
    return [
        {
            "canonical_pair_id": row["canonical_pair_id"],
            **{field: row.get(f"{prefix}_{field}") for field in fields},
        }
        for row in rows
    ]


def _weight(row: dict[str, Any]) -> float:
    population = row.get("stratum_population_n")
    sample = row.get("stratum_sample_n")
    if population not in {None, ""} and sample not in {None, ""}:
        sample_value = float(sample)
        require(sample_value > 0, "CALIBRATION_WEIGHT_INVALID", "stratum sample size must be positive")
        return float(population) / sample_value
    value = row.get("sample_weight", 1.0)
    return float(value) if value not in {None, ""} else 1.0


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _exact(label: dict[str, Any], prediction: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(str(label.get(f"human_{field}")) == str(prediction.get(field)) for field in fields)


def _reason_code_value(prediction: dict[str, Any], prefix: str) -> str | None:
    marker = f"{prefix}:"
    values = [
        reason[len(marker) :]
        for reason in prediction.get("reason_codes", [])
        if isinstance(reason, str) and reason.startswith(marker)
    ]
    require(
        len(values) <= 1,
        "CALIBRATION_LEDGER_DUPLICATE",
        "a prediction cannot contain duplicate semantic-ledger reason codes",
        prefix=prefix,
    )
    return values[0] if values else None


def _metric_block(rows: list[tuple[dict[str, Any], dict[str, Any]]], *, weighted: bool) -> dict[str, Any]:
    def row_weight(label: dict[str, Any]) -> float:
        return _weight(label) if weighted else 1.0

    total = sum(row_weight(label) for label, _ in rows)
    primary_exact = sum(row_weight(label) for label, prediction in rows if _exact(label, prediction, PRIMARY_FIELDS))
    taxonomy_exact = sum(row_weight(label) for label, prediction in rows if _exact(label, prediction, TAXONOMY_FIELDS))
    true_positive = sum(
        row_weight(label)
        for label, prediction in rows
        if label.get("human_same_duplicate_group") == "YES" and prediction.get("same_duplicate_group") == "YES"
    )
    false_positive = sum(
        row_weight(label)
        for label, prediction in rows
        if label.get("human_same_duplicate_group") == "NO" and prediction.get("same_duplicate_group") == "YES"
    )
    false_negative = sum(
        row_weight(label)
        for label, prediction in rows
        if label.get("human_same_duplicate_group") == "YES" and prediction.get("same_duplicate_group") != "YES"
    )
    true_negative = sum(
        row_weight(label)
        for label, prediction in rows
        if label.get("human_same_duplicate_group") == "NO" and prediction.get("same_duplicate_group") == "NO"
    )
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {
        "rows": len(rows),
        "weight_total": total,
        "primary_decision_exact": _rate(primary_exact, total),
        "taxonomy_exact": _rate(taxonomy_exact, total),
        "duplicate_precision": precision,
        "duplicate_recall": recall,
        "duplicate_f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "unresolved": sum(
            row_weight(label) for label, prediction in rows if prediction.get("same_duplicate_group") == "UNRESOLVED"
        ),
    }


def evaluate_predictions(
    labels: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one judge against human labels, including strata and confidence calibration."""

    label_rows = list(labels)
    prediction_rows = list(predictions)
    by_pair = {str(row["canonical_pair_id"]): row for row in prediction_rows}
    require(
        len(by_pair) == len(prediction_rows),
        "CALIBRATION_PREDICTION_DUPLICATE",
        "predictions must have unique canonical pair IDs",
        rows=len(prediction_rows),
        unique_pairs=len(by_pair),
    )
    joined = []
    for label in label_rows:
        pair_id = str(label["canonical_pair_id"])
        require(pair_id in by_pair, "CALIBRATION_JOIN_INCOMPLETE", "prediction is missing", pair_id=pair_id)
        joined.append((label, by_pair[pair_id]))

    by_reason: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_tier: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_ledger: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = {
        prefix: defaultdict(list) for prefix in SEMANTIC_LEDGER_REASON_PREFIXES
    }
    relation_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for label, prediction in joined:
        by_reason[str(label.get("human_reason_code") or "UNSPECIFIED")].append((label, prediction))
        if prediction.get("confidence_tier"):
            by_tier[str(prediction["confidence_tier"])].append((label, prediction))
        for prefix in SEMANTIC_LEDGER_REASON_PREFIXES:
            if (value := _reason_code_value(prediction, prefix)) is not None:
                by_ledger[prefix][value].append((label, prediction))
        relation_matrix[str(label.get("human_relation_type"))][str(prediction.get("relation_type"))] += 1

    return {
        "unweighted": _metric_block(joined, weighted=False),
        "weighted": _metric_block(joined, weighted=True),
        "over_group": sum(
            label.get("human_same_duplicate_group") != "YES" and prediction.get("same_duplicate_group") == "YES"
            for label, prediction in joined
        ),
        "containment_over_group": sum(
            label.get("human_same_duplicate_group") == "NO"
            and prediction.get("same_duplicate_group") == "YES"
            and prediction.get("relation_type") == "CONTAINMENT"
            for label, prediction in joined
        ),
        "under_group": sum(
            label.get("human_same_duplicate_group") == "YES" and prediction.get("same_duplicate_group") != "YES"
            for label, prediction in joined
        ),
        "containment_under_group": sum(
            label.get("human_relation_type") == "CONTAINMENT" and prediction.get("same_duplicate_group") != "YES"
            for label, prediction in joined
        ),
        "cohorts": {
            reason: {
                "rows": len(selected),
                "primary_decision_exact": _metric_block(selected, weighted=False)["primary_decision_exact"],
                "weighted_primary_decision_exact": _metric_block(selected, weighted=True)["primary_decision_exact"],
                "over_group": sum(
                    label.get("human_same_duplicate_group") != "YES"
                    and prediction.get("same_duplicate_group") == "YES"
                    for label, prediction in selected
                ),
                "under_group": sum(
                    label.get("human_same_duplicate_group") == "YES"
                    and prediction.get("same_duplicate_group") != "YES"
                    for label, prediction in selected
                ),
            }
            for reason, selected in sorted(by_reason.items())
        },
        "confidence_tiers": {
            tier: {
                "rows": len(selected),
                "primary_decision_accuracy": _metric_block(selected, weighted=False)["primary_decision_exact"],
                "weighted_primary_decision_accuracy": _metric_block(selected, weighted=True)["primary_decision_exact"],
                "duplicate_group_accuracy": sum(
                    label.get("human_same_duplicate_group") == prediction.get("same_duplicate_group")
                    for label, prediction in selected
                )
                / len(selected),
                "weighted_duplicate_group_accuracy": _rate(
                    sum(
                        _weight(label)
                        for label, prediction in selected
                        if label.get("human_same_duplicate_group") == prediction.get("same_duplicate_group")
                    ),
                    sum(_weight(label) for label, _ in selected),
                ),
            }
            for tier, selected in sorted(by_tier.items())
        },
        "semantic_ledger": {
            prefix: {
                value: {
                    "rows": len(selected),
                    "primary_decision_accuracy": _metric_block(selected, weighted=False)["primary_decision_exact"],
                    "weighted_primary_decision_accuracy": _metric_block(selected, weighted=True)[
                        "primary_decision_exact"
                    ],
                    "over_group": sum(
                        label.get("human_same_duplicate_group") != "YES"
                        and prediction.get("same_duplicate_group") == "YES"
                        for label, prediction in selected
                    ),
                    "under_group": sum(
                        label.get("human_same_duplicate_group") == "YES"
                        and prediction.get("same_duplicate_group") != "YES"
                        for label, prediction in selected
                    ),
                }
                for value, selected in sorted(groups.items())
            }
            for prefix, groups in by_ledger.items()
            if groups
        },
        "relation_matrix": {truth: dict(sorted(counts.items())) for truth, counts in sorted(relation_matrix.items())},
    }


def evaluate_development_gates(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    schema_completion_rate: float,
    retry_rate: float,
    thresholds: DevelopmentGateThresholds = DEFAULT_DEVELOPMENT_THRESHOLDS,
) -> dict[str, Any]:
    weighted = candidate["weighted"]
    checks = {
        "weighted_precision": weighted["duplicate_precision"] >= thresholds.weighted_precision,
        "weighted_recall": weighted["duplicate_recall"] >= thresholds.weighted_recall,
        "weighted_primary_exact": weighted["primary_decision_exact"] >= thresholds.weighted_primary_exact,
        "over_group": candidate["over_group"] <= thresholds.max_over_group,
        "schema_completion": schema_completion_rate >= thresholds.schema_completion,
        "retry_rate": retry_rate <= thresholds.max_retry_rate,
    }
    regressions = {}
    for cohort in ("identity_slot", "meaningful_addition", "translation"):
        candidate_accuracy = candidate["cohorts"][cohort]["weighted_primary_decision_exact"]
        baseline_accuracy = baseline["cohorts"][cohort]["weighted_primary_decision_exact"]
        delta = candidate_accuracy - baseline_accuracy
        regressions[cohort] = {"delta": delta, "passed": delta >= -thresholds.max_cohort_regression}
        checks[f"cohort_{cohort}"] = regressions[cohort]["passed"]
    return {"passed": all(checks.values()), "checks": checks, "cohort_regressions": regressions}


def _containment_misses(metrics: dict[str, Any]) -> int:
    return int(metrics["containment_under_group"])


def evaluate_holdout_gates(
    *,
    candidate_all: dict[str, Any],
    candidate_representative: dict[str, Any],
    baseline_representative: dict[str, Any],
    candidate_difficult: dict[str, Any],
    baseline_difficult: dict[str, Any],
    thresholds: HoldoutGateThresholds = DEFAULT_HOLDOUT_THRESHOLDS,
) -> dict[str, Any]:
    representative_delta = (
        candidate_representative["unweighted"]["primary_decision_exact"]
        - baseline_representative["unweighted"]["primary_decision_exact"]
    )
    baseline_over = baseline_difficult["containment_over_group"]
    candidate_over = candidate_difficult["containment_over_group"]
    over_reduction = (baseline_over - candidate_over) / baseline_over if baseline_over else 1.0
    baseline_misses = _containment_misses(baseline_difficult)
    candidate_misses = _containment_misses(candidate_difficult)
    miss_growth = (
        (candidate_misses - baseline_misses) / baseline_misses if baseline_misses else float(candidate_misses > 0)
    )
    checks = {
        "representative_noninferiority": representative_delta >= thresholds.representative_noninferiority,
        "precision": candidate_all["unweighted"]["duplicate_precision"] >= thresholds.precision,
        "recall": candidate_all["unweighted"]["duplicate_recall"] >= thresholds.recall,
        "difficult_over_group_reduction": over_reduction >= thresholds.difficult_over_group_reduction,
        "containment_miss_growth": miss_growth <= thresholds.max_containment_miss_growth,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "representative_primary_exact_delta": representative_delta,
        "difficult_over_group_reduction": over_reduction,
        "containment_miss_growth": miss_growth,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "CALIBRATION_INPUT_INVALID", "expected a JSON object", path=str(path))
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _run_inputs(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    return (
        _read_json(run_root / "run_manifest.json"),
        _read_json(run_root / "run_complete.json"),
        _read_jsonl(run_root / "data" / "judge_results.jsonl"),
    )


def compare_development_runs(
    *,
    labels: list[dict[str, Any]],
    baseline_run_root: Path,
    candidate_run_roots: dict[str, Path],
) -> dict[str, Any]:
    """Compare equal-model, equal-temperature, equal-payload development variants and apply gates."""

    baseline_manifest, baseline_complete, baseline_rows = _run_inputs(baseline_run_root)
    baseline_metrics = evaluate_predictions(labels, baseline_rows)
    stable_settings = (
        "hub_model",
        "logical_model",
        "temperature",
        "top_p",
        "max_output_tokens",
        "max_visible_tokens",
        "window_tokens",
        "window_overlap_tokens",
    )
    variants = {}
    for name, run_root in candidate_run_roots.items():
        manifest, complete, rows = _run_inputs(run_root)
        require(
            manifest["source_pair_ids_sha256"] == baseline_manifest["source_pair_ids_sha256"]
            and manifest["payload_membership_sha256"] == baseline_manifest["payload_membership_sha256"],
            "DEVELOPMENT_PAYLOAD_MISMATCH",
            "development variants must judge the same frozen payloads",
            variant=name,
        )
        require(
            all(manifest["settings"][key] == baseline_manifest["settings"][key] for key in stable_settings),
            "DEVELOPMENT_EXECUTION_MISMATCH",
            "development variants must use the same model, temperature, and context settings",
            variant=name,
        )
        metrics = evaluate_predictions(labels, rows)
        requested = int(complete["requested"])
        completion_rate = int(complete["valid"]) / requested
        retry_rate = int(complete["retried"]) / requested
        variants[name] = {
            "metrics": metrics,
            "operations": {"schema_completion_rate": completion_rate, "retry_rate": retry_rate},
            "development_gates": evaluate_development_gates(
                candidate=metrics,
                baseline=baseline_metrics,
                schema_completion_rate=completion_rate,
                retry_rate=retry_rate,
            ),
        }
    return {
        "schema_version": "dedup-v062-development-comparison-v1",
        "baseline": {
            "run_root": str(baseline_run_root),
            "metrics": baseline_metrics,
            "operations": {
                "schema_completion_rate": int(baseline_complete["valid"]) / int(baseline_complete["requested"]),
                "retry_rate": int(baseline_complete["retried"]) / int(baseline_complete["requested"]),
            },
        },
        "variants": variants,
        "selection_rule": "Only a variant with development_gates.passed=true may advance to the frozen holdout.",
    }


def _named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        require(bool(name and separator and path), "CALIBRATION_ARGUMENT_INVALID", "candidate must be NAME=RUN_ROOT")
        require(name not in result, "CALIBRATION_ARGUMENT_INVALID", "candidate names must be unique", name=name)
        result[name] = Path(path).resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=RUN_ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_development_runs(
        labels=_read_csv(args.labels),
        baseline_run_root=args.baseline_run_root.resolve(),
        candidate_run_roots=_named_paths(args.candidate),
    )
    write_json_atomic(args.output, comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
