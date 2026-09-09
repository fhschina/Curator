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

"""Build a human-review packet for semantic-policy and model disagreements."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from eval.dedup.validation import require, write_json_atomic

POLICY_BOUNDARY_RULES: Final = {
    "SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH",
    "NON_MAIN_MESSAGE_NOT_EQUIVALENT",
    "NO_SHARED_SUBSTANTIVE_ANCHOR",
    "NO_CONFIRMED_SAME_SUBSTANTIVE_RECORD",
    "NO_EQUIVALENT_NON_MAIN_MESSAGE",
    "MATERIAL_NON_MAIN_DIFFERENCE",
}
MODEL_CLASSIFICATION_RULES: Final = {
    "HARD_CONFLICT_VETO",
    "TWO_SIDED_MAIN_DIVERGENCE",
}
LEDGER_PREFIXES: Final = (
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
    "SEMANTIC_LEDGER_RULE",
)
ADJUDICATION_FIELDS: Final = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
)
DEFAULT_REVIEW_PROTOCOL: Final = "PREDICTION_ISOLATED_V062_POLICY"


def _reason_value(prediction: dict[str, Any], prefix: str) -> str | None:
    marker = f"{prefix}:"
    values = [
        code[len(marker) :]
        for code in prediction.get("reason_codes", [])
        if isinstance(code, str) and code.startswith(marker)
    ]
    require(
        len(values) <= 1,
        "POLICY_REVIEW_LEDGER_INVALID",
        "a prediction cannot contain duplicate semantic-ledger reason codes",
        prefix=prefix,
    )
    return values[0] if values else None


def _index(rows: Iterable[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("canonical_pair_id", ""))
        require(pair_id, "POLICY_REVIEW_JOIN_INVALID", "canonical pair ID is required", source=source)
        require(
            pair_id not in indexed,
            "POLICY_REVIEW_JOIN_INVALID",
            "canonical pair IDs must be unique",
            source=source,
            pair_id=pair_id,
        )
        indexed[pair_id] = row
    return indexed


def _weight(label: dict[str, Any]) -> float:
    population = label.get("stratum_population_n")
    sample = label.get("stratum_sample_n")
    if population not in {None, ""} and sample not in {None, ""}:
        sample_value = float(sample)
        require(sample_value > 0, "POLICY_REVIEW_WEIGHT_INVALID", "sample size must be positive")
        return float(population) / sample_value
    value = label.get("sample_weight", 1.0)
    return float(value) if value not in {None, ""} else 1.0


def _snippet(value: str, *, limit: int = 1200) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def _evidence(prediction: dict[str, Any], side: str) -> str:
    return " | ".join(
        str(item.get("quote", ""))
        for item in prediction.get("evidence", [])
        if item.get("side") == side and item.get("quote")
    )


def _payload_side_text(payload: dict[str, Any], side: str) -> str:
    document = payload.get("document_a" if side == "A" else "document_b", {})
    text = document.get("text") if isinstance(document, dict) else None
    if isinstance(text, str):
        return text
    windows = payload.get("long_document_evidence", {}).get("windows", [])
    return "\n".join(
        str(window["text"])
        for window in windows
        if window.get("side") == side and isinstance(window.get("text"), str)
    )


def _triage_category(label: dict[str, Any], prediction: dict[str, Any], ledger_rule: str | None) -> str:
    human_group = label.get("human_same_duplicate_group")
    predicted_group = prediction.get("same_duplicate_group")
    reason = label.get("human_reason_code")
    if human_group == "YES" and predicted_group != "YES":
        if reason == "translation":
            category = "TRANSLATION_COMPLETENESS_REVIEW"
        elif label.get("human_relation_type") == "CONTAINMENT" and ledger_rule in POLICY_BOUNDARY_RULES:
            category = "POTENTIAL_POLICY_LABEL_CONFLICT"
        elif ledger_rule in MODEL_CLASSIFICATION_RULES:
            category = "POTENTIAL_MODEL_CLASSIFICATION_ERROR"
        else:
            category = "OTHER_UNDER_GROUP_REVIEW"
    elif human_group == "NO" and predicted_group == "YES":
        if prediction.get("relation_type") == "CONTAINMENT":
            category = "REMAINING_OVER_CONTAINMENT"
        else:
            category = "OTHER_OVER_GROUP_REVIEW"
    else:
        category = "DIRECTION_OR_TAXONOMY_REVIEW"
    return category


def _review_question(category: str) -> str:
    return {
        "POTENTIAL_POLICY_LABEL_CONFLICT": (
            "Under the V0.6.2 non-empty containment rule, do both sides share substantive main content from "
            "the same record, or does the old label rely only on boilerplate/non-main overlap?"
        ),
        "POTENTIAL_MODEL_CLASSIFICATION_ERROR": (
            "Did the judge invent a hard conflict/two-sided divergence, or is the old duplicate label too broad?"
        ),
        "TRANSLATION_COMPLETENESS_REVIEW": (
            "Is this a faithful complete translation with no added or omitted substantive facts?"
        ),
        "REMAINING_OVER_CONTAINMENT": (
            "Is the claimed one-sided addition really from the same record, with no identity, state, role, or slot conflict?"
        ),
        "OTHER_OVER_GROUP_REVIEW": "Does either replacement direction preserve all substantive meaning?",
        "OTHER_UNDER_GROUP_REVIEW": "Why does the new judge reject a human-labeled duplicate?",
        "DIRECTION_OR_TAXONOMY_REVIEW": "Which replacement direction or relation taxonomy is incorrect?",
    }[category]


def build_policy_review_rows(
    *,
    labels: Iterable[dict[str, Any]],
    candidate_predictions: Iterable[dict[str, Any]],
    previous_predictions: Iterable[dict[str, Any]],
    payloads: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return every primary-decision disagreement with reproducible triage context."""

    candidate_by_pair = _index(candidate_predictions, source="candidate")
    previous_by_pair = _index(previous_predictions, source="previous")
    payload_by_pair = _index(payloads, source="payload")
    output = []
    for label in labels:
        pair_id = str(label.get("canonical_pair_id", ""))
        require(
            pair_id in candidate_by_pair and pair_id in previous_by_pair and pair_id in payload_by_pair,
            "POLICY_REVIEW_JOIN_INVALID",
            "every label must join candidate, previous, and payload rows",
            pair_id=pair_id,
        )
        candidate = candidate_by_pair[pair_id]
        primary_exact = all(
            str(label.get(f"human_{field}")) == str(candidate.get(field))
            for field in ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
        )
        if primary_exact:
            continue
        previous = previous_by_pair[pair_id]
        payload = payload_by_pair[pair_id].get("payload", {})
        ledger = {prefix: _reason_value(candidate, prefix) for prefix in LEDGER_PREFIXES}
        ledger_rule = ledger["SEMANTIC_LEDGER_RULE"]
        category = _triage_category(label, candidate, ledger_rule)
        output.append(
            {
                "review_priority": {
                    "POTENTIAL_POLICY_LABEL_CONFLICT": 1,
                    "TRANSLATION_COMPLETENESS_REVIEW": 1,
                    "REMAINING_OVER_CONTAINMENT": 1,
                    "POTENTIAL_MODEL_CLASSIFICATION_ERROR": 2,
                    "OTHER_UNDER_GROUP_REVIEW": 2,
                    "OTHER_OVER_GROUP_REVIEW": 2,
                    "DIRECTION_OR_TAXONOMY_REVIEW": 3,
                }[category],
                "triage_category": category,
                "review_question": _review_question(category),
                "review_id": label.get("review_id"),
                "canonical_pair_id": pair_id,
                "sample_weight": _weight(label),
                "human_reason_code": label.get("human_reason_code"),
                "human_same_duplicate_group": label.get("human_same_duplicate_group"),
                "human_a_can_replace_b": label.get("human_a_can_replace_b"),
                "human_b_can_replace_a": label.get("human_b_can_replace_a"),
                "human_relation_type": label.get("human_relation_type"),
                "human_material_difference": label.get("human_material_difference"),
                "previous_same_duplicate_group": previous.get("same_duplicate_group"),
                "previous_a_can_replace_b": previous.get("a_can_replace_b"),
                "previous_b_can_replace_a": previous.get("b_can_replace_a"),
                "candidate_same_duplicate_group": candidate.get("same_duplicate_group"),
                "candidate_a_can_replace_b": candidate.get("a_can_replace_b"),
                "candidate_b_can_replace_a": candidate.get("b_can_replace_a"),
                "candidate_relation_type": candidate.get("relation_type"),
                "candidate_material_difference": candidate.get("material_difference"),
                "candidate_confidence_tier": candidate.get("confidence_tier"),
                **{prefix.lower(): value for prefix, value in ledger.items()},
                "candidate_evidence_a": _evidence(candidate, "A"),
                "candidate_evidence_b": _evidence(candidate, "B"),
                "document_a": _snippet(_payload_side_text(payload, "A")),
                "document_b": _snippet(_payload_side_text(payload, "B")),
                "blind_human_reason": label.get("blind_human_reason"),
                "hs_policy_lesson": label.get("hs_policy_lesson"),
                "review_outcome": "",
                "review_notes": "",
            }
        )
    return sorted(output, key=lambda row: (row["review_priority"], row["triage_category"], row["review_id"] or ""))


def summarize_policy_review(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    categories = Counter(str(row["triage_category"]) for row in selected)
    human_reasons = Counter(str(row.get("human_reason_code") or "UNSPECIFIED") for row in selected)
    ledger_rules = Counter(str(row.get("semantic_ledger_rule") or "MISSING") for row in selected)
    return {
        "schema_version": "dedup-v062-policy-review-v1",
        "rows": len(selected),
        "weighted_rows": sum(float(row["sample_weight"]) for row in selected),
        "categories": dict(sorted(categories.items())),
        "human_reason_codes": dict(sorted(human_reasons.items())),
        "semantic_ledger_rules": dict(sorted(ledger_rules.items())),
        "disposition": (
            "Review candidates only. Do not mutate development labels automatically, advance to holdout, "
            "or treat these rows as proven annotation errors."
        ),
    }


def apply_adjudication_overlay(
    labels: Iterable[dict[str, Any]], adjudications: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply an explicit review overlay while retaining every frozen pre-review label."""

    label_rows = list(labels)
    adjudication_rows = list(adjudications)
    by_review_id = {str(row.get("review_id", "")): row for row in adjudication_rows}
    require(
        len(by_review_id) == len(adjudication_rows) and all(by_review_id),
        "POLICY_ADJUDICATION_INVALID",
        "adjudications must have unique non-empty review IDs",
    )
    label_ids = {str(row.get("review_id", "")) for row in label_rows}
    require(
        set(by_review_id) <= label_ids,
        "POLICY_ADJUDICATION_INVALID",
        "every adjudication must join an existing review ID",
        unknown_review_ids=sorted(set(by_review_id) - label_ids),
    )
    allowed = {
        "same_duplicate_group": {"YES", "NO", "UNRESOLVED"},
        "a_can_replace_b": {"YES", "NO", "UNRESOLVED"},
        "b_can_replace_a": {"YES", "NO", "UNRESOLVED"},
        "relation_type": {
            "EXACT",
            "CANONICAL_EXACT",
            "NEAR_SURFACE",
            "CONTAINMENT",
            "VERSION_RELATED",
            "RELATED_NON_DUPLICATE",
            "UNRELATED",
            "UNRESOLVED",
        },
        "material_difference": {"NONE", "MINOR", "MAJOR", "UNRESOLVED"},
    }
    for review_id, adjudication in by_review_id.items():
        require(
            all(adjudication.get(field) in allowed[field] for field in ADJUDICATION_FIELDS)
            and adjudication.get("reason_code")
            and adjudication.get("confidence_tier") in {"HIGH", "MEDIUM", "LOW"}
            and adjudication.get("reviewer_notes"),
            "POLICY_ADJUDICATION_INVALID",
            "adjudication decisions and rationale are required",
            review_id=review_id,
        )
        require(
            bool(adjudication.get("review_protocol") or DEFAULT_REVIEW_PROTOCOL),
            "POLICY_ADJUDICATION_INVALID",
            "adjudication review protocol must be non-empty",
            review_id=review_id,
        )
    output = []
    changed_primary = 0
    changed_group = 0
    for label in label_rows:
        review_id = str(label.get("review_id", ""))
        adjudication = by_review_id.get(review_id)
        if adjudication is None:
            output.append(dict(label))
            continue
        row = dict(label)
        for field in ADJUDICATION_FIELDS:
            row[f"pre_policy_review_human_{field}"] = row.get(f"human_{field}")
        changed_primary += any(
            str(row.get(f"human_{field}")) != str(adjudication[field])
            for field in ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
        )
        changed_group += str(row.get("human_same_duplicate_group")) != str(adjudication["same_duplicate_group"])
        for field in ADJUDICATION_FIELDS:
            row[f"human_{field}"] = adjudication[field]
        row["policy_review_reason_code"] = adjudication["reason_code"]
        row["policy_review_confidence_tier"] = adjudication["confidence_tier"]
        row["policy_review_notes"] = adjudication["reviewer_notes"]
        row["policy_review_protocol"] = adjudication.get("review_protocol") or DEFAULT_REVIEW_PROTOCOL
        output.append(row)
    protocols = sorted(
        {str(row.get("review_protocol") or DEFAULT_REVIEW_PROTOCOL) for row in adjudication_rows}
    )
    return output, {
        "schema_version": "dedup-v062-policy-adjudication-summary-v1",
        "labels": len(output),
        "adjudicated": len(adjudication_rows),
        "changed_primary_tuple": changed_primary,
        "changed_duplicate_group": changed_group,
        "preserved_original_fields": [f"pre_policy_review_human_{field}" for field in ADJUDICATION_FIELDS],
        "review_protocol": protocols[0] if len(protocols) == 1 else protocols,
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), "POLICY_REVIEW_EMPTY", "policy review packet cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--candidate-run-root", type=Path, required=True)
    parser.add_argument("--previous-run-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--reconciled-labels", type=Path)
    parser.add_argument("--adjudication-summary", type=Path)
    args = parser.parse_args()
    candidate_root = args.candidate_run_root.resolve()
    previous_root = args.previous_run_root.resolve()
    rows = build_policy_review_rows(
        labels=_read_csv(args.labels.resolve()),
        candidate_predictions=_read_jsonl(candidate_root / "data" / "judge_results.jsonl"),
        previous_predictions=_read_jsonl(previous_root / "data" / "judge_results.jsonl"),
        payloads=_read_jsonl(candidate_root / "data" / "judge_payloads.jsonl"),
    )
    _write_csv(args.output_csv.resolve(), rows)
    write_json_atomic(args.summary.resolve(), summarize_policy_review(rows))
    if args.adjudications is not None:
        require(
            args.reconciled_labels is not None and args.adjudication_summary is not None,
            "POLICY_ADJUDICATION_ARGUMENT_INVALID",
            "reconciled labels and adjudication summary are required with adjudications",
        )
        reconciled, adjudication_summary = apply_adjudication_overlay(
            _read_csv(args.labels.resolve()), _read_csv(args.adjudications.resolve())
        )
        _write_csv(args.reconciled_labels.resolve(), reconciled)
        write_json_atomic(args.adjudication_summary.resolve(), adjudication_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
