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

"""Select and blind the frozen V0.6.2 representative and difficult holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

from eval.dedup.validation import require, write_json_atomic, write_text_atomic

HOLDOUT_SCHEMA: Final = "dedup-v062-holdout-v1"
REPRESENTATIVE_COUNT: Final = 200
DIFFICULT_COUNT: Final = 200
DOUBLE_REVIEW_PER_SPLIT: Final = 50
TARGET_QUOTA: Final = 40
TARGET_CATEGORIES: Final = (
    "SUSPECT_BOILERPLATE_CONTAINMENT",
    "POSSIBLE_MISSED_CONTAINMENT",
    "CHROME_OR_NON_MAIN_ONLY",
    "TRANSLATION",
    "TRUNCATION_PAGE_ROLE_OR_SLOT",
)


def _stable_rank(seed: int, pair_id: str, namespace: str) -> str:
    return hashlib.sha256(f"{HOLDOUT_SCHEMA}\0{seed}\0{namespace}\0{pair_id}".encode()).hexdigest()


def _ratio_bucket(row: dict[str, Any]) -> str:
    ratio = float(row["token_length_ratio"])
    if ratio < 0.25:
        return "0-0.25"
    if ratio < 0.5:
        return "0.25-0.5"
    if ratio < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def _track(row: dict[str, Any]) -> str:
    if row.get("has_track_5a") and row.get("has_track_5b"):
        return "5a+5b"
    if row.get("has_track_5a"):
        return "5a"
    if row.get("has_track_5b"):
        return "5b"
    return "unknown"


def representative_stratum(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _track(row),
        "same_language" if row.get("same_language") else "cross_language",
        str(row.get("length_bucket_low")),
        _ratio_bucket(row),
        "same_hostname" if row.get("same_hostname") else "cross_hostname",
    )


def stratified_representative_sample(
    rows: Iterable[dict[str, Any]],
    *,
    count: int = REPRESENTATIVE_COUNT,
    seed: int,
) -> list[dict[str, Any]]:
    """Allocate proportional largest-remainder quotas and sample deterministically."""

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[representative_stratum(row)].append(row)
    total = sum(map(len, groups.values()))
    require(
        total >= count, "HOLDOUT_POOL_TOO_SMALL", "representative pool is too small", available=total, needed=count
    )
    exact = {key: count * len(selected) / total for key, selected in groups.items()}
    quotas = {key: min(len(groups[key]), int(value)) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(
        groups,
        key=lambda key: (-(exact[key] - int(exact[key])), _stable_rank(seed, "|".join(key), "stratum")),
    )
    for key in order:
        if not remaining:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    require(remaining == 0, "HOLDOUT_ALLOCATION_FAILED", "representative quota allocation did not close")
    output = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key], key=lambda row: _stable_rank(seed, str(row["canonical_pair_id"]), "representative")
        )
        output.extend({**row, "holdout_stratum": "|".join(key)} for row in ranked[: quotas[key]])
    return sorted(output, key=lambda row: _stable_rank(seed, str(row["canonical_pair_id"]), "representative-order"))


def target_categories(row: dict[str, Any]) -> set[str]:
    overlap = row.get("dominant_overlap_source")
    risk = row.get("primary_risk_factor")
    primary = row.get("primary_material_difference")
    categories = set()
    non_main_overlap = overlap in {
        "SHARED_PAGE_TEMPLATE",
        "SITE_CHROME",
        "COOKIE_CONSENT",
        "LEGAL_POLICY_TEMPLATE",
        "ERROR_AUTH_PAYWALL",
    }
    if row.get("relation_type") == "CONTAINMENT" and (
        non_main_overlap or risk in {"BOILERPLATE_DOMINATED_SIMILARITY", "TEMPLATE_SLOT_COLLISION"}
    ):
        categories.add("SUSPECT_BOILERPLATE_CONTAINMENT")
    if row.get("same_duplicate_group") == "NO" and (
        primary == "MAIN_CONTENT_ADDITION_DELETION" or risk == "CONTAINMENT_ASYMMETRY"
    ):
        categories.add("POSSIBLE_MISSED_CONTAINMENT")
    if non_main_overlap or risk in {"BOILERPLATE_DOMINATED_SIMILARITY", "PARSER_ARTIFACT_DOMINANCE"}:
        categories.add("CHROME_OR_NON_MAIN_ONLY")
    if risk == "TRANSLATION_EQUIVALENCE" or not row.get("same_language"):
        categories.add("TRANSLATION")
    if row.get("payload_truncated") or risk in {
        "EXTRACTION_OR_PAYLOAD_LIMIT",
        "PAGE_ROLE_COLLISION",
        "TEMPLATE_SLOT_COLLISION",
        "IDENTIFIER_UNDERWEIGHTING",
        "LIST_SNAPSHOT_COLLISION",
    }:
        categories.add("TRUNCATION_PAGE_ROLE_OR_SLOT")
    return categories


def targeted_difficult_sample(
    rows: Iterable[dict[str, Any]],
    *,
    excluded_pair_ids: set[str],
    seed: int,
    quota: int = TARGET_QUOTA,
) -> list[dict[str, Any]]:
    """Select equal, mutually exclusive difficult cohorts in a frozen priority order."""

    pool = [row for row in rows if str(row["canonical_pair_id"]) not in excluded_pair_ids]
    used: set[str] = set()
    output = []
    for category in TARGET_CATEGORIES:
        candidates = [
            row for row in pool if str(row["canonical_pair_id"]) not in used and category in target_categories(row)
        ]
        ranked = sorted(
            candidates, key=lambda row: _stable_rank(seed, str(row["canonical_pair_id"]), f"target:{category}")
        )
        require(
            len(ranked) >= quota,
            "HOLDOUT_TARGET_QUOTA_UNMET",
            "a difficult holdout category does not have enough unused candidates",
            category=category,
            available=len(ranked),
            needed=quota,
        )
        chosen = ranked[:quota]
        used.update(str(row["canonical_pair_id"]) for row in chosen)
        output.extend({**row, "holdout_target_category": category} for row in chosen)
    return output


def build_holdout(
    *,
    comparison_rows: Iterable[dict[str, Any]],
    payload_by_pair: dict[str, dict[str, Any]],
    excluded_pair_ids: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a private selection manifest and a prediction-blind reviewer packet."""

    eligible = [row for row in comparison_rows if str(row["canonical_pair_id"]) not in excluded_pair_ids]
    representative = stratified_representative_sample(eligible, seed=seed)
    representative_ids = {str(row["canonical_pair_id"]) for row in representative}
    difficult = targeted_difficult_sample(eligible, excluded_pair_ids=representative_ids, seed=seed)
    require(
        len(representative) == REPRESENTATIVE_COUNT and len(difficult) == DIFFICULT_COUNT,
        "HOLDOUT_SIZE_INVALID",
        "holdout splits must contain exactly 200 pairs each",
    )
    double_review_ids = {
        str(row["canonical_pair_id"])
        for split, rows in (("representative", representative), ("difficult", difficult))
        for row in sorted(
            rows, key=lambda item: _stable_rank(seed, str(item["canonical_pair_id"]), f"double:{split}")
        )[:DOUBLE_REVIEW_PER_SPLIT]
    }
    private_rows = []
    public_rows = []
    for index, (split, row) in enumerate(
        [("representative", item) for item in representative] + [("difficult", item) for item in difficult],
        start=1,
    ):
        pair_id = str(row["canonical_pair_id"])
        require(pair_id in payload_by_pair, "HOLDOUT_PAYLOAD_MISSING", "selected pair has no blind payload")
        qa_pair_id = f"H062-{index:04d}-{_stable_rank(seed, pair_id, 'qa')[:8]}"
        private_rows.append(
            {
                "schema_version": HOLDOUT_SCHEMA,
                "qa_pair_id": qa_pair_id,
                "canonical_pair_id": pair_id,
                "split": split,
                "selection_cell": row.get("holdout_stratum") or row.get("holdout_target_category"),
                "review_mode": "DOUBLE_INDEPENDENT" if pair_id in double_review_ids else "SINGLE",
                "disagreement_policy": "THIRD_REVIEWER_ADJUDICATION",
            }
        )
        public_rows.append({"qa_pair_id": qa_pair_id, "visible_payload": payload_by_pair[pair_id]})
    return private_rows, public_rows


def _read_excluded_pair_ids(paths: list[Path]) -> set[str]:
    result = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            result.update(str(row["canonical_pair_id"]) for row in csv.DictReader(file))
    return result


def _read_payloads(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return {str(row["canonical_pair_id"]): row["payload"] for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", required=True, type=Path)
    parser.add_argument("--payloads", required=True, type=Path)
    parser.add_argument("--exclude-csv", required=True, action="append", type=Path)
    parser.add_argument("--private-manifest", required=True, type=Path)
    parser.add_argument("--review-packet", required=True, type=Path)
    parser.add_argument("--review-dashboard", type=Path)
    parser.add_argument("--selection-summary", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=62026)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for holdout selection") from exc
    comparisons = pq.read_table(args.comparisons).to_pylist()
    payloads = _read_payloads(args.payloads)
    for row in comparisons:
        payload = payloads.get(str(row["canonical_pair_id"]), {})
        row["payload_truncated"] = payload.get("long_document_evidence", {}).get("truncated") is True
    private, public = build_holdout(
        comparison_rows=comparisons,
        payload_by_pair=payloads,
        excluded_pair_ids=_read_excluded_pair_ids(args.exclude_csv),
        seed=args.seed,
    )
    write_text_atomic(
        args.private_manifest,
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in private),
    )
    write_text_atomic(
        args.review_packet,
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in public),
    )
    if args.review_dashboard is not None:
        from eval.dedup.human_qa_dashboard import publish_human_qa_dashboard

        publish_human_qa_dashboard(
            packet_path=args.review_packet,
            destination=args.review_dashboard,
            evaluation_run_id="v0.6.2-holdout",
            packet_label="v062_holdout",
        )
    write_json_atomic(
        args.selection_summary,
        {
            "schema_version": HOLDOUT_SCHEMA,
            "seed": args.seed,
            "total": len(private),
            "representative": sum(row["split"] == "representative" for row in private),
            "difficult": sum(row["split"] == "difficult" for row in private),
            "double_review": sum(row["review_mode"] == "DOUBLE_INDEPENDENT" for row in private),
            "review_packet_hides_predictions_and_sampling_reason": True,
            "review_dashboard": str(args.review_dashboard) if args.review_dashboard is not None else None,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
