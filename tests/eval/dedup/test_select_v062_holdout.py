# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from collections import Counter

from eval.dedup.analysis.select_v062_holdout import build_holdout


def _row(index: int, category: str) -> dict:
    row = {
        "canonical_pair_id": f"cp1_{index:05d}",
        "has_track_5a": index % 2 == 0,
        "has_track_5b": index % 2 == 1,
        "same_language": True,
        "same_hostname": index % 3 == 0,
        "length_bucket_low": "medium",
        "token_length_ratio": 0.9,
        "same_duplicate_group": "NO",
        "relation_type": "UNRELATED",
        "primary_material_difference": "DOCUMENT_IDENTITY_CHANGE",
        "dominant_overlap_source": "NONE",
        "primary_risk_factor": "NONE",
        "payload_truncated": False,
    }
    if category == "boilerplate":
        row.update(relation_type="CONTAINMENT", dominant_overlap_source="COOKIE_CONSENT")
    elif category == "missed":
        row.update(primary_material_difference="MAIN_CONTENT_ADDITION_DELETION")
    elif category == "chrome":
        row.update(dominant_overlap_source="SITE_CHROME")
    elif category == "translation":
        row.update(primary_risk_factor="TRANSLATION_EQUIVALENCE")
    elif category == "page_role":
        row.update(primary_risk_factor="PAGE_ROLE_COLLISION")
    return row


def test_holdout_has_frozen_split_quotas_review_load_and_blind_public_packet() -> None:
    rows = []
    index = 0
    for category in ("neutral", "boilerplate", "missed", "chrome", "translation", "page_role"):
        for _ in range(300):
            rows.append(_row(index, category))
            index += 1
    payloads = {
        row["canonical_pair_id"]: {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "beta"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
        for row in rows
    }

    private, public = build_holdout(
        comparison_rows=rows,
        payload_by_pair=payloads,
        excluded_pair_ids={"cp1_00000"},
        seed=62026,
    )

    assert Counter(row["split"] for row in private) == {"representative": 200, "difficult": 200}
    assert sum(row["review_mode"] == "DOUBLE_INDEPENDENT" for row in private) == 100
    assert Counter(row["selection_cell"] for row in private if row["split"] == "difficult") == {
        "SUSPECT_BOILERPLATE_CONTAINMENT": 40,
        "POSSIBLE_MISSED_CONTAINMENT": 40,
        "CHROME_OR_NON_MAIN_ONLY": 40,
        "TRANSLATION": 40,
        "TRUNCATION_PAGE_ROLE_OR_SLOT": 40,
    }
    assert all(set(row) == {"qa_pair_id", "visible_payload"} for row in public)
    assert all("canonical_pair_id" not in row for row in public)
