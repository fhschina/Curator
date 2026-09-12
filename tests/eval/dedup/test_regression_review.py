# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import regression_review as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V1
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json


def fixture():
    payloads = {
        "pair1": {
            "payload_schema_version": VISIBLE_PAYLOAD_V1,
            "document_a": {"text": "界面 Cookie"},
            "document_b": {"text": "界面 Cookie\nArticle X"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
        "pair2": {
            "payload_schema_version": VISIBLE_PAYLOAD_V1,
            "document_a": {"text": "Boots\nPay"},
            "document_b": {"text": "Pay"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    }
    rows = []
    for n, (pid, payload) in enumerate(payloads.items(), 1):
        rows.append(
            {
                "review_id": f"H{n}",
                "canonical_pair_id": pid,
                "weight": 3.0 * n,
                "payload_sha256": sha256_json(payload),
                "transition": "REGRESSED",
                "errors": {
                    "old_main": "OVER_GROUP",
                    "old_final": "CORRECT",
                    "new_main": "OVER_GROUP",
                    "new_final": "OVER_GROUP",
                },
                "reference": {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "same_duplicate_group": "NO"},
                "reference_review_flags": [],
                "observed_stage_pattern": subject.REPAIR,
                "primary": {"old_final": {"same_duplicate_group": "NO"}, "new_final": {"same_duplicate_group": "YES"}},
            }
        )
    spec = {
        "review_method": subject.METHOD,
        "reference_changed": False,
        "policy": "dedup-composite-containment-policy-v1",
        "columns": subject.COLUMNS,
        "cases": [
            [
                "H1",
                "CLEAR_MODEL_ERROR",
                "EMPTY_MAIN_ANCHOR",
                "ABSENT",
                "NO_NO",
                "Cookie",
                "Article X",
                "Only a shared cookie message.",
            ],
            [
                "H2",
                "POLICY_REFERENCE_DISPUTE",
                "BARE_TITLE_OR_CTA",
                "ABSENT",
                "UNDECIDED",
                "Boots",
                "Pay",
                "Bare title binding awaits adjudication.",
            ],
        ],
    }
    return rows, payloads, spec


def test_reviews_preserve_cohort_weight_evidence_and_ai_provenance():
    rows, payloads, spec = fixture()
    result = subject.validate_reviews(spec, rows, payloads)
    assert len(result) == 2
    assert sum(r["weight"] for r in result) == 9
    assert result[0]["evidence"][0]["start_char"] == 3
    assert result[0]["evidence"][0]["start_byte_utf8"] == 7
    for row in result:
        assert row["independently_adjudicated"] is False
        assert row["reference"]["same_duplicate_group"] == "NO"
    assert subject.aggregate(result, "assessment")["POLICY_REFERENCE_DISPUTE"]["weight"] == 6


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "short",
        "payload",
        "gold",
        "not_fp",
    ],
)
def test_invalid_review_cannot_be_frozen(mutation):
    rows, payloads, spec = fixture()
    if mutation == "missing":
        spec["cases"].pop()
    elif mutation == "duplicate":
        spec["cases"].append(deepcopy(spec["cases"][0]))
    elif mutation == "short":
        spec["cases"][0].pop()
    elif mutation == "payload":
        payloads["pair1"]["document_a"]["text"] += "changed"
    elif mutation == "gold":
        spec["review_method"] = "HUMAN_GOLD"
    elif mutation == "not_fp":
        rows[0]["errors"]["new_final"] = "CORRECT"
    with pytest.raises(DedupEvaluationError):
        subject.validate_reviews(spec, rows, payloads)


@pytest.mark.parametrize(
    ("column", "value"),
    [(0, "H9999"), (5, "not in source"), (3, "YES"), (-1, ""), (4, "EQUIVALENT"), (1, "PENDING_JUDGMENT")],
)
def test_invalid_review_fields_fail_closed(column, value):
    rows, payloads, spec = fixture()
    spec["cases"][0][column] = value
    with pytest.raises(DedupEvaluationError):
        subject.validate_reviews(spec, rows, payloads)


def test_repeated_inputs_keep_population_weight_but_mirrored_directions_must_agree():
    rows, payloads, spec = fixture()
    payloads["pair2"] = deepcopy(payloads["pair1"])
    p = payloads["pair2"]
    p["document_a"], p["document_b"] = p["document_b"], p["document_a"]
    rows[1]["payload_sha256"] = sha256_json(p)
    spec["cases"][0][1:5] = ["POLICY_REFERENCE_DISPUTE", "INDEPENDENT_Y_NOW_ALLOWED", "PRESENT", "B_CONTAINS_A"]
    spec["cases"][1] = [
        "H2",
        "POLICY_REFERENCE_DISPUTE",
        "INDEPENDENT_Y_NOW_ALLOWED",
        "PRESENT",
        "A_CONTAINS_B",
        "Article X",
        "Cookie",
        "Mirrored proposal.",
    ]
    result = subject.validate_reviews(spec, rows, payloads)
    group = subject.aggregate(result, "assessment")["POLICY_REFERENCE_DISPUTE"]
    assert (group["pairs"], group["unique_unordered_text_pairs"], group["weight"]) == (2, 1, 9)
    spec["cases"][1][4] = "B_CONTAINS_A"
    with pytest.raises(DedupEvaluationError, match="REGRESSION_DUPLICATE_DISAGREEMENT"):
        subject.validate_reviews(spec, rows, payloads)


def test_prior_repair_inventory_retains_disputes_without_forcing_old_labels():
    rows, payloads, spec = fixture()
    reviewed = subject.validate_reviews(spec, rows, payloads)
    raw = {r["canonical_pair_id"]: ({"direction": "YES"}, {"reason": "different record"}) for r in rows}
    inventory = subject.repair_inventory(reviewed, raw)
    assert len(inventory) == 2
    assert [r["review_id"] for r in inventory if r["include_in_development_regression"]] == ["H1"]
    assert inventory[0]["old_reason_status"].startswith("CONCLUSION_SUPPORTED_USE_NONEMPTY_X")
    assert not inventory[1]["expected_primary_is_human_gold"]
    assert inventory[1]["old_critic_raw"] == raw["pair2"][1]
    raw["pair2"] = ({}, {})
    with pytest.raises(DedupEvaluationError, match="REGRESSION_REPAIR_RAW"):
        subject.repair_inventory(reviewed, raw)


def test_panel_keeps_parent_disputes_and_adds_every_supported_repair():
    rows, payloads, spec = fixture()
    reviewed = subject.validate_reviews(spec, rows, payloads)
    parent = [{"review_id": "H2", "role": "suspected_template_containment"}]
    panel_spec = {"additions": [{"review_id": "H1", "role": "supported_prior_repair"}]}
    result = subject.panel_revision(parent, reviewed, rows, payloads, panel_spec)
    assert [r["review_id"] for r in result] == ["H2", "H1"]
    assert result[0]["routing"] == "POLICY_ARBITRATION"
    assert all(r["independent_shared_x_gold"] is None for r in result)
    assert result[1]["mandatory_supported_prior_repair"]
    with pytest.raises(DedupEvaluationError, match="REGRESSION_PANEL_GUARDS"):
        subject.panel_revision(parent, reviewed, rows, payloads, {"additions": []})


def test_panel_retains_historical_review_flags():
    rows, payloads, spec = fixture()
    reviewed = subject.validate_reviews(spec, rows, payloads)
    rows[0]["reference_review_flags"] = ["HISTORICAL_DISPUTE_OR_PENDING"]
    result = subject.panel_revision(
        [{"review_id": r["review_id"], "role": "retained"} for r in rows], reviewed, rows, payloads, {"additions": []}
    )
    assert result[0]["routing"] == "HISTORICAL_REVIEW_FLAG_RETAINED"
    assert result[1]["routing"] == "POLICY_ARBITRATION"


def test_panel_rejects_exact_duplicate_samples_and_missing_mechanisms():
    rows, payloads, spec = fixture()
    reviewed = subject.validate_reviews(spec, rows, payloads)
    parent = [{"review_id": r["review_id"], "role": "retained"} for r in rows]
    payloads["pair2"] = deepcopy(payloads["pair1"])
    rows[1]["payload_sha256"] = sha256_json(payloads["pair2"])
    with pytest.raises(DedupEvaluationError, match="REGRESSION_PANEL_DUPLICATE"):
        subject.panel_revision(parent, reviewed, rows, payloads, {"additions": []})
    reviewed[1]["assessment"] = "CLEAR_MODEL_ERROR"
    reviewed[1]["observed_stage_pattern"] = "NEW_MAIN_DISAGREEMENT"
    with pytest.raises(DedupEvaluationError, match="REGRESSION_PANEL_MECHANISMS"):
        subject.panel_revision(parent[:1], reviewed, rows, payloads, {"additions": []})


def test_blind_packet_contains_full_text_no_labels_and_is_immutable(tmp_path):
    rows, payloads, _ = fixture()
    original = deepcopy(payloads)
    blind = subject.write_packets(tmp_path, "arbitration", rows, payloads)
    assert payloads == original
    assert len(blind) == 2
    assert all(set(r) == {"case_id", "payload"} for r in blind)
    assert {json.dumps(r["payload"], sort_keys=True) for r in blind} == {
        json.dumps(p, sort_keys=True) for p in payloads.values()
    }
    path = tmp_path / "arbitration/inputs_blind.jsonl"
    digest = sha256_file(path)
    subject.write_packets(tmp_path, "arbitration", rows, payloads)
    assert sha256_file(path) == digest
    payloads["pair1"]["document_a"]["text"] = "changed"
    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        subject.write_packets(tmp_path, "arbitration", rows, payloads)


@pytest.mark.parametrize(("a", "b"), [("", "字\n"), ("same", "same"), ("a\nb", "b\na\n尾"), ("短", "长\n完整原文")])
def test_display_uses_complete_shorter_text_and_all_line_changes(a, b):
    value = subject.full_text_recipe({"document_a": {"text": a}, "document_b": {"text": b}})
    base = a if len(a) <= len(b) else b
    assert f"FULL ({len(base)} chars):\n{base}" in value
    for line in (b if len(a) <= len(b) else a).splitlines():
        assert line in value
