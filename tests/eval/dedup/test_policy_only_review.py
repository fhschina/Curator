# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import policy_only_review as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V1
from eval.dedup.validation import DedupEvaluationError, sha256_json


def fixture():
    policy = "退货政策:14 天内退款,退回运费由买家承担。"
    payload = {
        "payload_schema_version": VISIBLE_PAYLOAD_V1,
        "document_a": {"text": policy},
        "document_b": {"text": policy + "\n商品 K7:重量 300 克。"},
        "long_document_evidence": {"truncated": False, "windows": []},
    }
    row = {
        "review_id": "H1",
        "canonical_pair_id": "p1",
        "weight": 10.0,
        "payload_sha256": sha256_json(payload),
        "reference": {"a_can_replace_b": "NO", "b_can_replace_a": "NO", "same_duplicate_group": "NO"},
        "primary": {"old_final": {"same_duplicate_group": "NO"}},
        "transition": "REGRESSED",
        "errors": {"new_final": "OVER_GROUP"},
    }
    other = deepcopy(row)
    other.update(review_id="H2", canonical_pair_id="p2")
    spec = {
        "policy_version": subject.POLICY,
        "policy_status": "USER_APPROVED",
        "application_provenance": subject.PROVENANCE,
        "reference_changed": False,
        "runtime_changed": False,
        "supported": [
            {
                "review_id": "H1",
                "product_side": "B",
                "policy_start": "退货政策:",
                "policy_witness": "14 天内退款",
                "product_witness": "商品 K7",
                "note": "商品保留完整政策。",
            }
        ],
        "deferred": [{"review_id": "H2", "status": "RECHECK_POLICY_COMPLETENESS", "note": "短条款完整性待核。"}],
    }
    return spec, [row, other], {"p1": payload, "p2": deepcopy(payload)}


def test_policy_itself_is_x_without_product_record_on_other_side():
    spec, rows, payloads = fixture()
    before = deepcopy((spec, rows, payloads))
    supported, deferred = subject.apply_cases(spec, rows, payloads)
    r = supported[0]
    assert r["proposed_primary_not_gold"] == {
        "a_can_replace_b": "NO",
        "b_can_replace_a": "YES",
        "same_duplicate_group": "YES",
    }
    assert r["expected_shared_x_not_gold"] == "PRESENT"
    assert r["old_reference"]["same_duplicate_group"] == "NO"
    assert r["old_reference_differs_from_proposal"]
    assert r["independently_adjudicated"] is False
    assert len(r["evidence"]) == 3
    for evidence in r["full_policy_evidence"]:
        text = payloads["p1"][f"document_{evidence['side'].lower()}"]["text"]
        assert text.encode()[evidence["start_byte_utf8"] : evidence["end_byte_utf8"]].decode() == evidence["quote"]
    assert deferred[0]["proposed_primary_not_gold"] is None
    assert (spec, rows, payloads) == before


def test_mirrored_application_swaps_replacement_directions():
    spec, rows, payloads = fixture()
    p = payloads["p1"]
    p["document_a"], p["document_b"] = p["document_b"], p["document_a"]
    rows[0]["payload_sha256"] = sha256_json(p)
    spec["supported"][0]["product_side"] = "A"
    supported, _ = subject.apply_cases(spec, rows, payloads)
    assert supported[0]["proposed_primary_not_gold"]["a_can_replace_b"] == "YES"
    assert supported[0]["proposed_primary_not_gold"]["b_can_replace_a"] == "NO"


@pytest.mark.parametrize(
    ("field", "value"),
    [("runtime_changed", True), ("reference_changed", True), ("application_provenance", "HUMAN_GOLD")],
)
def test_policy_approval_cannot_be_promoted_to_runtime_or_gold(field, value):
    spec, rows, payloads = fixture()
    spec[field] = value
    with pytest.raises(DedupEvaluationError, match="POLICY_ONLY_SCOPE"):
        subject.apply_cases(spec, rows, payloads)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("product_side", "X"),
        ("policy_start", "missing"),
        ("policy_witness", "missing"),
        ("product_witness", "退货政策"),
        ("product_witness", "not in product"),
        ("review_id", "unknown"),
    ],
)
def test_invalid_application_witness_or_direction_is_rejected(field, value):
    spec, rows, payloads = fixture()
    spec["supported"][0][field] = value
    with pytest.raises(DedupEvaluationError):
        subject.apply_cases(spec, rows, payloads)


@pytest.mark.parametrize("mutation", ["changed_input", "changed_policy", "truncated", "duplicate", "deferred_decided"])
def test_incomplete_or_conflicting_application_cannot_be_frozen(mutation):
    spec, rows, payloads = fixture()
    if mutation == "changed_input":
        payloads["p1"]["document_b"]["text"] += " changed"
    elif mutation == "changed_policy":
        payloads["p1"]["document_b"]["text"] = payloads["p1"]["document_b"]["text"].replace("14 天", "7 天")
        rows[0]["payload_sha256"] = sha256_json(payloads["p1"])
    elif mutation == "truncated":
        payloads["p1"]["long_document_evidence"]["truncated"] = True
        rows[0]["payload_sha256"] = sha256_json(payloads["p1"])
    elif mutation == "duplicate":
        spec["deferred"][0]["review_id"] = "H1"
    else:
        spec["deferred"][0]["status"] = "NO_NO"
    with pytest.raises(DedupEvaluationError):
        subject.apply_cases(spec, rows, payloads)


def test_policy_overlay_retains_panel_and_flags_without_filling_independent_gold():
    spec, rows, payloads = fixture()
    supported, deferred = subject.apply_cases(spec, rows, payloads)
    panel = [
        {
            **r,
            "role": "old_role",
            "routing": "old_routing",
            "reference_review_flags": ["OLD_DISPUTE"],
            "mandatory_supported_prior_repair": True,
        }
        for r in rows
    ]
    before = deepcopy(panel)
    result = subject.panel_overlay(panel, supported, deferred)
    assert panel == before
    assert [r["canonical_pair_id"] for r in result] == [r["canonical_pair_id"] for r in panel]
    assert result[1]["current_negative_guard_candidate"] is False
    assert result[0]["current_negative_guard_candidate"] is False
    assert all(r["independent_shared_x_gold"] is None for r in result)
    assert result[0]["historical_review_flags"] == ["OLD_DISPUTE"]


def test_checked_in_synthetic_materials_validate_real_v3_and_mirror_contract():
    spec = json.loads((subject.HERE / "policy_only_review_v1.json").read_text())
    packets, expectations = subject.regression_packets(spec["examples"])
    index = {r["canonical_pair_id"]: r["expected"] for r in expectations}
    assert len(packets) == len(spec["examples"]) * 2
    assert index["complete_policy_plus_product:ab"]["a_can_replace_b"] == "YES"
    assert index["complete_policy_plus_product:ba"]["b_can_replace_a"] == "YES"
    for name in (
        "same_policy_distinct_products",
        "product_overrides_policy",
        "policy_obligor_changes",
        "cookie_ui_not_full_policy",
    ):
        assert index[f"{name}:ab"]["same_duplicate_group"] == "NO"
    assert index["policy_truncated:ab"]["same_duplicate_group"] == "UNRESOLVED"
    assert index["equivalent_cookie_ui:ab"]["a_can_replace_b"] == "YES"
