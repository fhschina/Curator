# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis.bottleneck_audit import blind_packets
from eval.dedup.analysis.composite_containment_review import SPEC, regression_packets, review_packet
from eval.dedup.validation import DedupEvaluationError, sha256_json


def fixtures():
    spec = json.loads(SPEC.read_text())
    packets, _ = regression_packets(spec["examples"][:1])
    packet = packets[0]
    packet["canonical_pair_id"] = "p1"
    positive = {"same_duplicate_group": "YES", "a_can_replace_b": "YES", "b_can_replace_a": "NO"}
    negative = dict.fromkeys(positive, "NO")
    ledger = [
        {
            "review_id": "H1",
            "canonical_pair_id": "p1",
            "cluster": "record_unit",
            "weight": "7.5",
            "payload_sha256": sha256_json(packet["payload"]),
            "reference": json.dumps(positive),
            "main": json.dumps(positive),
            "final": json.dumps(negative),
        }
    ]
    spec["cases"] = {
        "H1": {
            "status": "SUPPORTED_COMPOSITE",
            "direction": "A_REPLACES_B",
            "evidence_ids": ["S001", "A001"],
            "note": "Independent collection paragraph does not change the retained closure notice.",
        }
    }
    return ledger, [packet], spec


def test_review_preserves_gold_and_predictions_and_locates_bilateral_evidence():
    ledger, packets, spec = fixtures()
    original = deepcopy((ledger, packets, spec))
    report = review_packet(ledger, packets, spec)
    assert report["weights_by_status"] == {"SUPPORTED_COMPOSITE": 7.5}
    assert report["proposals_differing_from_reference"] == []
    assert not report["reference_changed"]
    assert not report["runtime_changed"]
    assert not report["new_scores_reported"]
    row = report["cases"][0]
    assert row["historical_12"]["main"] == row["proposed_primary_not_gold"]
    assert row["historical_12"]["final"]["same_duplicate_group"] == "NO"
    assert {e["side"] for e in row["evidence"]} == {"A", "B"}
    assert (ledger, packets, spec) == original


def test_pending_boundary_does_not_create_a_negative_or_positive_reference():
    ledger, packets, spec = fixtures()
    spec["cases"]["H1"].update(status="NONEMPTY_GATE_REVIEW", direction=None)
    report = review_packet(ledger, packets, spec)
    assert report["cases"][0]["proposed_primary_not_gold"] is None
    assert report["weights_by_status"] == {"NONEMPTY_GATE_REVIEW": 7.5}


def test_reference_disagreement_is_flagged_but_not_written_back():
    ledger, packets, spec = fixtures()
    ledger[0]["reference"] = ledger[0]["final"]
    report = review_packet(ledger, packets, spec)
    assert report["proposals_differing_from_reference"] == ["H1"]
    assert ledger[0]["reference"] == ledger[0]["final"]
    assert report["reference_changed"] is False


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing_case", "COMPOSITE_COVERAGE"),
        ("extra_case", "COMPOSITE_COVERAGE"),
        ("missing_payload", "COMPOSITE_PAYLOAD"),
        ("changed_text", "COMPOSITE_HASH"),
        ("bad_quote", "JUDGE_EVIDENCE_OFFSET_INVALID"),
        ("unknown_evidence", "COMPOSITE_EVIDENCE"),
        ("pending_direction", "COMPOSITE_PENDING"),
        ("invalid_direction", "COMPOSITE_DIRECTION"),
        ("no_anchor", "COMPOSITE_INPUT"),
        ("no_addition", "COMPOSITE_INPUT"),
        ("wrong_direction", "COMPOSITE_INPUT"),
        ("truncated", "COMPOSITE_INPUT"),
        ("promote_to_gold", "COMPOSITE_SCOPE"),
    ],
)
def test_fail_closed_review_materials(mutation, code):
    ledger, packets, spec = fixtures()
    review = spec["cases"]["H1"]
    payload = packets[0]["payload"]
    actions = {
        "missing_case": lambda: spec["cases"].clear(),
        "extra_case": lambda: spec["cases"].update(H2=deepcopy(review)),
        "missing_payload": packets.clear,
        "changed_text": lambda: payload["document_a"].update(text="Changed"),
        "bad_quote": lambda: payload["semantic_diff_evidence"]["spans"][0].update(a_text="invented quote"),
        "unknown_evidence": lambda: review.update(evidence_ids=["MISSING"]),
        "pending_direction": lambda: review.update(status="OTHER_BOUNDARY_REVIEW"),
        "invalid_direction": lambda: review.update(direction="BOTH"),
        "wrong_direction": lambda: review.update(direction="B_REPLACES_A"),
        "no_anchor": lambda: review.update(evidence_ids=["A001"]),
        "no_addition": lambda: review.update(evidence_ids=["S001"]),
        "truncated": lambda: payload["long_document_evidence"].update(truncated=True),
        "promote_to_gold": lambda: spec.update(reference_changed=True),
    }
    actions[mutation]()
    if mutation in {"bad_quote", "truncated"}:
        ledger[0]["payload_sha256"] = sha256_json(packets[0]["payload"])
    with pytest.raises(DedupEvaluationError, match=code):
        review_packet(ledger, packets, spec)


def test_all_declared_examples_have_valid_contracts_and_mirrored_directions():
    spec = json.loads(SPEC.read_text())
    packets, expectations = regression_packets(spec["examples"])
    originals = deepcopy(spec)
    by_id = {r["canonical_pair_id"]: r["expected"] for r in expectations}
    for example in spec["examples"]:
        a, b = (by_id[f"{example['id']}:{orientation}"] for orientation in ("ab", "ba"))
        assert a["a_can_replace_b"] == b["b_can_replace_a"]
        assert a["b_can_replace_a"] == b["a_can_replace_b"]
        assert a["relation_type"] == b["relation_type"]
    assert by_id["independent_addition:ab"]["a_can_replace_b"] == "YES"
    assert by_id["cookie_empty_anchor:ab"]["same_duplicate_group"] == "NO"
    assert by_id["chrome_only:ab"]["b_can_replace_a"] == "YES"
    assert by_id["faithful_translation:ab"]["material_difference"] == "NONE"
    assert by_id["truncated_unknown_remainder:ab"]["same_duplicate_group"] == "UNRESOLVED"
    blind, key = blind_packets(packets, set(by_id))
    assert len(blind) == len(key) == 2 * len(spec["examples"])
    assert all(set(r) == {"case_id", "payload"} for r in blind)
    assert {r["canonical_pair_id"] for r in key} == set(by_id)
    assert spec == originals


@pytest.mark.parametrize("mutation", ["containment_both_yes", "duplicate_id", "incomplete_as_resolved"])
def test_invalid_regression_expectations_are_not_silently_repaired(mutation):
    examples = json.loads(SPEC.read_text())["examples"][:1]
    if mutation == "containment_both_yes":
        examples[0]["directions"] = ["YES", "YES"]
    elif mutation == "duplicate_id":
        examples.append(deepcopy(examples[0]))
    else:
        examples[0]["truncated"] = True
    with pytest.raises(DedupEvaluationError):
        regression_packets(examples)
