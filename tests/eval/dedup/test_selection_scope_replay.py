# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis.selection_scope_replay import matched_selections, probe_scope
from eval.dedup.judging.coverage_selection import adapt_selection
from eval.dedup.judging.coverage_witness import COLUMN
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import DedupEvaluationError, sha256_json, write_text_atomic
from tests.eval.dedup.test_coverage_selection import selection
from tests.eval.dedup.test_coverage_witness import _case, _uncover


@pytest.mark.parametrize("non_main", [True, False])
def test_bilateral_coverage_does_not_require_agreement_on_main_content_taxonomy(non_main):
    main, value, payload = _case("Cookies need consent.", "Cookie 需要获得同意。", non_main=non_main)
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD" if non_main else "NON_MAIN_MESSAGES"
    new = selection(value)
    snapshot = deepcopy((main, new, payload))
    before, after, audit = probe_scope(main, new, payload)
    assert before == adapt_selection(main, new, payload)
    assert before["same_duplicate_group"] == "UNRESOLVED"
    assert after["a_can_replace_b"] == after["b_can_replace_a"] == "YES"
    assert after["material_difference"] == "NONE"
    assert after["confidence_tier"] == "MEDIUM"
    assert audit["action"] == "BILATERAL_COVERAGE_INDEPENDENT_OF_TAXONOMY"
    assert {e["side"] for e in after["evidence"]} == {"A", "B"}
    validate_evidence_offsets(after, payload)
    assert (main, new, payload) == snapshot


@pytest.mark.parametrize("non_main", [True, False])
@pytest.mark.parametrize(
    "kind", ["POLICY_MEANING", "RECORD_IDENTITY", "PAGE_ROLE", "STATE_VERSION", "MEMBERSHIP", "OTHER_RETAINED"]
)
def test_retained_loss_is_decisive_despite_profile_disagreement(non_main, kind):
    main, value, payload = _case(non_main=non_main)
    _uncover(value, payload, kind=kind)
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD" if non_main else "NON_MAIN_MESSAGES"
    before, after, audit = probe_scope(main, selection(value), payload)
    assert before["same_duplicate_group"] == "UNRESOLVED"
    assert after["a_can_replace_b"] == after["b_can_replace_a"] == "NO"
    assert after["material_difference"] == "MAJOR"
    assert (after["relation_type"] == "VERSION_RELATED") == (kind == "STATE_VERSION")
    assert audit["action"] == "RETAINED_LOSS_INDEPENDENT_OF_TAXONOMY"
    validate_evidence_offsets(after, payload)


@pytest.mark.parametrize("non_main", [True, False])
def test_main_content_addition_still_needs_consistent_substantive_scope(non_main):
    main, value, payload = _case(non_main=non_main)
    _uncover(value, payload)
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD" if non_main else "NON_MAIN_MESSAGES"
    before, after, audit = probe_scope(main, selection(value), payload)
    assert before == after
    assert after["same_duplicate_group"] == "UNRESOLVED"
    assert audit["action"] == "MAIN_CONTENT_SCOPE_CONFLICT_STILL_UNRESOLVED"


@pytest.mark.parametrize("non_main", [True, False])
def test_agreed_scope_keeps_existing_equivalence_or_extension(non_main):
    main, value, payload = _case(non_main=non_main)
    _uncover(value, payload)
    before, after, audit = probe_scope(main, selection(value), payload)
    assert after == before
    assert after["relation_type"] == ("RELATED_NON_DUPLICATE" if non_main else "CONTAINMENT")
    assert audit["action"] == "UNCHANGED"


@pytest.mark.parametrize("basis", ["none", "unresolved"])
def test_negative_and_unknown_main_cannot_be_reopened(basis):
    main, value, payload = _case(non_main=True)
    main["span_shared_basis"]["score"] = basis
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD"
    before, after, audit = probe_scope(main, selection(value), payload)
    assert before == after
    assert after["same_duplicate_group"] == ("NO" if basis == "none" else "UNRESOLVED")
    assert not audit["checks"]["critic_owned_positive_main"]


def test_unbound_scope_never_becomes_bilateral_equivalence():
    main, value, payload = _case(non_main=True)
    value["record_scope"] = "DISTINCT_OR_UNBOUND_RECORDS"
    before, after, audit = probe_scope(main, selection(value), payload)
    assert after == before
    assert after["same_duplicate_group"] == "UNRESOLVED"
    assert audit["action"] == "UNCHANGED"


def test_exact_is_owned_even_when_critic_taxonomy_disagrees():
    main, value, payload = _case("Same text.", "Same text.", non_main=True)
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD"
    before, after, audit = probe_scope(main, selection(value), payload)
    assert after == before
    assert after["relation_type"] == "EXACT"
    assert audit["action"] == "UNCHANGED"


@pytest.mark.parametrize("truncated", [False, True])
def test_genuine_unresolved_is_not_reinterpreted_as_a_taxonomy_problem(truncated):
    main, value, payload = _case(non_main=True)
    value.update(input_status="UNRESOLVED", record_scope="UNRESOLVED", anchor_a_ids="", anchor_b_ids="")
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        value[field].update(status="UNRESOLVED", coverage_mode="NOT_COVERED", coverage_counterpart_ids="")
    payload["long_document_evidence"]["truncated"] = truncated
    before, after, audit = probe_scope(main, selection(value), payload)
    assert after == before
    assert after["same_duplicate_group"] == "UNRESOLVED"
    assert after["confidence_tier"] == "LOW"
    assert audit["action"] == "UNCHANGED"


def test_scope_probe_does_not_claim_to_prove_model_entailment():
    main, value, payload = _case("Alice | private profile", "Bob | private profile", non_main=True)
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD"
    for field in ("a_meaning_in_b", "b_meaning_in_a"):
        value[field].update(coverage_mode="HARMLESS_ONLY", coverage_counterpart_ids="")
    _, after, _ = probe_scope(main, selection(value), payload)
    assert after["same_duplicate_group"] == "YES"
    assert after["material_difference"] == "MINOR"
    # Grounded spans cannot detect a model incorrectly calling actual identity differences harmless.
    assert any(code.startswith("OFFLINE_SCOPE_POLICY:") for code in after["reason_codes"])


def test_bad_evidence_is_rejected_not_repaired():
    main, value, payload = _case(non_main=True)
    _uncover(value, payload, kind="POLICY_MEANING")
    value["record_scope"] = "SAME_SUBSTANTIVE_RECORD"
    new = selection(value)
    new["b_meaning_in_a"]["source_span_id"] = "A999"
    with pytest.raises(DedupEvaluationError):
        probe_scope(main, new, payload)


@pytest.mark.parametrize("corruption", [None, "attempts", "raw_output_sha256", "selection_response_sha256"])
def test_only_published_attempt_and_complete_raw_digest_can_supply_the_probe(tmp_path, corruption):
    _, value, _ = _case()
    new = selection(value)
    old = {**new, "scope_explanation": "Earlier response, not selected."}
    first = {"canonical_pair_id": "p", COLUMN: old}
    accepted = {"canonical_pair_id": "p", COLUMN: new}
    for number, row in ((1, first), (2, accepted)):
        write_text_atomic(tmp_path / f"attempt_{number:02d}/output/rows.jsonl", json.dumps(row) + "\n")
    prediction = {
        "canonical_pair_id": "p",
        "critic_request_status": "REQUESTED",
        "attempts": 2,
        "raw_output_sha256": sha256_json(accepted),
        "selection_response_sha256": sha256_json(new),
    }
    owned = {"canonical_pair_id": "owned", "critic_request_status": "NOT_REQUESTED_OWNED_BRANCH"}
    if corruption:
        prediction[corruption] = 1 if corruption == "attempts" else "bad"
        with pytest.raises(DedupEvaluationError):
            matched_selections(tmp_path, [prediction, owned])
    else:
        assert matched_selections(tmp_path, [prediction, owned]) == {"p": new}
