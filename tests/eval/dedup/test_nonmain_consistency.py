# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis.judge_calibration import evaluate_predictions
from eval.dedup.analysis.nonmain_consistency import matched_critics, probe_nonmain_equivalence
from eval.dedup.judging.local_ndd import RECORD_BINDING_CRITIC_COLUMN, adapt_ndd_judge_output
from eval.dedup.judging.payload import _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.judging.schema_v3 import JUDGE_SCHEMA_V3, validate_judge_output_v3
from eval.dedup.validation import DedupEvaluationError, sha256_json, write_text_atomic


def _case(text_a="Cookies need consent.", text_b="Cookies need consent. Analytics require prior consent."):
    packet = _semantic_diff_packet(text_a, text_b, truncated=False)
    payload = {
        "document_a": {"text": text_a},
        "document_b": {"text": text_b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": packet,
    }
    citations = " ".join(span["span_id"] for span in packet["spans"])
    scores = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "cookie_consent",
        "primary_risk_factor": "template_slot_collision",
        "confidence_tier": "medium",
        "span_content_profile_a": "non_main_only",
        "span_content_profile_b": "non_main_only",
        "span_shared_basis": "verified_equivalent_non_main_message",
        "span_a_delta": "universal_ui_or_repetition" if packet["span_counts"]["A_ONLY"] else "none",
        "span_b_delta": "universal_ui_or_repetition" if packet["span_counts"]["B_ONLY"] else "none",
        "span_hard_conflict": "none",
        "span_translation_status": "not_translation",
    }
    main = {key: {"score": value, "reasoning": citations} for key, value in scores.items()}
    critic = {
        key: {"score": value, "reasoning": citations}
        for key, value in {
            "record_scope": "same_specific_record",
            "record_binding_verdict": "atomic_same_record_extension",
            "retained_conflict": "none",
        }.items()
    }
    return main, critic, payload


def test_supported_conflict_abstains_without_inventing_direction_or_modifying_inputs():
    main, critic, payload = _case()
    snapshot = deepcopy((main, critic, payload))
    before, after, diagnostic = probe_nonmain_equivalence(main, critic, payload)
    assert before == adapt_ndd_judge_output(
        main, JUDGE_SCHEMA_V3, payload=payload, record_binding_critic=critic, record_binding_policy="v8"
    )
    assert before["a_can_replace_b"] == before["b_can_replace_a"] == "YES"
    assert diagnostic["triggered"]
    assert all(diagnostic["checks"].values())
    assert after["a_can_replace_b"] == after["b_can_replace_a"] == after["same_duplicate_group"] == "UNRESOLVED"
    assert after["confidence_tier"] == "LOW"
    assert before["evidence"]
    assert not after["evidence"]
    validate_judge_output_v3(after)
    validate_evidence_offsets(before, payload)
    assert (main, critic, payload) == snapshot


@pytest.mark.parametrize("score", ["benign_non_record_delta", "unresolved", "not_applicable"])
def test_benign_or_missing_extension_proof_keeps_native_output(score):
    main, critic, payload = _case()
    critic["record_binding_verdict"]["score"] = score
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert not check["triggered"]


@pytest.mark.parametrize("field", ["record_binding_verdict", "record_scope", "retained_conflict"])
def test_no_cross_field_citation_borrowing(field):
    main, critic, payload = _case()
    critic[field]["reasoning"] = "S001" if field != "retained_conflict" else "A002-A001"
    if field == "record_scope":
        critic[field]["reasoning"] = "B001"
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert not check["triggered"]


@pytest.mark.parametrize("scope", ["equivalent_complete_message", "generic_context_only", "unresolved"])
def test_unbound_scope_does_not_manufacture_extension_proof(scope):
    main, critic, payload = _case()
    critic["record_scope"]["score"] = scope
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert not check["triggered"]


@pytest.mark.parametrize("boundary", ["truncated", "incomplete", "main_unresolved", "substantive", "containment"])
def test_existing_owned_boundaries_are_not_reopened(boundary):
    main, critic, payload = _case()
    if boundary == "truncated":
        payload["long_document_evidence"]["truncated"] = True
    elif boundary == "incomplete":
        payload["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    elif boundary == "main_unresolved":
        main["span_shared_basis"]["score"] = "unresolved"
    else:
        for side in ("a", "b"):
            main[f"span_content_profile_{side}"]["score"] = "substantive_main"
        main["span_shared_basis"]["score"] = "verified_substantive_record"
        if boundary == "containment":
            main["span_b_delta"]["score"] = "same_record_content_extension"
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert not check["triggered"]


def test_supported_permission_veto_stays_negative():
    main, critic, payload = _case()
    critic["record_binding_verdict"]["score"] = "non_main_policy_or_state_change"
    critic["retained_conflict"]["score"] = "policy_permission_change"
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert after["same_duplicate_group"] == "NO"
    assert not check["triggered"]


def test_exact_identity_remains_protected_even_with_atomic_score():
    main, critic, payload = _case("Identical cookie message.", "Identical cookie message.")
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert after["relation_type"] == "EXACT"
    assert not check["triggered"]


def test_translation_and_unicode_are_not_conflicts_without_extension_verdict():
    main, critic, payload = _case("Cookies need consent.", "需要同意才能使用 Cookie。")
    main["span_translation_status"]["score"] = "complete_faithful"
    critic["record_binding_verdict"]["score"] = "benign_non_record_delta"
    before, after, check = probe_nonmain_equivalence(main, critic, payload)
    assert after == before
    assert after["same_duplicate_group"] == "YES"
    assert not check["triggered"]
    validate_evidence_offsets(after, payload)


def test_bad_alignment_fails_even_if_public_evidence_uses_other_valid_spans():
    main, critic, payload = _case()
    payload["semantic_diff_evidence"]["spans"][-1]["text"] = "invented extension"
    with pytest.raises(DedupEvaluationError):
        probe_nonmain_equivalence(main, critic, payload)


def test_missing_real_critic_contract_cannot_be_backfilled():
    main, critic, payload = _case()
    critic.pop("record_scope")
    with pytest.raises(DedupEvaluationError, match="record-scope"):
        probe_nonmain_equivalence(main, critic, payload)


def test_only_published_digest_is_selected_not_first_last_or_most_favorable(tmp_path):
    _, critic, _ = _case()
    first, last = deepcopy(critic), deepcopy(critic)
    first["record_binding_verdict"]["score"] = "benign_non_record_delta"
    last["record_binding_verdict"]["score"] = "unresolved"
    path = tmp_path / "repeat_1_control/attempt_0/output/result.jsonl"
    write_text_atomic(
        path,
        "".join(
            json.dumps({"canonical_pair_id": "p", RECORD_BINDING_CRITIC_COLUMN: c}) + "\n"
            for c in (first, critic, last)
        ),
    )
    rows = [{"canonical_pair_id": "p", "critic_response_sha256": sha256_json(critic)}]
    assert matched_critics(tmp_path, "repeat_1_control", rows) == {"p": critic}
    rows[0]["critic_response_sha256"] = "missing"
    with pytest.raises(DedupEvaluationError, match="no raw match"):
        matched_critics(tmp_path, "repeat_1_control", rows)


def test_error_to_abstention_is_not_counted_as_a_correct_decision():
    main, critic, payload = _case()
    before, after, _ = probe_nonmain_equivalence(main, critic, payload)
    labels = [
        {
            "canonical_pair_id": pid,
            "human_same_duplicate_group": value,
            "human_a_can_replace_b": value,
            "human_b_can_replace_a": value,
            "human_relation_type": "NEAR_SURFACE" if value == "YES" else "RELATED_NON_DUPLICATE",
            "human_material_difference": "NONE" if value == "YES" else "MAJOR",
            "human_reason_code": "boilerplate",
            "stratum_population_n": "100",
            "stratum_sample_n": "10",
        }
        for pid, value in (("positive", "YES"), ("negative", "NO"))
    ]
    old = evaluate_predictions(labels, [{"canonical_pair_id": row["canonical_pair_id"], **before} for row in labels])
    new = evaluate_predictions(labels, [{"canonical_pair_id": row["canonical_pair_id"], **after} for row in labels])
    assert old["weighted"]["primary_decision_exact"] == 0.5
    assert new["weighted"]["primary_decision_exact"] == 0
    assert old["over_group"] == 1
    assert new["over_group"] == 0
    assert old["under_group"] == 0
    assert new["under_group"] == 1
    assert new["weighted"]["duplicate_recall"] == 0
