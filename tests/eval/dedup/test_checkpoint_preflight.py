# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import checkpoint_preflight as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, write_json_atomic


def fixture(pid="one", a="Returns are accepted.\n\nPostage is prepaid.", b="Returns are accepted."):
    payload = subject.synthetic_payload(a, b)
    raw = subject.synthetic_raw(payload)
    public = subject.main_adapter.main_decision(raw, payload)
    row = {
        "canonical_pair_id": pid,
        "review_id": pid.upper(),
        "payload": payload,
        "raw_main": raw,
        "main_public": public,
    }
    pred = dict(canonical_pair_id=pid, **public)
    ref = {
        "canonical_pair_id": pid,
        "weight": "2",
        "scoring_eligibility": "INCLUDED",
        "reference_provenance": "INHERITED_REFERENCE_NOT_REVALIDATED",
        "revised_reference": json.dumps(subject.primary(public)),
        "revised_errors": json.dumps({subject.VERSION: "CORRECT"}),
    }
    return row, pred, ref


def unresolved_fixture(kind):
    row, pred, ref = fixture()
    payload, raw = row["payload"], row["raw_main"]
    if kind == "TRUNCATED_NO_SPAN_INPUT":
        payload["document_a"]["text"] = payload["document_b"]["text"] = None
        payload["long_document_evidence"]["truncated"] = True
        payload["semantic_diff_evidence"] = subject._semantic_diff_packet(None, None, truncated=True)
        for key in raw:
            raw[key] = {
                "score": "unreadable"
                if key.startswith("span_content_profile")
                else "low"
                if key == "confidence_tier"
                else "extraction_or_payload_limit"
                if key == "primary_risk_factor"
                else "unresolved",
                "reasoning": "Input unavailable",
            }
    elif kind == "DIFF_PACKET_LIMIT":
        payload["semantic_diff_evidence"]["status"] = "INCOMPLETE_LIMIT"
    elif kind == "INVALID_SPAN_CITATION":
        raw["span_a_delta"]["reasoning"] += " B999"
    else:
        raw["span_shared_basis"]["score"] = "unresolved"
    row["main_public"] = subject.main_adapter.main_decision(raw, payload)
    pred.update(row["main_public"])
    ref["revised_errors"] = json.dumps({subject.VERSION: "UNRESOLVED"})
    return row, pred, ref


@pytest.mark.parametrize(
    "category",
    ["TRUNCATED_NO_SPAN_INPUT", "DIFF_PACKET_LIMIT", "INVALID_SPAN_CITATION", "AUXILIARY_LEDGER_UNRESOLVED"],
)
def test_attribution_separates_input_loss_citation_and_auxiliary_uncertainty(category):
    row, pred, ref = unresolved_fixture(category)
    before = deepcopy((row, pred, ref))
    output = subject.unresolved_inventory([row], [pred], [ref])
    assert len(output) == 1
    assert output[0]["observed_failure_category"] == category
    assert output[0]["final_primary"]["same_duplicate_group"] == "UNRESOLVED"
    assert (row, pred, ref) == before
    if category == "INVALID_SPAN_CITATION":
        assert output[0]["unknown_citations"] == ["B999"]
    if category == "AUXILIARY_LEDGER_UNRESOLVED":
        assert output[0]["unresolved_ledger_fields"] == ["span_shared_basis"]
        assert output[0]["raw_primary_not_validated_verdict"]["same_duplicate_group"] == "YES"


def test_comparison_exclusion_does_not_remove_engineering_case_or_count_its_weight_as_error():
    row, pred, ref = unresolved_fixture("DIFF_PACKET_LIMIT")
    ref["scoring_eligibility"] = "EXCLUDED"
    other, _, other_ref = fixture("other")
    inventory = subject.unresolved_inventory(
        [row, other], [pred, dict(canonical_pair_id="other", **other["main_public"])], [ref, other_ref]
    )
    assert len(inventory) == 1
    summary = subject.unresolved_summary(inventory, [ref, other_ref])["categories"]["DIFF_PACKET_LIMIT"]
    assert summary["count"] == 1
    assert summary["reference_errors"] == summary["reference_error_weight"] == summary["primary_exact_gap_pp"] == 0


def test_unresolved_gap_mass_uses_full_scored_denominator_and_is_not_a_precision_gain():
    row, pred, ref = unresolved_fixture("DIFF_PACKET_LIMIT")
    _, _, other = fixture("other")
    other["weight"] = "6"
    inventory = subject.unresolved_inventory([row], [pred], [ref])
    summary = subject.unresolved_summary(inventory, [ref, other])
    assert summary["categories"]["DIFF_PACKET_LIMIT"]["primary_exact_gap_pp"] == 25
    assert summary["categories"]["DIFF_PACKET_LIMIT"]["recall_gap_pp"] == 25
    assert "NOT achievable repair gains" in summary["interpretation"]


@pytest.mark.parametrize("issue", ["missing", "duplicate", "extra"])
def test_audit_rejects_partial_or_ambiguous_population_joins(issue):
    row, pred, ref = fixture()
    final = (
        []
        if issue == "missing"
        else [pred, deepcopy(pred)]
        if issue == "duplicate"
        else [pred, {**pred, "canonical_pair_id": "extra"}]
    )
    with pytest.raises(DedupEvaluationError):
        subject.unresolved_inventory([row], final, [ref])


def test_exact_pair_reversal_is_not_a_direction_disagreement_and_prompts_are_bound():
    a, b = "Returns accepted.\n\nPostage prepaid.", "Returns accepted."
    left, pred_left, _ = fixture("left", a, b)
    right, _, _ = fixture("right", b, a)
    right["raw_main"] = subject.synthetic_raw(right["payload"], deltas=("none", "same_record_content_extension"))
    right["raw_main"]["a_can_replace_b"]["score"] = "no"
    right["raw_main"]["b_can_replace_a"]["score"] = "yes"
    right["main_public"] = subject.main_adapter.main_decision(right["raw_main"], right["payload"])
    pred_right = {"canonical_pair_id": "right", **right["main_public"]}
    rows = [left, right]
    result = subject.exact_input_groups(rows, rows, [pred_left, pred_right], "{{ payload.semantic_diff_evidence }}")
    assert len(result) == 1
    assert not any(result[0]["stage_disagreement"].values())
    assert result[0]["distinct_oriented_payloads"] == 2
    assert len({r["initial_pair_prompt_sha256"] for r in result[0]["cases"]}) == 2


def test_exact_same_payload_stage_trace_distinguishes_adapter_from_critic_loss():
    left, pred, _ = fixture("left")
    right = deepcopy(left)
    right.update(canonical_pair_id="right", review_id="RIGHT")
    right["raw_main"]["span_content_profile_b"]["score"] = "non_main_only"
    right["main_public"] = subject.main_adapter.main_decision(right["raw_main"], right["payload"])
    rows = [left, right]
    result = subject.exact_input_groups(
        rows,
        rows,
        [pred, dict(canonical_pair_id="right", **right["main_public"])],
        "{{ payload.semantic_diff_evidence }}",
    )
    group = result[0]
    assert group["distinct_oriented_payloads"] == 1
    assert not group["stage_disagreement"]["raw_directions"]
    assert group["stage_disagreement"]["adapted_main"]
    assert group["stage_disagreement"]["final"]
    assert not any(r["coverage_changed_primary"] or r["subject_verification_changed_primary"] for r in group["cases"])


def test_exact_grouping_does_not_promote_format_similar_empty_or_truncated_pairs():
    row, pred, _ = fixture("original")
    rows, finals = [row], [pred]
    for name in ("format", "empty", "truncated"):
        other = deepcopy(row)
        other.update(canonical_pair_id=name, review_id=name)
        if name == "format":
            other["payload"]["document_a"]["text"] += " "
        elif name == "empty":
            other["payload"]["document_a"]["text"] = ""
        else:
            other["payload"]["long_document_evidence"]["truncated"] = True
        rows.append(other)
        finals.append(dict(canonical_pair_id=name, **other["main_public"]))
    assert subject.exact_input_groups(rows, rows, finals, "{{ payload.semantic_diff_evidence }}") == []


def test_legacy_policy_probes_reproduce_conflicts_without_relaxing_protection_or_schema():
    probes = {r["name"]: r for r in subject.contract_probes()}
    assert not probes["policy_plus_product"]["current_policy_compatible"]
    assert probes["policy_plus_product"]["actual_primary"]["same_duplicate_group"] == "NO"
    assert not probes["independent_article_addition"]["current_policy_compatible"]
    assert probes["proper_subset_with_navigation"]["actual_primary"]["b_can_replace_a"] == "YES"
    assert not probes["minor_nonmain_containment_schema"]["current_policy_compatible"]
    assert "containment" in probes["minor_nonmain_containment_schema"]["error_message"]
    assert all(
        probes[k]["current_policy_compatible"]
        for k in ("true_substantive_extension", "incompatible_values", "two_sided_uncovered")
    )


def test_final_bypass_replay_validates_full_result_and_rejects_projection_drift(tmp_path):
    row, _, _ = fixture()
    row["raw_main"]["span_content_profile_b"]["score"] = "non_main_only"
    row["main_public"] = subject.main_adapter.main_decision(row["raw_main"], row["payload"])
    write_json_atomic(tmp_path / "panel_private.json", [row])
    write_json_atomic(tmp_path / "assessment.json", {"response_artifacts": {}})
    write_json_atomic(tmp_path / "complete.json", {"assessment_sha256": sha256_file(tmp_path / "assessment.json")})
    write_json_atomic(tmp_path / "requests_frozen.json", [])
    expected = [{"canonical_pair_id": row["canonical_pair_id"], **subject.primary(row["main_public"])}]
    results, _ = subject.replay_final(tmp_path, expected)
    assert results == [{"canonical_pair_id": row["canonical_pair_id"], **row["main_public"]}]
    expected[0]["same_duplicate_group"] = "YES"
    with pytest.raises(DedupEvaluationError, match="PREFLIGHT_FINAL_REPLAY"):
        subject.replay_final(tmp_path, expected)


def test_preflight_refuses_to_reuse_an_existing_output_root(tmp_path):
    with pytest.raises(DedupEvaluationError, match="PREFLIGHT_OUTPUT"):
        subject.run(tmp_path, tmp_path / "missing", tmp_path / "missing")
