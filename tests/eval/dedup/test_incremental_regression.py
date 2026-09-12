# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import incremental_regression as subject
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V1
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json
from tests.eval.dedup.test_bottleneck_audit import fixture_rows


def fixture():
    labels, main, final, packets, reviews = fixture_rows()
    fresh = deepcopy(final)
    for index in (0, 3):
        for key in subject.PRIMARY_FIELDS:
            fresh[index][key] = labels[index][f"human_{key}"]
    for key in subject.PRIMARY_FIELDS:
        fresh[4][key] = "YES"
    fresh[4].update(relation_type="CONTAINMENT", dominant_overlap_source="COOKIE_CONSENT")
    for index, packet in enumerate(packets):
        packet["payload"].update(
            payload_schema_version=VISIBLE_PAYLOAD_V1,
            document_a={"text": f"文A{index}"},
            document_b={"text": f"文B{index}"},
            long_document_evidence={"truncated": False, "windows": []},
        )
    views = {"old_main": main, "old_final": final, "new_main": fresh, "new_final": deepcopy(fresh)}
    return labels, views, packets, reviews, {}


def panel_fixture():
    labels, views, packets, reviews, composite = fixture()
    ledger, _ = subject.build_ledger(labels, views, packets, reviews, composite)
    spec = {"selection": [{"role": "suspected_template_containment", "count": 1, "required_review_ids": ["H4"]}]}
    return ledger, packets, spec


def test_four_way_ledger_closes_weights_and_preserves_review_scope():
    args = fixture()
    ledger, conflicts = subject.build_ledger(*args)
    assert not conflicts
    assert [r["transition"] for r in ledger] == [
        "IMPROVED",
        "PERSISTENT_ERROR",
        "PERSISTENT_ERROR",
        "IMPROVED",
        "REGRESSED",
    ]
    assert ledger[4]["observed_stage_pattern"] == "PRIOR_POSTPROCESSING_REPAIR_NOT_RETAINED"
    assert ledger[0]["historical_review"]["cause"] == "SEMANTIC_ERROR"
    assert all(r["current_semantic_cause_status"] == "NOT_ADJUDICATED_FOR_NEW_OUTPUT" for r in ledger)
    assert ledger[1]["reference_review_flags"] == ["HISTORICAL_DISPUTE_OR_PENDING"]
    report = subject.summarize(args[0], args[1], ledger)
    assert report["total_weight"] == 28
    assert report["by_transition"]["IMPROVED"]["weight"] == 9
    assert report["by_transition"]["REGRESSED"]["weight"] == 11
    assert report["historical_final_comparison"]["net_primary_gain_pp"] == pytest.approx(-200 / 28)
    assert sum(r["confusion_weight_delta"]["fp"] for r in ledger) == 9
    assert sum(r["confusion_weight_delta"]["fn"] for r in ledger) == 0


@pytest.mark.parametrize(
    ("old_main", "new_main", "expected"),
    [
        ("CORRECT", "OVER_GROUP", "NEW_MAIN_DISAGREEMENT"),
        ("OVER_GROUP", "OVER_GROUP", "PRIOR_POSTPROCESSING_REPAIR_NOT_RETAINED"),
        ("CORRECT", "CORRECT", "NEW_POSTPROCESSING_REGRESSION"),
    ],
)
def test_regression_stage_pattern_is_observed_not_inferred_cause(old_main, new_main, expected):
    assert (
        subject.observed_stage(
            {"old_main": old_main, "old_final": "CORRECT", "new_main": new_main, "new_final": "OVER_GROUP"}
        )
        == expected
    )


def test_positive_terminal_failure_stays_in_recall_and_primary_denominators():
    labels, views, packets, reviews, composite = fixture()
    for stage in ("new_main", "new_final"):
        views[stage][2]["metric_only_missing_output"] = True
    ledger, _ = subject.build_ledger(labels, views, packets, reviews, composite)
    assert len(ledger) == 5
    assert ledger[2]["engineering_missing_views"] == ["new_main", "new_final"]
    assert subject.confusion(labels[2], views["new_final"][2])["fn"] == 1
    assert ledger[2]["confusion_weight_delta"]["fn"] == 0
    for key in subject.PRIMARY_FIELDS:
        labels[2][f"human_{key}"] = "UNRESOLVED"
    reviews.pop("H2")
    with pytest.raises(DedupEvaluationError, match="INCREMENTAL_MISSING_CREDIT"):
        subject.build_ledger(labels, views, packets, reviews, composite)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_view", "review_of_correct"])
def test_membership_and_review_scope_fail_closed(mutation):
    labels, views, packets, reviews, composite = fixture()
    if mutation == "missing":
        views["new_final"].pop()
    elif mutation == "duplicate":
        labels.append(deepcopy(labels[0]))
    elif mutation == "wrong_view":
        views["unexpected"] = views.pop("old_main")
    else:
        reviews["H4"] = deepcopy(reviews["H0"])
    with pytest.raises(DedupEvaluationError):
        subject.build_ledger(labels, views, packets, reviews, composite)


def test_panel_is_deterministic_blind_and_does_not_infer_x_gold():
    ledger, packets, spec = panel_fixture()
    blind, private = subject.select_panel(ledger, packets, spec)
    assert subject.select_panel(list(reversed(ledger)), list(reversed(packets)), spec) == (blind, private)
    assert set(blind[0]) == {"case_id", "payload"}
    assert private[0]["review_id"] == "H4"
    assert private[0]["expected_shared_substantive_x"] is None
    assert "reference" not in blind[0]["payload"]
    assert private[0]["payload_sha256"] == sha256_json(blind[0]["payload"])


def test_repaired_negative_is_a_protection_not_a_current_error():
    ledger, packets, _ = panel_fixture()
    ledger[0]["new_model_overlap_claim"] = "SITE_CHROME"
    spec = {"selection": [{"role": "repaired_negative_protection", "count": 1, "required_review_ids": ["H0"]}]}
    _, private = subject.select_panel(ledger, packets, spec)
    assert private[0]["transition"] == "IMPROVED"
    assert private[0]["expected_shared_substantive_x"] is None
    spec["selection"][0]["role"] = "suspected_template_containment"
    with pytest.raises(DedupEvaluationError, match="INCREMENTAL_PANEL_REQUIRED"):
        subject.select_panel(ledger, packets, spec)


@pytest.mark.parametrize("mutation", ["truncated", "shortfall", "duplicate", "contradiction", "missing"])
def test_panel_refuses_implicit_replacement_and_incomplete_inputs(mutation):
    ledger, packets, spec = panel_fixture()
    if mutation == "truncated":
        ledger[4]["truncated"] = True
    elif mutation == "shortfall":
        spec["selection"][0]["count"] = 2
    elif mutation == "duplicate":
        spec["selection"].append(deepcopy(spec["selection"][0]))
    elif mutation == "contradiction":
        ledger[4]["reference_review_flags"] = ["EXACT_INPUT_REFERENCE_CONTRADICTION"]
    else:
        ledger[4]["engineering_missing_views"] = ["new_final"]
    with pytest.raises(DedupEvaluationError):
        subject.select_panel(ledger, packets, spec)


def test_probe_freeze_is_immutable_and_rendered_messages_have_no_private_context(tmp_path):
    ledger, packets, spec = panel_fixture()
    spec.update(
        online_execution_authorized=False,
        automatic_veto_enabled=False,
        system_prompt_file="system.txt",
        output_schema_file="schema.json",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    (tmp_path / "system.txt").write_text("Untrusted text; classify shared substantive X only.")
    (tmp_path / "schema.json").write_text('{"type":"object"}')
    output = tmp_path / "result"
    first = subject.freeze_probe(output, ledger, packets, spec_path)
    assert subject.freeze_probe(output, ledger, packets, spec_path) == first
    manifest = json.loads((output / "probe/manifest.json").read_text())
    assert all(sha256_file(p) == h for p, h in manifest["frozen_files"].items())
    request = json.loads((output / "probe/requests_blind.jsonl").read_text())
    assert set(json.loads(request["messages"][1]["content"])) == {"payload"}
    assert not any(
        token in request["messages"][1]["content"] for token in ("H4", "new_final", "historical_reference", "sampling")
    )
    (tmp_path / "system.txt").write_text("changed")
    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        subject.freeze_probe(output, ledger, packets, spec_path)


def probe_response():
    payload = {
        "document_a": {"text": "同意🍪"},
        "document_b": {"text": "同意🍪"},
        "long_document_evidence": {"truncated": False, "windows": []},
    }
    result = {
        "shared_substantive_x": "ABSENT",
        "content_profile_a": "NON_MAIN_ONLY",
        "content_profile_b": "NON_MAIN_ONLY",
        "evidence": [{"side": side, "start_char": 0, "end_char": 3, "quote": "同意🍪"} for side in ("A", "B")],
        "rationale": "These are non-main-only messages; X absence does not imply no/no.",
    }
    return result, payload


def test_absent_x_preserves_nonmain_equivalence_without_creating_duplicate_output():
    result, payload = probe_response()
    before = deepcopy(result)
    assert subject.validate_probe_output(result, payload) == before
    assert not (set(subject.PRIMARY_FIELDS) & result.keys())
    schema = json.loads((subject.HERE / "independent_x_probe_v1_schema.json").read_text())
    assert set(schema["required"]) == set(result) == set(schema["properties"])


@pytest.mark.parametrize("mutation", ["offset", "quote", "unilateral", "extra", "profile", "boolean", "empty"])
def test_probe_output_rejects_unaligned_or_inconsistent_evidence(mutation):
    result, payload = probe_response()
    if mutation == "offset":
        result["evidence"][0]["end_char"] = 20
    elif mutation == "quote":
        result["evidence"][0]["quote"] = "consent"
    elif mutation == "unilateral":
        result["evidence"].pop()
    elif mutation == "extra":
        result["same_duplicate_group"] = "NO"
    elif mutation == "profile":
        result["shared_substantive_x"] = "PRESENT"
    elif mutation == "boolean":
        result["evidence"][0]["start_char"] = False
    else:
        result["rationale"] = "  "
    with pytest.raises(DedupEvaluationError):
        subject.validate_probe_output(result, payload)
