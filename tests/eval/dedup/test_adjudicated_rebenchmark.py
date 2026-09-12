# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

from copy import deepcopy

import pytest

from eval.dedup.analysis import adjudicated_rebenchmark as subject
from eval.dedup.validation import DedupEvaluationError


def decision(group, a=None, b=None):
    return {"same_duplicate_group": group, "a_can_replace_b": a or group, "b_can_replace_a": b or group}


def fixture():
    labels = [
        dict(
            canonical_pair_id=pid,
            review_id=pid.upper(),
            sample_weight=weight,
            **{f"human_{k}": v for k, v in decision(group).items()},
            human_relation_type="HISTORICAL",
            human_material_difference="MINOR",
        )
        for pid, group, weight in [("a", "NO", 4), ("b", "YES", 2), ("c", "NO", 1)]
    ]
    predictions = [
        dict(canonical_pair_id=pid, **decision(group)) for pid, group in [("a", "YES"), ("b", "NO"), ("c", "YES")]
    ]
    updates = [
        {
            "canonical_pair_id": "a",
            "decision": decision("YES", "YES", "NO"),
            "provenance": "ASSISTANT_RULE_APPLICATION",
            "independent_blind_gold": False,
            "source": "review.json",
        }
    ]
    return labels, predictions, updates


def test_revision_keeps_weights_provenance_and_taxonomy_but_does_not_label_exclusions():
    labels, predictions, updates = fixture()
    before = deepcopy((labels, predictions, updates))
    revised, changes = subject.revise_reference(labels, updates, {"c"})
    assert len(changes) == 1
    assert changes[0]["weight"] == 4
    assert [subject._weight(r) for r in revised] == [4, 2, 1]
    assert revised[0]["adjudication_provenance"] == "ASSISTANT_RULE_APPLICATION"
    assert revised[1]["adjudication_provenance"] == "INHERITED_REFERENCE_NOT_REVALIDATED"
    assert revised[2]["adjudication_provenance"] == "USER_COMPARISON_EXCLUSION"
    assert all(revised[2][f"human_{k}"] is None for k in subject.PRIMARY_FIELDS)
    assert all(r["human_material_difference"] == "MINOR" for r in revised)
    assert all(r["independent_adjudication"] is False for r in revised)
    assert (labels, predictions, updates) == before


@pytest.mark.parametrize("issue", ["excluded", "unknown", "duplicate", "gold", "direction"])
def test_revision_rejects_invalid_or_misattributed_changes(issue):
    labels, _, updates = fixture()
    excluded = set()
    if issue == "excluded":
        excluded.add("a")
    elif issue == "unknown":
        updates[0]["canonical_pair_id"] = "missing"
    elif issue == "duplicate":
        updates.append(deepcopy(updates[0]))
    elif issue == "gold":
        updates[0]["independent_blind_gold"] = True
    else:
        updates[0]["decision"] = decision("NO", "YES", "NO")
    with pytest.raises(DedupEvaluationError):
        subject.revise_reference(labels, updates, excluded)


@pytest.mark.parametrize("issue", ["missing_excluded", "extra", "duplicate", "unknown_mask"])
def test_mask_checks_complete_membership_before_excluding(issue):
    labels, predictions, _ = fixture()
    excluded = {"c"}
    if issue == "missing_excluded":
        predictions.pop()
    elif issue == "extra":
        predictions.append(dict(canonical_pair_id="extra", **decision("NO")))
    elif issue == "duplicate":
        predictions.append(deepcopy(predictions[0]))
    else:
        excluded.add("unknown")
    with pytest.raises(DedupEvaluationError):
        subject.masked_score(labels, predictions, excluded)


def test_mask_does_not_credit_excluded_rows_and_preserves_engineering_population():
    labels, predictions, updates = fixture()
    predictions[2].update(decision("UNRESOLVED"), metric_only_missing_output=True)
    revised, _ = subject.revise_reference(labels, updates, {"c"})
    score = subject.masked_score(revised, predictions, {"c"})
    assert score["original_rows"] == 3
    assert score["scored_rows"] == 2
    assert score["weighted"]["weight_total"] == 6
    assert score["weighted"]["duplicate_precision"] == 1
    assert score["weighted"]["duplicate_recall"] == pytest.approx(4 / 6)
    assert score["weighted"]["primary_decision_exact"] == 0
    assert score["full_population_missing_output_pairs"] == 1
    assert score["full_population_group_unresolved"] == 1
    assert score["missing_output_pairs"] == 0
    assert "taxonomy_exact" not in score["weighted"]
    with pytest.raises(DedupEvaluationError, match="ADJUDICATION_MASK_REQUIRED"):
        subject.masked_score(revised, predictions, set())


def test_three_phases_separate_reference_changes_from_mask_changes_and_keep_predictions():
    labels, predictions, updates = fixture()
    revised, _ = subject.revise_reference(labels, updates, {"c"})
    views = {"old": predictions, "new": deepcopy(predictions)}
    views["new"][1].update(decision("YES"))
    before = deepcopy(views)
    result = subject.compare(labels, revised, views, {"c"})
    for phase in result["phases"].values():
        assert phase["old"]["scored_rows"] == phase["new"]["scored_rows"]
    assert result["phases"]["prior_reference_full1000"]["old"]["scored_rows"] == 3
    assert result["phases"]["revised_reference_masked995"]["old"]["scored_rows"] == 2
    assert result["reference_and_mask_effects"]["old"]["reference_only_pp"]["weighted"]["duplicate_precision"] == 100
    assert views == before


def test_exact_reverse_companion_swaps_directions_not_group_and_rejects_approximate_match():
    _, _, updates = fixture()
    panel = [
        {"canonical_pair_id": pid, "review_id": rid, "payload": {"document_a": {"text": a}, "document_b": {"text": b}}}
        for pid, rid, a, b in [("a", "A", "政策＋商品", "政策"), ("b", "B", "政策", "政策＋商品")]
    ]
    checks = [
        {
            "in_scope_review_id": "A",
            "companion_review_id": "B",
            "companion_canonical_pair_id": "b",
            "input_relation": "EXACT_A_B_REVERSE",
            "companion_proposed_decision": decision("YES", "NO", "YES"),
        }
    ]
    result = subject.companion_updates(checks, updates, panel)
    assert result[0]["decision"] == decision("YES", "NO", "YES")
    assert result[0]["provenance"] == "EXACT_INPUT_COMPANION"
    panel[1]["payload"]["document_a"]["text"] += "近似但不同"
    with pytest.raises(DedupEvaluationError, match="ADJUDICATION_COMPANION"):
        subject.companion_updates(checks, updates, panel)


def test_saved_projection_retains_all_bypasses_and_missing_sentinels():
    cell = {
        "pairs": [
            {"canonical_pair_id": "a", "primary": decision("YES"), "status": "DETERMINISTIC_BYPASS"},
            {"canonical_pair_id": "b", "primary": decision("UNRESOLVED"), "status": "ENGINEERING_FAILURE"},
        ]
    }
    preds = subject.project_cell(cell)
    assert len(preds) == 2
    assert "metric_only_missing_output" not in preds[0]
    assert preds[1]["metric_only_missing_output"] is True
    cell["pairs"].append(deepcopy(cell["pairs"][0]))
    with pytest.raises(DedupEvaluationError):
        subject.project_cell(cell)


def test_exact_closure_ignores_similar_truncated_and_excluded_pairs():
    _, _, updates = fixture()
    panel = [
        {
            "canonical_pair_id": pid,
            "review_id": pid.upper(),
            "payload": {
                "document_a": {"text": a},
                "document_b": {"text": b},
                "long_document_evidence": {"truncated": truncated},
            },
        }
        for pid, a, b, truncated in [
            ("a", "policy product", "policy", False),
            ("reverse", "policy", "policy product", False),
            ("similar", "policy", "policy product!", False),
            ("excluded", "policy", "policy product", False),
            ("truncated", "policy", "policy product", True),
        ]
    ]
    before = deepcopy((updates, panel))
    result = subject.close_exact_companions(updates, panel, {"excluded"})
    assert len(result) == 1
    assert result[0]["canonical_pair_id"] == "reverse"
    assert result[0]["decision"] == decision("YES", "NO", "YES")
    assert result[0]["input_relation"] == "EXACT_A_B_REVERSE"
    assert (updates, panel) == before
    updates.append({**updates[0], "canonical_pair_id": "reverse", "decision": decision("NO")})
    with pytest.raises(DedupEvaluationError, match="ADJUDICATION_EXACT_SOURCE"):
        subject.close_exact_companions(updates, panel, set())


@pytest.mark.skipif(not subject.ADJUDICATION.is_dir(), reason="archived development artifacts are not installed")
def test_archived_replay_reproduces_prior_scores_and_exports_a_frozen_uniform_comparison(tmp_path):
    output = tmp_path / "comparison"
    summary = subject.export(output)
    assert summary["external_model_calls"] == 0
    assert summary["versions"] == ["v0.6.2.12", "v0.6.2.33-exp1"]
    assert (summary["population"], summary["scored_population"], summary["excluded_population"]) == (1000, 995, 5)
    assert summary["reference_provenance"] == {
        "user_confirmed": 11,
        "assistant_rule_applications": 46,
        "exact_input_companions": 8,
        "excluded": 5,
    }
    scores = subject.read(output / "scores.json")["phases"]
    old = scores["prior_reference_full1000"]
    assert old[subject.BASELINE]["weighted"]["duplicate_precision"] == pytest.approx(0.6798539627801528)
    assert old[subject.CANDIDATE]["weighted"]["duplicate_recall"] == pytest.approx(0.8558718403463219)
    assert all(v["scored_rows"] == 995 for v in scores["revised_reference_masked995"].values())
    assert subject.read(output / "experiment_checkpoint.json")["runtime_defaults_changed"] is False
    assert subject.export(output) == summary
    (output / "scores.json").write_text("{}")
    with pytest.raises(DedupEvaluationError, match="REBENCHMARK_FROZEN_SOURCE"):
        subject.export(output)
