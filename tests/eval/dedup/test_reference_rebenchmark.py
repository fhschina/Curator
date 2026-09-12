# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pytest

from eval.dedup.analysis import reference_rebenchmark as subject
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json


def fixture():
    labels = []
    for pid, group, a, b, weight in (
        ("p1", "NO", "NO", "NO", 4),
        ("p2", "YES", "YES", "YES", 2),
        ("p3", "NO", "NO", "NO", 1),
    ):
        labels.append(
            {
                "canonical_pair_id": pid,
                "review_id": pid.upper(),
                "human_same_duplicate_group": group,
                "human_a_can_replace_b": a,
                "human_b_can_replace_a": b,
                "human_relation_type": "UNRELATED",
                "human_material_difference": "MAJOR",
                "stratum_population_n": str(weight * 3),
                "stratum_sample_n": "3",
            }
        )
    supported = [
        {
            "canonical_pair_id": "p1",
            "review_id": "P1",
            "provenance": subject.policy.PROVENANCE,
            "independently_adjudicated": False,
            "old_reference": subject.primary(labels[0], "human_"),
            "proposed_primary_not_gold": {
                "same_duplicate_group": "YES",
                "a_can_replace_b": "NO",
                "b_can_replace_a": "YES",
            },
        }
    ]
    deferred = [{"canonical_pair_id": "p3"}]
    predictions = [
        {"canonical_pair_id": "p1", "same_duplicate_group": "YES", "a_can_replace_b": "NO", "b_can_replace_a": "YES"},
        {"canonical_pair_id": "p2", "same_duplicate_group": "NO", "a_can_replace_b": "NO", "b_can_replace_a": "NO"},
        {"canonical_pair_id": "p3", "same_duplicate_group": "NO", "a_can_replace_b": "NO", "b_can_replace_a": "NO"},
    ]
    return labels, supported, deferred, predictions


def test_partial_draft_preserves_original_weights_taxonomy_and_pending_denominator():
    labels, supported, deferred, predictions = fixture()
    before = deepcopy((labels, supported, deferred, predictions))
    draft, changes = subject.revised_reference(labels, supported, deferred)
    assert len(changes) == 1
    assert changes[0]["weight"] == 4
    assert [subject._weight(r) for r in draft] == [4, 2, 1]
    assert subject.primary(draft[2], "human_") == subject.primary(labels[2], "human_")
    assert draft[2]["reference_application_status"] == "DEFERRED_OLD_PRIMARY_RETAINED"
    assert all(r["reference_status"] == subject.DRAFT_STATUS and not r["independent_adjudication"] for r in draft)
    assert draft[0]["human_relation_type"] == labels[0]["human_relation_type"]
    scores = subject.score(draft, predictions)
    assert scores["weighted"]["rows"] == 3
    assert scores["weighted"]["weight_total"] == 7
    assert scores["weighted"]["duplicate_precision"] == 1
    assert scores["weighted"]["duplicate_recall"] == pytest.approx(4 / 6)
    assert scores["weighted"]["primary_decision_exact"] == pytest.approx(5 / 7)
    assert "taxonomy_exact" not in scores["weighted"]
    assert (labels, supported, deferred, predictions) == before


def test_relabel_delta_is_same_predictions_not_model_improvement():
    labels, supported, deferred, preds = fixture()
    draft, _ = subject.revised_reference(labels, supported, deferred)
    delta = subject.metric_deltas({"view": subject.score(labels, preds)}, {"view": subject.score(draft, preds)})
    assert delta["view"]["weighted"]["primary_decision_exact"] == pytest.approx(100 * 4 / 7)
    assert delta["view"]["weighted"]["duplicate_precision"] == 100


@pytest.mark.parametrize("kind", ["extra", "missing", "duplicate_prediction", "duplicate_label"])
def test_scoring_rejects_inexact_membership(kind):
    labels, _, _, preds = fixture()
    if kind == "extra":
        preds.append({**preds[0], "canonical_pair_id": "extra"})
    elif kind == "missing":
        preds.pop()
    elif kind == "duplicate_prediction":
        preds.append(deepcopy(preds[0]))
    else:
        labels.append(deepcopy(labels[0]))
    with pytest.raises(DedupEvaluationError):
        subject.score(labels, preds)


@pytest.mark.parametrize("kind", ["unknown", "overlap", "gold", "two_yes", "wrong_old_reference", "wrong_review_id"])
def test_application_rejects_undeclared_scope_or_gold_promotion(kind):
    labels, supported, deferred, _ = fixture()
    case = supported[0]
    if kind == "unknown":
        case["canonical_pair_id"] = "unknown"
    elif kind == "overlap":
        deferred.append({"canonical_pair_id": "p1"})
    elif kind == "gold":
        case["independently_adjudicated"] = True
    elif kind == "two_yes":
        case["proposed_primary_not_gold"]["a_can_replace_b"] = "YES"
    elif kind == "wrong_old_reference":
        case["old_reference"]["same_duplicate_group"] = "YES"
    else:
        case["review_id"] = "OTHER"
    with pytest.raises(DedupEvaluationError):
        subject.revised_reference(labels, supported, deferred)


@pytest.mark.parametrize("reference_group", ["YES", "NO", "UNRESOLVED"])
def test_missing_output_never_gets_credit_or_dropped(reference_group):
    labels, _, _, preds = fixture()
    for key in subject.PRIMARY_FIELDS:
        labels[0][f"human_{key}"] = reference_group
        preds[0][key] = "UNRESOLVED"
    preds[0]["metric_only_missing_output"] = True
    metrics = subject.score(labels, preds)
    assert metrics["missing_output_pairs"] == 1
    assert metrics["error_counts"]["ENGINEERING_MISSING_OUTPUT"] == 1
    assert metrics["weighted"]["primary_decision_exact"] == pytest.approx(1 / 7)
    assert metrics["weighted"]["rows"] == 3
    if reference_group == "YES":
        assert metrics["weighted"]["false_negative"] == 6


def test_missing_output_cannot_carry_a_semantic_answer():
    labels, _, _, preds = fixture()
    preds[0]["metric_only_missing_output"] = True
    with pytest.raises(DedupEvaluationError, match="REBENCHMARK_SENTINEL"):
        subject.score(labels, preds)


def payload(a, b, truncated=False, spans=()):
    return {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": truncated},
        "semantic_diff_evidence": {"spans": list(spans)},
    }


def test_text_screen_preserves_full_population_and_is_not_a_label_rule():
    packets = {
        "candidate": payload("政策", "政策" + "商品" * 100),
        "chrome_only": payload("正文", "正文" + "导航" * 100),
        "small_delta": payload("政策", "政策商品"),
        "missing": payload(None, "商品"),
        "truncated": payload("政策", "政策" + "商品" * 100, True),
    }
    rows = subject.scan_payloads(packets)
    assert len(rows) == len(packets)
    assert {r["canonical_pair_id"] for r in rows if r["candidate"]} == {"candidate", "chrome_only"}
    assert len({tuple(sorted(r)) for r in rows}) == 1
    assert "missing" in subject.historical._csv_text(rows)
    assert all("proposed_primary" not in r for r in rows)


def test_known_policy_scan_includes_mirrors_short_additions_and_already_positive_cases():
    supported = [{"canonical_pair_id": "seed", "review_id": "H1", "product_side": "B", "policy_start": "政策"}]
    packets = {
        "seed": payload("政策完整", "政策完整商品"),
        "mirror": payload("政策完整商品", "政策完整"),
        "other_product": payload("政策完整", "政策完整别的商品"),
        "incomplete": payload(None, "政策完整"),
    }
    result = subject.exact_policy_families(supported, packets)
    assert {r["canonical_pair_id"] for r in result} == {"seed", "mirror", "other_product"}
    assert all(not r["automatic_label_change"] for r in result)


def test_mirror_primary_is_checked_in_text_order():
    labels, supported, _, _ = fixture()
    labels = [
        *labels[:1],
        {
            **labels[0],
            "canonical_pair_id": "mirror",
            "review_id": "MIRROR",
            "human_same_duplicate_group": "YES",
            "human_a_can_replace_b": "YES",
        },
    ]
    packets = [
        {"canonical_pair_id": "p1", "payload": payload("政策", "政策商品")},
        {"canonical_pair_id": "mirror", "payload": payload("政策商品", "政策")},
    ]
    assert len(subject.historical.exact_conflicts(labels, packets)) == 1
    draft, _ = subject.revised_reference(labels, supported, [])
    assert subject.historical.exact_conflicts(draft, packets) == []


def test_export_does_not_overwrite_existing_directory(tmp_path):
    path = tmp_path / "existing"
    path.mkdir()
    sentinel = path / "user.txt"
    sentinel.write_text("preserve")
    with pytest.raises(DedupEvaluationError, match="REBENCHMARK_OUTPUT_EXISTS"):
        subject.export(path)
    assert sentinel.read_text() == "preserve"


def test_freeze_detects_tampered_artifact(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text("{}")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"sources": {}, "artifacts": {str(path): sha256_file(path)}}))
    subject.verify_freeze(summary)
    path.write_text('{"changed": true}')
    with pytest.raises(DedupEvaluationError, match="REBENCHMARK_FROZEN_SOURCE"):
        subject.verify_freeze(summary)


def test_freeze_detects_metadata_tampering(tmp_path):
    manifest = {"sources": {}, "artifacts": {}, "reference_status": subject.DRAFT_STATUS}
    manifest["contract_digest"] = sha256_json(manifest)
    manifest["reference_status"] = "HUMAN_GOLD"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(manifest))
    with pytest.raises(DedupEvaluationError, match="REBENCHMARK_MANIFEST_DIGEST"):
        subject.verify_freeze(summary)


@pytest.mark.skipif(
    not (subject.POLICY_ROOT / "summary.json").exists(), reason="local frozen diagnostic artifacts unavailable"
)
def test_real_saved_outputs_replay_and_partial_reference_export(tmp_path):
    output = tmp_path / "benchmark"
    result = subject.export(output)
    assert result["population"] == 1000
    assert result["weight_total"] == pytest.approx(10023)
    assert result["primary_reference_changes"] == 5
    assert result["changed_weight"] == pytest.approx(50.225)
    assert result["external_model_calls"] == 0
    assert result["independently_adjudicated_new_pairs"] == 0
    assert result["supported_application_pairs"] == 6
    assert sha256_file(output / "reference_historical.csv") == subject.historical.REFERENCE_SHA256
    assert result["known_policy_family_matches"] == 6
    assert result["unreviewed_known_policy_family_ids"] == []
    old = json.loads((output / "historical_scores.json").read_text())["scores"]
    new = json.loads((output / "partial_draft_scores.json").read_text())["scores"]
    assert len(old) == len(new) == 6
    assert all(v["weighted"]["rows"] == 1000 for v in old.values())
    assert old["v0.6.2.32/final"]["missing_output_pairs"] == 3
    assert old["v0.6.2.9/final"]["weighted"]["duplicate_precision"] == pytest.approx(0.7386, abs=0.00005)
    assert old["v0.6.2.12/final"]["weighted"]["duplicate_recall"] == pytest.approx(0.7759, abs=0.00005)
    assert (
        new["v0.6.2.32/final"]["error_counts"]["OVER_GROUP"]
        == old["v0.6.2.32/final"]["error_counts"]["OVER_GROUP"] - 5
    )
    assert subject.export(output) == result
    assert result["contract_digest"] == sha256_json({k: v for k, v in result.items() if k != "contract_digest"})
