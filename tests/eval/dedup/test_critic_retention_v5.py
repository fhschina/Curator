# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_selected_experiment as experiment
from eval.dedup.judging import critic_retention_v5 as subject
from eval.dedup.validation import DedupEvaluationError

ROOT = experiment.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v4-paired96"


@pytest.mark.skipif(not (ROOT / "complete.json").exists(), reason="local source proofs unavailable")
@pytest.mark.parametrize("rid", ["H0208", "H0232", "H0312", "H0606", "H0641", "H0711", "H0907"])
def test_prompt_revision_preserves_real_proof_compilation_and_veto_scope(rid):
    row = next(r for r in json.loads((ROOT / "panel_private.json").read_text()) if r["review_id"] == rid)
    receipt = json.loads((ROOT / "responses" / f"1-candidate-{row['canonical_pair_id']}.json").read_text())
    value = receipt["parsed_review"]
    assert subject.apply_review(row["main_public"], row["payload"], value) == subject.previous.apply_review(
        row["main_public"], row["payload"], value
    )
    assert subject.response_schema() == subject.previous.response_schema()
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(row["main_public"], row["payload"], {k: v for k, v in value.items() if k != "conflict"})


@pytest.mark.skipif(not (ROOT / "review_complete.json").exists(), reason="review unavailable")
def test_actual_renderer_changes_criterion_without_adding_source_or_prediction_context():
    spec, _ = experiment.specification(
        "eval.dedup.judging.critic_retention_v5", experiment.base.previous.RESOURCES / "v06212_retention_v5.yaml"
    )
    old = json.loads((ROOT / "manifest.json").read_text())["specs"]["candidate"]
    row = next(r for r in json.loads((ROOT / "panel_private.json").read_text()) if r["review_id"] == "H0606")
    messages = experiment.renderer(spec)(row["payload"])
    control = experiment.renderer(old)(row["payload"])
    assert messages[1]["content"].split("<semantic_diff")[1] == control[1]["content"].split("<semantic_diff")[1]
    assert "complete sentence" in messages[0]["content"]
    assert "shared predicate" in messages[1]["content"]
    assert row["review_id"] not in json.dumps(messages)
    assert row["canonical_pair_id"] not in json.dumps(messages)
