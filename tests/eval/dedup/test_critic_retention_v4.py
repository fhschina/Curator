# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_selected_experiment as experiment
from eval.dedup.judging import critic_retention_v4 as subject
from eval.dedup.validation import DedupEvaluationError

ROOT = experiment.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v3-material96"


@pytest.mark.skipif(not (ROOT / "complete.json").exists(), reason="local original input unavailable")
@pytest.mark.parametrize("rid", ["H0108", "H0126", "H0546", "H0641", "H0312", "H0907"])
def test_parser_and_direction_scope_are_unchanged_for_real_proofs(rid):
    row = next(r for r in json.loads((ROOT / "panel_private.json").read_text()) if r["review_id"] == rid)
    receipt = next(
        json.loads(p.read_text())
        for p in (ROOT / "responses").glob("1-candidate-*.json")
        if json.loads(p.read_text())["canonical_pair_id"] == row["canonical_pair_id"]
    )
    value = receipt["parsed_review"]
    assert subject.apply_review(row["main_public"], row["payload"], value) == subject.previous.apply_review(
        row["main_public"], row["payload"], value
    )
    assert subject.response_schema() == subject.previous.response_schema()
    value = {k: v for k, v in value.items() if k != "a_loss_span_id"}
    with pytest.raises(DedupEvaluationError):
        subject.apply_review(row["main_public"], row["payload"], value)


@pytest.mark.skipif(not (ROOT / "review_complete.json").exists(), reason="local predecessor review unavailable")
def test_actual_new_renderer_keeps_original_spans_and_omits_all_prediction_metadata():
    spec, _ = experiment.specification(
        "eval.dedup.judging.critic_retention_v4", experiment.base.previous.RESOURCES / "v06212_retention_v4.yaml"
    )
    row = next(r for r in json.loads((ROOT / "panel_private.json").read_text()) if r["review_id"] == "H0907")
    messages = experiment.renderer(spec)(row["payload"])
    text = json.dumps(messages, ensure_ascii=False)
    assert row["review_id"] not in text
    assert row["canonical_pair_id"] not in text
    for span in row["payload"]["semantic_diff_evidence"]["spans"]:
        assert span["span_id"] in messages[1]["content"]
        for key in ("a_text", "b_text") if span["kind"] == "SHARED" else ("text",):
            assert span[key] in messages[1]["content"]
    assert "bare menu/section label" in messages[0]["content"]
    assert "named failed object" in messages[1]["content"]
