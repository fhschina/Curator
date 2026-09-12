# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
import re

import pytest

from eval.dedup.analysis import critic_selected_experiment as experiment
from eval.dedup.judging import critic_retention_v6 as subject

ROOT = experiment.base.PRIOR.parent / "critic-five-hour-20260911T070052Z/retention-v4-paired96"


@pytest.mark.skipif(not (ROOT / "complete.json").exists(), reason="local complete proof corpus unavailable")
@pytest.mark.parametrize("rid", ["H0208", "H0232", "H0312", "H0546", "H0606", "H0641", "H0907"])
def test_source_order_revision_keeps_actual_proof_and_public_direction_contract(rid):
    row = next(r for r in json.loads((ROOT / "panel_private.json").read_text()) if r["review_id"] == rid)
    receipt = json.loads((ROOT / "responses" / f"1-candidate-{row['canonical_pair_id']}.json").read_text())
    assert subject.response_schema() == subject.previous.response_schema()
    assert subject.apply_review(
        row["main_public"], row["payload"], receipt["parsed_review"]
    ) == subject.previous.apply_review(row["main_public"], row["payload"], receipt["parsed_review"])


@pytest.mark.skipif(
    not (ROOT.parent / "retention-v4-order96-transport-v2/review_complete.json").exists(),
    reason="latest review unavailable",
)
def test_actual_renderer_preserves_every_span_in_each_original_side_order_without_metadata():
    spec, _ = experiment.specification(
        "eval.dedup.judging.critic_retention_v6", experiment.base.previous.RESOURCES / "v06212_retention_v6.yaml"
    )
    render = experiment.renderer(spec)
    original = json.loads((ROOT / "manifest.json").read_text())["specs"]["candidate"]
    control_render = experiment.renderer(original)
    for row in json.loads((ROOT / "panel_private.json").read_text()):
        messages = render(row["payload"])
        assert messages[0] == control_render(row["payload"])[0]
        text = messages[1]["content"]
        assert row["review_id"] not in text
        assert row["canonical_pair_id"] not in text
        for side in ("a", "b"):
            view = text.split(f"<document_{side}_visible_order>")[1].split(f"</document_{side}_visible_order>")[0]
            expected = []
            for span in row["payload"]["semantic_diff_evidence"]["spans"]:
                if span["kind"] == "SHARED":
                    expected.append((span[f"{side}_start_char"], span["span_id"], span[f"{side}_text"]))
                elif span["side"] == side.upper():
                    expected.append((span["start_char"], span["span_id"], span["text"]))
            expected.sort()
            actual = re.findall(r"\[([ABS]\d{3}) (?:SHARED|[AB]_ONLY) chars \d+:\d+\]", view)
            assert actual == [item[1] for item in expected]
            assert all(item[2] in view for item in expected)
