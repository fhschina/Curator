# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from eval.dedup.analysis import critic_adjudication_packets as packets
from eval.dedup.analysis.critic_subject_experiment import SESSION

SOURCE = SESSION / "v4-subject-v2-full1000"


@pytest.mark.skipif(not (SOURCE / "cause_accounting_v2.json").exists(), reason="full reviewed cause audit absent")
def test_all_pending_sources_are_exact_and_predictions_stay_private(tmp_path):
    root = tmp_path / "blind"
    manifest = packets.build(root, SOURCE)
    packets.previous.selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assert manifest["population"] == 88
    blind = [json.loads(line) for line in (root / "inputs_blind.jsonl").read_text().splitlines()]
    keys = {k["case_id"]: k for k in json.loads((root / "private_key.json").read_text())}
    originals = {r["canonical_pair_id"]: r for r in json.loads((SOURCE / "panel_private.json").read_text())}
    assert len(blind) == len(keys) == 88
    for item in blind:
        assert "review_id" not in item
        assert "human_same_duplicate_group" not in item
        original = originals[keys[item["case_id"]]["canonical_pair_id"]]
        assert item["payload"]["document_a"]["text"] == original["payload"]["document_a"]["text"]
        assert item["payload"]["document_b"]["text"] == original["payload"]["document_b"]["text"]
        assert (root / "cases" / f"{item['case_id']}.md").is_file()
    annotations = json.loads((root / "annotation_template.json").read_text())
    assert all(a["a_can_replace_b"] is None and a["reviewer"] is None for a in annotations)
