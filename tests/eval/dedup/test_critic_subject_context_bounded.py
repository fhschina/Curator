# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import pytest

from eval.dedup.analysis import critic_subject_context_bounded as trial

PARENT = trial.previous.proof.previous.SESSION / "subject-context-v5-proof-pilot32"


@pytest.mark.skipif(not (PARENT / "review_complete.json").exists(), reason="reviewed composed trial absent")
def test_length_gate_keeps_all_cases_and_exact_original_fallback(tmp_path):
    root = tmp_path / "bounded"
    manifest = trial.prepare(root, PARENT)
    trial.previous.proof.reference.verify_freeze(root / "manifest.json")
    origin = Path(manifest["coverage_origin"])
    rows = json.loads((root / "panel_private.json").read_text())
    assert rows == json.loads((origin / "specialist/panel_private.json").read_text())
    assert len(rows) == 1000
    by = {r["canonical_pair_id"]: r for r in rows}
    old = {r["canonical_pair_id"]: r for r in json.loads((origin / "specialist/requests_frozen.json").read_text())}
    requests = json.loads((root / "requests_frozen.json").read_text())
    assert len(requests) == len(old) == 129
    assert {r["canonical_pair_id"] for r in requests} == old.keys()
    fallbacks = []
    for request in requests:
        before = old[request["canonical_pair_id"]]
        assert request["input_tokens"] + 6144 <= 32768
        if request["presentation_route"] == "EXACT_ORIGINAL_SPAN_PRESENTATION":
            assert request["body"] == before["body"]
            assert request["request_sha256"] == before["request_sha256"]
            assert request["intact_context_input_tokens"] + 6144 > 32768
            fallbacks.append(by[request["canonical_pair_id"]]["review_id"])
        else:
            row = by[request["canonical_pair_id"]]
            user = request["body"]["messages"][1]["content"]
            assert row["payload"]["document_a"]["text"] in user
            assert row["payload"]["document_b"]["text"] in user
            assert user.endswith(before["body"]["messages"][1]["content"])
    assert fallbacks == manifest["fallback_review_ids_private"] == ["H0837"]
    assert not (root / "responses").exists()
