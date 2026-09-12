# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy

import pyarrow as pa
import pytest

from eval.dedup.judging.payload import _semantic_diff_packet
from eval.dedup.judging.payload_transport import (
    bind_transported_payload,
    encode_repair_feedback,
    validate_payload_transport,
)
from eval.dedup.validation import DedupEvaluationError, sha256_json


@pytest.mark.parametrize("feedback", [None, {"code": "QUOTE_INVALID", "message": "Retry", "details": {}}])
def test_feedback_remains_falsey_or_exact_json_after_real_arrow_roundtrip(feedback):
    encoded = encode_repair_feedback(feedback)
    row = json.loads(pa.Table.from_pylist([{"repair_feedback": encoded}]).to_pandas().to_json(orient="records"))[0]
    assert row["repair_feedback"] == encoded
    if feedback is None:
        assert not encoded
    else:
        assert encoded.startswith("Validation issue: ")
        assert json.loads(encoded.removeprefix("Validation issue: ")) == feedback


def test_nan_is_not_an_input_feedback_sentinel():
    with pytest.raises(DedupEvaluationError, match="REPAIR_FEEDBACK_TYPE"):
        encode_repair_feedback(float("nan"))


def test_real_arrow_and_jsonl_writer_roundtrip_preserves_mixed_span_meaning(tmp_path):
    from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter
    from nemo_curator.tasks import DocumentBatch

    a, b = "Shared notice.", "Shared notice. Settings"
    payload = {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=False),
    }
    packet = {"canonical_pair_id": "p", "judge_payload_hash": sha256_json(payload), "payload": payload}
    table = pa.Table.from_pylist([packet])
    output = tmp_path / "out.jsonl"
    JsonlWriter(path=str(tmp_path)).write_data(
        DocumentBatch(dataset_name="transport", data=table.to_pandas()), str(output)
    )
    observed = json.loads(output.read_text())
    snapshot = deepcopy((packet, observed))
    assert observed["payload"] != payload
    bound, audit = bind_transported_payload(packet, observed)
    assert bound["payload"] == payload
    assert audit["representation_changes"]["added_null_keys"] > 0
    assert audit["representation_changes"]["equal_integer_to_float"] > 0
    assert (packet, observed) == snapshot
    bound["payload"]["document_a"]["text"] = "Changed view"
    assert packet["payload"] == payload


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ({"text": "Original"}, {"text": "Changed"}),
        ({"text": "a\nb"}, {"text": "a b"}),
        ({"x": None}, {}),
        ({}, {"x": "unexpected"}),
        ([1, 2], [2, 1]),
        ([1], [1, 2]),
        (1, True),
        (False, 0),
        (2, 2.5),
        (2**53 + 1, float(2**53 + 1)),
        (2, float("inf")),
        (2, float("nan")),
        (1, "1"),
    ],
)
def test_text_membership_value_precision_and_type_changes_are_not_transport_noise(source, value):
    with pytest.raises(DedupEvaluationError, match="PAYLOAD_TRANSPORT_CHANGED"):
        validate_payload_transport(source, value)


@pytest.mark.parametrize("field", ["canonical_pair_id", "judge_payload_hash"])
def test_unchanged_echo_does_not_bypass_pair_or_digest_identity(field):
    packet = {
        "canonical_pair_id": "p",
        "judge_payload_hash": sha256_json({"text": "Original"}),
        "payload": {"text": "Original"},
    }
    with pytest.raises(DedupEvaluationError, match="PAYLOAD_TRANSPORT_IDENTITY"):
        bind_transported_payload(packet, {**packet, field: "other"})
