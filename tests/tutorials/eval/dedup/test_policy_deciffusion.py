# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import pytest
from policy_cases import aligned, controlled
from policy_deciffusion import FIELDS, parse_response, request_body


def response():
    return {"answers": {field: {"probabilities": {"YES": 0.5, "NO": 0.5, "UNRESOLVED": 0.0}} for field in FIELDS}}


def test_native_tied_probability_is_unresolved():
    assert set(parse_response(response()).values()) == {"unresolved"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True, 1.1])
def test_invalid_native_probabilities_are_not_decisions(bad: object):
    value = copy.deepcopy(response())
    value["answers"][FIELDS[0]]["probabilities"]["YES"] = bad
    with pytest.raises(ValueError, match="finite"):
        parse_response(value)


def test_choice_input_has_only_evidence_not_qwen_protocol_or_labels():
    row = aligned(controlled()[:1])[0]
    body = request_body(
        row,
        "Retention policy\nSPAN PACKET AND LEDGER\nHIDDEN_QWEN_PROTOCOL",
        '<semantic_diff status="COMPLETE">{{ semantic_diff }}</semantic_diff>',
    )
    assert "HIDDEN_QWEN_PROTOCOL" not in str(body)
    assert "a_can_replace_b" not in body["state"]["visible_material"]
    assert body["questions"][FIELDS[0]]["type"] == "choice"
