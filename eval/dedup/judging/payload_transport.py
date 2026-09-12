# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Validate Arrow's lossless JSON representation changes against a frozen payload."""

from __future__ import annotations

import json
import math
from collections import Counter
from copy import deepcopy
from typing import Any

from eval.dedup.validation import require, sha256_json


def encode_repair_feedback(value: dict | None) -> str:
    """Keep missing feedback falsey and prevent NDD from decoding JSON-looking seed strings."""
    require(
        value is None or isinstance(value, dict), "REPAIR_FEEDBACK_TYPE", "expected structured safe feedback or None"
    )
    return (
        ""
        if value is None
        else "Validation issue: " + json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def validate_payload_transport(expected: Any, observed: Any) -> dict[str, int]:
    """Allow null struct padding and exact integer-to-float promotion, never text repair."""
    changes: Counter = Counter()

    def compare(source: Any, value: Any, path: str) -> None:
        if isinstance(source, dict) and isinstance(value, dict):
            require(source.keys() <= value.keys(), "PAYLOAD_TRANSPORT_CHANGED", "payload field missing", path=path)
            for key in value.keys() - source.keys():
                require(value[key] is None, "PAYLOAD_TRANSPORT_CHANGED", "non-null field added", path=f"{path}.{key}")
                changes["added_null_keys"] += 1
            for key, child in source.items():
                compare(child, value[key], f"{path}.{key}")
        elif isinstance(source, list) and isinstance(value, list):
            require(len(source) == len(value), "PAYLOAD_TRANSPORT_CHANGED", "list membership changed", path=path)
            for index, (child, item) in enumerate(zip(source, value, strict=True)):
                compare(child, item, f"{path}[{index}]")
        elif type(source) is int and type(value) is float:
            require(
                math.isfinite(value) and value.is_integer() and int(value) == source,
                "PAYLOAD_TRANSPORT_CHANGED",
                "integer value changed in transport",
                path=path,
            )
            changes["equal_integer_to_float"] += 1
        else:
            require(
                type(source) is type(value) and source == value,
                "PAYLOAD_TRANSPORT_CHANGED",
                "payload value or type changed",
                path=path,
            )

    compare(expected, observed, "payload")
    return dict(changes)


def bind_transported_payload(packet: dict, row: dict) -> tuple[dict, dict]:
    """Bind an in-memory view to original input only after checking the untouched echo."""
    require(
        packet["canonical_pair_id"] == row.get("canonical_pair_id")
        and packet["judge_payload_hash"] == row.get("judge_payload_hash") == sha256_json(packet["payload"]),
        "PAYLOAD_TRANSPORT_IDENTITY",
        "pair or frozen payload digest differs",
    )
    changes = validate_payload_transport(packet["payload"], row.get("payload"))
    audit = {
        "transport_contract": "dedup-arrow-payload-binding-v1",
        "original_payload_sha256": sha256_json(packet["payload"]),
        "raw_echo_payload_sha256": sha256_json(row["payload"]),
        "representation_changes": changes,
    }
    return {**row, "payload": deepcopy(packet["payload"])}, audit
