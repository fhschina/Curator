# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Require named actual targets before an optional subject-only veto."""

from copy import deepcopy

from eval.dedup.core.validation import require
from eval.dedup.judging import critic_subject_scope as previous

CONTRACT = "dedup-critic-subject-kinds-v3"
KINDS = ("NAMED_ACTUAL_TARGET", "GENERIC_ROLE_OR_OBJECT", "UI_LABEL_OR_INSTRUCTION", "UNSUPPORTED")
route = previous.route
messages = previous.messages


def response_schema(payload: dict | None = None) -> dict:
    schema = previous.response_schema(payload)
    for side in ("a", "b"):
        key = f"{side}_subject_kind"
        schema["properties"][key] = {"type": "string", "enum": list(KINDS)}
        schema["required"].append(key)
    return schema


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    routing = route(main, payload)
    if routing != "REVIEW_BILATERAL_SUBJECTS":
        return deepcopy(main), routing
    require(
        isinstance(value, dict)
        and set(value) == set(response_schema()["required"])
        and all(value[f"{side}_subject_kind"] in KINDS for side in ("a", "b")),
        "SUBJECT_KINDS_SCHEMA",
        "explicit kind for each actual subject required",
    )
    original = {k: v for k, v in value.items() if not k.endswith("_subject_kind")}
    # Validate the entire original proof even when a scope condition will discard it.
    public, rule = previous.apply_review(main, payload, original)
    if any(value[f"{side}_subject_kind"] != "NAMED_ACTUAL_TARGET" for side in ("a", "b")):
        return deepcopy(main), "SUBJECT_VETO_REQUIRES_TWO_NAMED_ACTUAL_TARGETS"
    return public, rule
