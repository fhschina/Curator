# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""One failure-only coverage retry for incomplete output ending in a tab loop."""

from __future__ import annotations

import json
from copy import deepcopy

CONTRACT = "dedup-coverage-truncated-tab-recovery-v1"
TAIL_CHARACTERS = 4096
MIN_TAB_FRACTION = 0.98
FEEDBACK = (
    "Output-format recovery only: the previous coverage response ended at the token limit "
    "while repeating whitespace. Answer the original coverage task again using the unchanged "
    "documents, rubric and response schema. Return exactly one complete JSON object. Keep the "
    "explanation concise and do not emit repeated whitespace. Preserve all required fields and "
    "actual concrete span IDs required as evidence; do not replace them with generic descriptions. "
    "The failed output is intentionally omitted: do not continue it. This instruction changes "
    "only output formatting, not the semantic decision criteria or evidence requirements."
)


def detect(response: dict) -> dict | None:
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if choice.get("finish_reason") != "length" or not isinstance(text, str) or len(text) < TAIL_CHARACTERS:
        return None
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        pass
    else:
        return None
    fraction = text[-TAIL_CHARACTERS:].count("\t") / TAIL_CHARACTERS
    if fraction < MIN_TAB_FRACTION:
        return None
    return {
        "contract_version": CONTRACT,
        "finish_reason": "length",
        "content_characters": len(text),
        "tail_characters": TAIL_CHARACTERS,
        "tail_tab_fraction": fraction,
    }


def repair_request(initial: dict) -> dict:
    result = deepcopy(initial)
    result["messages"].append({"role": "user", "content": [{"type": "text", "text": FEEDBACK}]})
    return result
