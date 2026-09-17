# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded, format-only recovery from demonstrably repetitive truncated output."""

from __future__ import annotations

import json
from copy import deepcopy

# The selected recovery changed only the follow-up wording. Historical replay
# diagnostics were emitted by the unchanged detector and therefore retain its
# original contract identity.
CONTRACT = "dedup-main-truncated-repetition-recovery-v1"
REPAIR_CONTRACT = "dedup-main-truncated-repetition-recovery-v2"
TAIL_CHARACTERS = 4096
MAX_PERIOD = 128
MIN_AGREEMENT = 0.98
FEEDBACK = (
    "Output-format recovery only: the previous attempt was truncated while repeating an explanation. "
    "Start a new complete answer using the unchanged rubric and documents above. Return exactly one "
    "complete JSON object with all required fields. Keep each reasoning field to one short sentence "
    "of at most 40 words. Do not enumerate or repeatedly list identifiers or examples. The failed "
    "output is intentionally omitted; do not try to continue it. Evaluate all input content under "
    "the original rules: concise explanations must not change the decision criteria or omit "
    "required evidence."
)
FEEDBACK = FEEDBACK.replace(
    "Do not enumerate or repeatedly list identifiers or examples.",
    "Do not repeat source names or examples. Required evidence span IDs are not optional: "
    "retain compact, concrete citations in span_a_delta, span_b_delta and every other reasoning "
    "field where the original rubric requires them; phrases like 'A-only spans' or 'B-only spans' "
    "do not substitute for actual IDs.",
)


class TruncatedRepetition(BaseException):
    """Bypass SDK format correction without feeding the repetitive answer back to it."""

    def __init__(self, diagnostic: dict, stage: str):
        super().__init__(CONTRACT)
        self.diagnostic, self.stage = diagnostic, stage


def detect(response: dict) -> dict | None:
    """Only incomplete length-terminated outputs with a strongly periodic suffix qualify."""
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if choice.get("finish_reason") != "length" or not isinstance(content, str) or len(content) < TAIL_CHARACTERS:
        return None
    text = content.strip()
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        json.loads(text)
    except (ValueError, RecursionError):
        pass
    else:
        return None
    tail = content[-TAIL_CHARACTERS:]
    if not any(character.isalnum() for character in tail):
        return None
    for period in range(1, MAX_PERIOD + 1):
        comparisons = len(tail) - period
        agreement = sum(a == b for a, b in zip(tail[period:], tail[:-period], strict=True)) / comparisons
        if agreement >= MIN_AGREEMENT:
            return {
                "contract_version": CONTRACT,
                "finish_reason": "length",
                "content_characters": len(content),
                "tail_characters": len(tail),
                "period_characters": period,
                "period_agreement": agreement,
            }
    return None


def repair_request(initial: dict) -> dict:
    """Preserve the entire initial request and append static, content-free format feedback."""
    result = deepcopy(initial)
    result["messages"].append({"role": "user", "content": [{"type": "text", "text": FEEDBACK}]})
    return result
