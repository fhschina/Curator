# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Citation-preserving clarification of the failure-only repetition recovery note."""

from eval.dedup.judging import main_repetition_recovery as previous

CONTRACT = "dedup-main-truncated-repetition-recovery-v2"
TruncatedRepetition = previous.TruncatedRepetition
detect = previous.detect
FEEDBACK = previous.FEEDBACK.replace(
    "Do not enumerate or repeatedly list identifiers or examples.",
    "Do not repeat source names or examples. Required evidence span IDs are not optional: "
    "retain compact, concrete citations in span_a_delta, span_b_delta and every other reasoning "
    "field where the original rubric requires them; phrases like 'A-only spans' or 'B-only spans' "
    "do not substitute for actual IDs.",
)


def repair_request(initial: dict) -> dict:
    result = previous.repair_request(initial)
    result["messages"][-1]["content"][0]["text"] = FEEDBACK
    return result
