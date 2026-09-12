# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Original-source span presentation with unchanged retention proof semantics."""

from eval.dedup.judging import critic_retention_v4 as previous

CONTRACT = "dedup-critic-retention-v6"


def response_schema() -> dict:
    return previous.response_schema()


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    return previous.apply_review(main, payload, value)
