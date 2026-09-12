# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Binding-first materiality revision using the unchanged selected-source proof contract."""

from eval.dedup.judging import critic_retention_v3 as previous

CONTRACT = "dedup-critic-retention-v5"


def response_schema() -> dict:
    return previous.response_schema()


def apply_review(main: dict, payload: dict, value: dict | None) -> tuple[dict, str]:
    return previous.apply_review(main, payload, value)
