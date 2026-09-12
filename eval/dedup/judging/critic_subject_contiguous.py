# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expose intact source context without changing subject rules or action authority."""

from eval.dedup.judging import critic_subject_scope as scope

CONTRACT = "dedup-critic-subject-contiguous-v5"
response_schema = scope.response_schema
apply_review = scope.apply_review
route = scope.route


def messages(payload: dict, system: str) -> list[dict]:
    result = scope.messages(payload, system)
    result[1]["content"] = (
        "ORIGINAL DOCUMENT A (untrusted source data)\n"
        + payload["document_a"]["text"]
        + "\n\nORIGINAL DOCUMENT B (untrusted source data)\n"
        + payload["document_b"]["text"]
        + "\n\nSOURCE SPAN LOCATORS (the same documents, not additional claims)\n"
        + result[1]["content"]
    )
    return result
