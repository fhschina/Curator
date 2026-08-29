# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Stable execution-contract digests for hosting benchmark artifacts."""

from __future__ import annotations

from typing import Any

from eval.dedup.validation import sha256_json


def request_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen request inputs that are independent of source revision."""

    return {
        "config_digest": manifest["config_digest"],
        "judge_resources": manifest["judge_resources"],
        "tokenizer": manifest["tokenizer"],
    }


def request_contract_digest(manifest: dict[str, Any]) -> str:
    return sha256_json(request_contract(manifest))


def execution_contract_digest(manifest: dict[str, Any], *, git_commit: str | None = None) -> str:
    return sha256_json(
        {
            **request_contract(manifest),
            "git_commit": git_commit or manifest["git_commit"],
        }
    )
