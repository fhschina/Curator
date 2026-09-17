# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transport-only schema projections; local semantic validators remain authoritative."""

from __future__ import annotations

from copy import deepcopy

from eval.dedup.core.validation import require

MODES = ("without_unique_items", "portable_structure")


def project_schema(schema: dict, mode: str) -> dict:
    require(mode in MODES, "CRITIC_SCHEMA_MODE", "unknown transport projection")
    omitted = {"uniqueItems"} if mode == "without_unique_items" else {"uniqueItems", "minLength", "maxLength"}

    def visit(node: object) -> object:
        if isinstance(node, list):
            return [visit(value) for value in node]
        if not isinstance(node, dict):
            return deepcopy(node)
        return {key: visit(value) for key, value in node.items() if key not in omitted}

    return visit(schema)


def response_format(schema: dict, mode: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "critic_retention", "strict": True, "schema": project_schema(schema, mode)},
    }
