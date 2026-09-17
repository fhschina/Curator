# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lossless spelling recovery for shared coverage-anchor references."""

from __future__ import annotations

import re
from copy import deepcopy

from eval.dedup.judging.coverage_selection import span_inventory

CONTRACT = "dedup-shared-anchor-prefix-recovery-v2"
ZERO_PADDING_CONTRACT = "dedup-shared-anchor-zero-padding-v1"


def normalize_shared_anchor_ids(payload: dict, value: dict) -> tuple[dict, list[dict]]:
    """Canonicalize only unambiguous shared IDs; never select or remove evidence."""
    anchors = value.get("shared_anchor_ids") if isinstance(value, dict) else None
    if (
        not isinstance(anchors, list)
        or not all(isinstance(sid, str) for sid in anchors)
        or len(anchors) != len(set(anchors))
    ):
        return value, []
    if not any(re.fullmatch(r"[0-9]{3}", sid) or re.fullmatch(r"S[0-9]{1,2}", sid) for sid in anchors):
        return value, []

    shared = {sid for sid, span in span_inventory(payload).items() if span["kind"] == "SHARED"}
    normalized, corrections = [], []
    for index, sid in enumerate(anchors):
        target = sid
        if sid not in shared:
            if re.fullmatch(r"[0-9]{3}", sid):
                target = "S" + sid
            elif re.fullmatch(r"S[0-9]{1,2}", sid):
                target = "S" + sid[1:].zfill(3)
            else:
                return value, []
            if target not in shared:
                return value, []
            corrections.append({"index": index, "original": sid, "canonical": target})
        normalized.append(target)
    if len(normalized) != len(set(normalized)):
        return value, []
    result = deepcopy(value)
    result["shared_anchor_ids"] = normalized
    return result, corrections
