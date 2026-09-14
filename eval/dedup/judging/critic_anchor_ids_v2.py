# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in spelling recovery for explicitly shared three-digit anchor references."""

from __future__ import annotations

import re
from copy import deepcopy

from eval.dedup.judging import critic_anchor_ids as previous
from eval.dedup.judging.coverage_selection import span_inventory

CONTRACT = "dedup-shared-anchor-prefix-recovery-v2"


def normalize_shared_anchor_ids(payload: dict, value: dict) -> tuple[dict, list[dict]]:
    anchors = value.get("shared_anchor_ids") if isinstance(value, dict) else None
    if not isinstance(anchors, list) or not any(
        isinstance(sid, str) and re.fullmatch(r"[0-9]{3}", sid) for sid in anchors
    ):
        return previous.normalize_shared_anchor_ids(payload, value)
    if not all(isinstance(sid, str) for sid in anchors) or len(anchors) != len(set(anchors)):
        return value, []
    # The field fixes the evidence kind; the validated inventory fixes the exact target.
    shared = {sid for sid, span in span_inventory(payload).items() if span["kind"] == "SHARED"}
    normalized, changes = [], []
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
            changes.append({"index": index, "original": sid, "canonical": target})
        normalized.append(target)
    if len(normalized) != len(set(normalized)):
        return value, []
    result = deepcopy(value)
    result["shared_anchor_ids"] = normalized
    return result, changes
