# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Internal v0.7 release frozen from the selected Exp6 runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eval.dedup.analysis import exp6_coverage_format_fix as candidate

VERSION = "v0.7"
SOURCE_VERSION = candidate.VERSION
old = candidate.old
coverage_format_recovery = candidate.coverage_format_recovery
critic_anchor_ids = candidate.critic_anchor_ids
repetition = candidate.repetition


def execute_case(root: Path, row: dict, endpoint: str, frozen: dict, render_coverage: Callable) -> dict:
    """Run the frozen candidate while recording the internal release identity."""
    return candidate.execute_case(root, row, endpoint, frozen, render_coverage, version=VERSION)
