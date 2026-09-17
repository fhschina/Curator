# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command-line entry point for Dedup Eval v0.7.1."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from eval.dedup.runtime import TOOL_VERSION

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_repository_env(env_file: Path | None = None) -> None:
    """Load simple KEY=VALUE credentials without overriding the process environment."""
    path = env_file or (_REPOSITORY_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if key and key.replace("_", "a").isalnum() and not key[0].isdigit():
            os.environ.setdefault(key, value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args == ["--version"]:
        print(TOOL_VERSION)
        return 0
    from eval.dedup.runtime.backends import main as backend_main

    return backend_main(args)


__all__ = ["main"]
