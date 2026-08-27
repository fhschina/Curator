# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Artifact readers and immutable block-attempt helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.dedup.validation import require

from .config import HostingBenchmarkConfig, load_config


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "HOSTING_ARTIFACT_INVALID", "JSON artifact must be an object", path=str(path))
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                "HOSTING_ARTIFACT_INVALID",
                "JSONL row must be an object",
                path=str(path),
                line=line_number,
            )
            rows.append(value)
    return rows


def load_run(run_root: str | Path) -> tuple[Path, dict[str, Any], HostingBenchmarkConfig]:
    root = Path(run_root).resolve()
    manifest = read_json(root / "benchmark_manifest.json")
    config = load_config(manifest["config_path"])
    require(config.digest == manifest["config_digest"], "HOSTING_CONFIG_CHANGED", "benchmark config changed")
    return root, manifest, config


def next_attempt_root(root: Path) -> Path:
    existing = sorted(path for path in root.glob("attempt-*") if path.is_dir())
    path = root / f"attempt-{len(existing) + 1:02d}"
    path.mkdir(parents=True)
    return path


def complete_marker(root: Path) -> Path | None:
    markers = sorted(root.glob("attempt-*/complete.json"))
    require(len(markers) <= 1, "HOSTING_MULTIPLE_COMPLETE_ATTEMPTS", "more than one complete block attempt exists")
    return markers[0] if markers else None
