# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Serial tokenizer initialization before the frozen pilot's concurrent workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dedup.analysis import retention_v4_experiment as experiment
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

CONTRACT = "retention-v4-serial-tokenizer-execution-v1"


def warmup() -> object:
    """Transformers' lazy import must complete before simultaneous first requests."""
    return experiment.local_tokenizer()


def prepare(root: Path, prior: Path) -> dict:
    root, prior = root.resolve(), prior.resolve()
    experiment.reference.verify_freeze(prior / "manifest.json")
    completion = experiment.preflight.read(prior / "complete.json")
    require(
        completion["assessment_sha256"] == sha256_file(prior / "assessment.json"),
        "RETENTION_EXECUTION_PRIOR",
        "preserve a completed, bound predecessor pilot",
    )
    previous = experiment.preflight.read(prior / "manifest.json")
    current = experiment.prepare(root)
    require(
        all(
            current[k] == previous[k]
            for k in ("version", "model", "endpoint", "generation", "population", "repeat_count")
        ),
        "RETENTION_EXECUTION_POLICY",
        "startup repair must not alter the experiment",
    )
    require(
        sha256_file(prior / "main_requests.json") == sha256_file(root / "main_requests.json")
        and sha256_file(prior / "panel_private.json") == sha256_file(root / "panel_private.json"),
        "RETENTION_EXECUTION_REQUESTS",
        "identical frozen population and main request bodies required",
    )
    sources = {
        str(p): sha256_file(p)
        for p in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[3] / "tests/eval/dedup/test_retention_v4_execution.py",
            prior / "manifest.json",
            prior / "assessment.json",
            prior / "complete.json",
            root / "manifest.json",
        )
    }
    result = {
        "contract_version": CONTRACT,
        "status": "EXECUTION_REPAIR_NOT_NEW_SEMANTIC_VERSION",
        "prior": str(prior),
        "sources": sources,
        "artifacts": {},
        "tokenizer_initialization": "SERIAL_BEFORE_WORKERS",
        "semantic_changes": False,
        "old_responses_reused": False,
    }
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "execution_manifest.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    root = root.resolve()
    experiment.reference.verify_freeze(root / "execution_manifest.json")
    require(
        experiment.preflight.read(root / "execution_manifest.json")["contract_version"] == CONTRACT,
        "RETENTION_EXECUTION_VERSION",
        "explicit startup repair contract required",
    )
    warmup()
    return experiment.run(root, env_file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    require(
        args.prior is not None if args.command == "prepare" else args.env_file is not None,
        "RETENTION_EXECUTION_ARGUMENT",
        "explicit predecessor or existing credential source required",
    )
    result = prepare(args.root, args.prior) if args.command == "prepare" else run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"sources", "artifacts", "response_artifacts"}}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
