# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""CLI for the controlled Local B200 versus Inference Hub benchmark."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .config import load_config
from .metrics import summarize
from .recovery import create_headline_validation_run, create_recovery_run
from .runner import run_benchmark, static_preflight
from .workload import prepare_workload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze the paired workload")
    prepare.add_argument("--config", required=True)
    recover = commands.add_parser("recover", help="create an audited recovery run")
    recover.add_argument("--parent-run-root", required=True)
    recover.add_argument("--through-block", required=True)
    validate_c8 = commands.add_parser("validate-c8", help="replay the exact three concurrency-8 blocks")
    validate_c8.add_argument("--parent-run-root", required=True)
    for command in ("preflight", "run", "summarize"):
        child = commands.add_parser(command)
        child.add_argument("--run-root", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result: Any
    if args.command == "prepare":
        result = {"run_root": str(prepare_workload(load_config(args.config)))}
    elif args.command == "recover":
        result = {"run_root": str(create_recovery_run(args.parent_run_root, through_block=args.through_block))}
    elif args.command == "validate-c8":
        result = {"run_root": str(create_headline_validation_run(args.parent_run_root))}
    elif args.command == "preflight":
        result = static_preflight(args.run_root)
    elif args.command == "run":
        result = run_benchmark(args.run_root)
    else:
        result = summarize(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
