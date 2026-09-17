# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Create a machine-local runner without editing the frozen release assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from eval.dedup.core.validation import require
from eval.dedup.runtime.backends import DEFAULT_LOCAL_RUNNER_CONFIG


def configure(output: Path, cuda_home: Path, *, disable_deep_gemm: bool = False, max_jobs: int = 8) -> Path:
    output = output.expanduser().resolve()
    cuda_home = cuda_home.expanduser().resolve()
    release = DEFAULT_LOCAL_RUNNER_CONFIG.parent
    require(not output.is_relative_to(release), "H100_PROFILE", "output outside the frozen release runner")
    require(not output.exists(), "H100_PROFILE", "a new profile directory")
    require((cuda_home / "bin/nvcc").is_file(), "H100_CUDA", "CUDA toolkit containing bin/nvcc")
    require(max_jobs > 0, "H100_PROFILE", "positive compiler parallelism")

    config = yaml.safe_load(DEFAULT_LOCAL_RUNNER_CONFIG.read_text(encoding="utf-8"))
    env = config["models"][0]["dynamo_model"]["runtime_env"].setdefault("env_vars", {})
    env.update(CUDA_HOME=str(cuda_home), MAX_JOBS=str(max_jobs))
    if disable_deep_gemm:
        env["VLLM_USE_DEEP_GEMM"] = "0"

    shutil.copytree(release, output, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    path = output / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New runner directory outside the repository.")
    parser.add_argument("--cuda-home", type=Path, required=True, help="CUDA toolkit visible on every worker node.")
    parser.add_argument("--max-jobs", type=int, default=8, help="Maximum compiler jobs per model worker.")
    parser.add_argument(
        "--disable-deep-gemm",
        action="store_true",
        help="Use the fallback validated on CW-DFW H100 with driver 535.216.03.",
    )
    args = parser.parse_args()
    print(configure(args.output, args.cuda_home, disable_deep_gemm=args.disable_deep_gemm, max_jobs=args.max_jobs))


if __name__ == "__main__":
    main()
