from pathlib import Path

import pytest
import yaml

from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.deploy.configure_h100 import configure
from eval.dedup.runtime.backends import DEFAULT_LOCAL_RUNNER_CONFIG, _runner_contract


@pytest.mark.parametrize("disable_deep_gemm", [False, True])
def test_profile_preserves_frozen_runner_and_scopes_worker_settings(tmp_path: Path, disable_deep_gemm: bool) -> None:
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin/nvcc").touch()
    original = DEFAULT_LOCAL_RUNNER_CONFIG.read_bytes()
    path = configure(tmp_path / "profile", toolkit, disable_deep_gemm=disable_deep_gemm, max_jobs=4)
    config = yaml.safe_load(path.read_text())
    env = config["models"][0]["dynamo_model"]["runtime_env"].pop("env_vars")
    assert env.pop("CUDA_HOME") == str(toolkit)
    assert env.pop("MAX_JOBS") == "4"
    assert env == ({"VLLM_USE_DEEP_GEMM": "0"} if disable_deep_gemm else {})
    assert config == yaml.safe_load(original)
    assert DEFAULT_LOCAL_RUNNER_CONFIG.read_bytes() == original
    assert _runner_contract(path)["engine"] == _runner_contract(DEFAULT_LOCAL_RUNNER_CONFIG)["engine"]
    for asset in ("pair.jinja", "system.jinja", "cutlass_compat/sitecustomize.py"):
        assert (path.parent / asset).read_bytes() == (DEFAULT_LOCAL_RUNNER_CONFIG.parent / asset).read_bytes()
    with pytest.raises(DedupEvaluationError, match="H100_PROFILE"):
        configure(path.parent, toolkit)


def test_profile_rejects_release_destination_before_creating_files(tmp_path: Path) -> None:
    with pytest.raises(DedupEvaluationError, match="H100_PROFILE"):
        configure(DEFAULT_LOCAL_RUNNER_CONFIG.parent / "generated", tmp_path)


def test_profile_requires_cuda_compiler(tmp_path: Path) -> None:
    output = tmp_path / "profile"
    with pytest.raises(DedupEvaluationError, match="H100_CUDA"):
        configure(output, tmp_path / "missing")
    assert not output.exists()
