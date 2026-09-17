from pathlib import Path

import pytest

from nemo_curator.eval.llm_judge import runtime as subject


def test_ray_temp_path_validation_rejects_socket_overflow(tmp_path: Path) -> None:
    subject._validate_ray_temp_dir("/tmp/ray")  # noqa: S108 - validates the documented short default

    with pytest.raises(ValueError, match="too long"):
        subject._validate_ray_temp_dir(tmp_path / ("x" * 100))


def test_external_runtime_freezes_cpu_only_ray_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def initialize(_self: object, config_path: str | Path, **kwargs: object) -> None:
        captured.update(config_path=config_path, **kwargs)

    monkeypatch.setattr(subject.JudgePipelineRuntime, "__init__", initialize)
    subject.ExternalJudgeRuntime("judge.yaml", endpoint="https://judge.invalid/v1", provider_api_key="secret")

    assert captured["config_path"] == "judge.yaml"
    assert captured["endpoint"] == "https://judge.invalid/v1"
    assert captured["provider_api_key"] == "secret"
    assert captured["num_gpus"] == 0


def test_local_runtime_stops_ray_when_server_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = object.__new__(subject.LocalJudgeRuntime)
    runtime.config = {}
    runtime.models = []
    runtime.config_path = Path("judge.yaml")
    stopped: list[bool] = []

    def start(instance: object) -> object:
        return instance

    def stop(_instance: object) -> None:
        stopped.append(True)

    def fail(*_args: object, **_kwargs: object) -> None:
        message = "boom"
        raise RuntimeError(message)

    monkeypatch.setattr(subject.JudgePipelineRuntime, "start", start)
    monkeypatch.setattr(subject.JudgePipelineRuntime, "stop", stop)
    monkeypatch.setattr(subject, "_start_inference_server", fail)

    with pytest.raises(RuntimeError, match="boom"):
        runtime.start()

    assert stopped == [True]
