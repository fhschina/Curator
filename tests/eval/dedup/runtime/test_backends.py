import json
from pathlib import Path

import pytest

from eval.dedup.core.validation import DedupEvaluationError
from eval.dedup.runtime import backends


def test_historical_manifest_defaults_to_hub() -> None:
    assert backends._backend({}) == "hub"
    assert backends._backend({"backend": "local"}) == "local"


def test_local_devices_are_explicit_unique_and_nonempty() -> None:
    assert backends._parse_devices("0, 2") == ["0", "2"]
    for value in ("", "0,0", "-1"):
        with pytest.raises(DedupEvaluationError, match="V07_LOCAL_DEVICE"):
            backends._parse_devices(value)


def test_release_local_runner_freezes_selected_engine_settings() -> None:
    value = backends._runner_contract(backends.DEFAULT_LOCAL_RUNNER_CONFIG)

    assert value["path"] == str(backends.DEFAULT_LOCAL_RUNNER_CONFIG)
    assert value["engine"] == {
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.8,
        "enforce_eager": True,
    }
    assert Path(value["path"]).is_file()


def test_hub_prepare_cli_forwards_smoke_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []
    monkeypatch.setattr(
        backends.release,
        "prepare",
        lambda root, **kwargs: calls.append((root, kwargs)) or {"mode": backends.release.SMOKE_ONLY_MODE},
    )

    assert (
        backends.main(
            [
                "prepare",
                "--root",
                str(tmp_path / "run"),
                "--source-run",
                str(tmp_path / "source"),
                "--smoke-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"mode": backends.release.SMOKE_ONLY_MODE}
    assert calls == [
        (
            (tmp_path / "run").resolve(),
            {"source": (tmp_path / "source").resolve(), "smoke_only": True},
        )
    ]
