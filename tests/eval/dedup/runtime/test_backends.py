import json
import subprocess
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


@pytest.mark.parametrize("name", ["NVIDIA B200", "NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"])
def test_local_gpu_probe_accepts_supported_replicas(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def probe(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        device = argv[2]
        return subprocess.CompletedProcess(argv, 0, stdout=f"{device}, {name}, GPU-{device}, 81559\n")

    monkeypatch.setattr(backends.subprocess, "run", probe)

    assert len(backends._local_gpus(["2", "5"])) == 2
    assert [argv[2] for argv in calls] == ["2", "5"]


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [""],
        ["0, NVIDIA H100 80GB HBM3"],
        ["0, NVIDIA H100 80GB HBM3, GPU-0, 81559\n1, NVIDIA H100 80GB HBM3, GPU-1, 81559"],
        ["0, NVIDIA A100, GPU-0, 81559"],
        ["0, NVIDIA H1000, GPU-0, 81559"],
        ["0, NVIDIA H100 80GB HBM3, , 81559"],
        ["0, NVIDIA H100 80GB HBM3, GPU-0, 81559", "1, NVIDIA B200, GPU-1, 183359"],
        ["0, NVIDIA H100 80GB HBM3, GPU-0, 81559", "0, NVIDIA H100 80GB HBM3, GPU-0, 81559"],
    ],
)
def test_local_gpu_probe_rejects_unsupported_or_ambiguous_devices(
    outputs: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def probe(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=outputs[int(argv[2])])

    monkeypatch.setattr(backends.subprocess, "run", probe)

    with pytest.raises(DedupEvaluationError, match="V07_LOCAL_GPU"):
        backends._local_gpus([str(index) for index in range(len(outputs))])


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
