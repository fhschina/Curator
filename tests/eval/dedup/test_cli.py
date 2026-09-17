from __future__ import annotations

from typing import TYPE_CHECKING

from eval.dedup import cli, recommended

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_version_entry_points(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == "v0.7.1\n"

    assert recommended.main(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "v0.7.1\n"
    assert "DEPRECATED" in captured.err


def test_env_loader_is_non_overriding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\nexport CURATOR_V07_SOURCE_RUN='/source'\nKEEP=from-file\n", encoding="utf-8")
    monkeypatch.setenv("KEEP", "from-process")
    monkeypatch.delenv("CURATOR_V07_SOURCE_RUN", raising=False)

    cli._load_repository_env(path)

    assert cli.os.environ["CURATOR_V07_SOURCE_RUN"] == "/source"
    assert cli.os.environ["KEEP"] == "from-process"
