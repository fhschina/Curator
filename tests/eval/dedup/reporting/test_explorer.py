from pathlib import Path

from eval.dedup.reporting.explorer import _clip_visible_text, pair_explorer_destination


def test_pair_explorer_output_is_run_scoped(tmp_path: Path) -> None:
    assert pair_explorer_destination(tmp_path / "final_report.md") == tmp_path / "pair_explorer.html"
    assert pair_explorer_destination(tmp_path / "final_report.sample.md") == tmp_path / "pair_explorer.sample.html"


def test_visible_text_is_bounded() -> None:
    value = _clip_visible_text("a" * 2_000, center=1_000, limit=100)
    assert len(value) == 102
    assert value.startswith("…")
    assert value.endswith("…")
