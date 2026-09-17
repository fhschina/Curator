from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[3]
SOURCE = ROOT / "eval/dedup"
FORBIDDEN_SUFFIXES = {".csv", ".docx", ".html", ".log", ".pptx"}
HISTORICAL_NAME = re.compile(r"(?:^|[_-])(?:v?0?62\d*|exp(?:1|5|6))(?:[_-]|$)", re.IGNORECASE)
HISTORICAL_IMPORT = re.compile(r"(?:^|\.)(?:v?0?62\w*|exp(?:1|5|6)\w*)(?:\.|$)", re.IGNORECASE)


def _source_files() -> list[Path]:
    return sorted(path for path in SOURCE.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


def test_dedup_tree_stays_small_and_contains_no_run_artifacts() -> None:
    files = _source_files()
    assert len(files) <= 100
    assert sum(path.stat().st_size for path in files) <= 5 * 1024 * 1024
    assert not [path for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not [path for path in files if HISTORICAL_NAME.search(path.name)]


def test_stable_source_has_no_historical_imports_or_machine_paths() -> None:
    violations = []
    for path in _source_files():
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".json", ".jinja", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "/raid/hfang" in text or re.search(r"\b(?:10|192\.168)\.\d+\.\d+\.\d+\b", text):
            violations.append(f"machine path: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            if any(HISTORICAL_IMPORT.search(name) for name in imports):
                violations.append(f"historical import: {path.relative_to(ROOT)}")
    assert not violations


def test_runtime_does_not_depend_on_analysis_modules() -> None:
    for path in (SOURCE / "runtime").glob("*.py"):
        assert "eval.dedup.analysis" not in path.read_text(encoding="utf-8")


def test_dedup_tree_has_no_lfs_managed_files() -> None:
    paths = [str(path.relative_to(ROOT)) for path in _source_files()]
    git = shutil.which("git")
    assert git is not None
    result = subprocess.run(  # noqa: S603 - fixed git inspection command
        [git, "check-attr", "filter", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not [line for line in result.stdout.splitlines() if line.endswith(": lfs")]
