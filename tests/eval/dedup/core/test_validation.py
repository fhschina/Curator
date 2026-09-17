from pathlib import Path

import pytest

from eval.dedup.core.validation import DedupEvaluationError, read_json, write_json_atomic


def test_immutable_json_write_accepts_identical_content_only(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"value": 1})
    write_json_atomic(path, {"value": 1})

    with pytest.raises(DedupEvaluationError, match="IMMUTABLE_ARTIFACT_COLLISION"):
        write_json_atomic(path, {"value": 2})

    assert read_json(path) == {"value": 1}
