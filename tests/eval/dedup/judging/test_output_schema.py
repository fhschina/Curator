import json
from pathlib import Path

from eval.dedup.judging.output_schema import (
    judge_output_schema,
    unresolved_judge_output,
    validate_judge_output,
)

RELEASE = Path(__file__).parents[4] / "eval/dedup/release"


def test_release_schema_matches_selected_python_contract() -> None:
    assert json.loads((RELEASE / "judge_schema.json").read_text(encoding="utf-8")) == judge_output_schema()


def test_unresolved_output_is_schema_and_policy_valid() -> None:
    value = unresolved_judge_output()

    assert validate_judge_output(value) == value
    assert value["same_duplicate_group"] == "UNRESOLVED"
    assert value["evidence"] == []
