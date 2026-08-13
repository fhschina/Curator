# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.dedup.config import load_config
from eval.dedup.run import create_run
from eval.dedup.validation import DedupEvaluationError


def test_example_config_loads() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    config = load_config(path)
    assert config.profile("smoke").anchor_count == 20
    assert config.profile("full").removal_pair_budget == 10_000
    assert config.judge.max_retries == 2
    assert config.judge.model == "nvidia/deepseek-ai/deepseek-v4-flash"
    assert config.retrieval.lsh_grid == ((5, 1), (6, 1), (7, 1), (8, 1))
    assert config.retrieval.max_candidates_per_anchor == 250_000


def test_full_config_freezes_pro_and_immutable_tokenizer_revision() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.full.json"
    config = load_config(path)

    assert config.profile("full").formal_v0 is True
    assert config.profile("full").anchor_count == 1_000
    assert config.judge.model == "nvidia/deepseek-ai/deepseek-v4-pro"
    assert config.judge.structured_output_mode == "json_schema"
    assert config.tokenizer.revision == "b5968e9190ef611bbf34a7229255be88a0e937c1"


def test_flash_config_cannot_create_formal_full_run() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"

    with pytest.raises(DedupEvaluationError) as error:
        create_run(load_config(path), "full", evaluation_run_id="must-not-be-created")

    assert error.value.issue.code == "FORMAL_V0_JUDGE_MODEL_MISMATCH"


def test_config_rejects_unknown_field(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["silent_default"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(DedupEvaluationError) as error:
        load_config(path)
    assert error.value.issue.code == "UNKNOWN_CONFIG_FIELDS"


def test_production_config_rejects_changed_frozen_seed(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["seeds"]["pilot_seed"] += 1
    path = tmp_path / "changed-seed.json"
    path.write_text(json.dumps(value))
    with pytest.raises(DedupEvaluationError) as error:
        load_config(path)
    assert error.value.issue.code == "V0_SEED_MISMATCH"
