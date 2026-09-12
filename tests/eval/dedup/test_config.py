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

from eval.dedup.config import (
    HS_MINHASH_PROMPT_VERSION,
    HS_MINHASH_V061_PROMPT_VERSION,
    HS_V062_PROMPT_VERSION,
    HS_V0621_PROMPT_VERSION,
    HS_V06210_PROMPT_VERSION,
    HS_V06211_POLICY_PROMPT_VERSION,
    HS_V06211_PROMPT_VERSION,
    HS_V06212_PROMPT_VERSION,
    HS_V06212_ROUTE_PROMPT_VERSION,
    HS_V06213_EXACT_PROMPT_VERSION,
    HS_V06213_PROMPT_VERSION,
    HS_V06214_CONTROL_PROMPT_VERSION,
    HS_V06214_PROMPT_VERSION,
    HS_V06215_CONTROL_PROMPT_VERSION,
    HS_V06215_PROMPT_VERSION,
    HS_V0622_PROMPT_VERSION,
    HS_V0623_PROMPT_VERSION,
    HS_V0624_PROMPT_VERSION,
    HS_V0625_PROMPT_VERSION,
    HS_V0626_PROMPT_VERSION,
    HS_V0627_PROMPT_VERSION,
    HS_V0628_PROMPT_VERSION,
    HS_V0629_PROMPT_VERSION,
    load_config,
)
from eval.dedup.run import create_run
from eval.dedup.validation import DedupEvaluationError


def test_example_config_loads() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    config = load_config(path)
    assert config.profile("smoke").anchor_count == 20
    assert config.profile("full").removal_pair_budget == 10_000
    assert config.profile("full").formal_v0 is False
    assert config.judge.max_retries == 2
    assert config.judge.backend == "local_ndd"
    assert config.judge.model == "Qwen/Qwen3.8-27B"
    assert config.judge.visible_payload_version == "judge-visible-payload-v2"
    assert config.retrieval.lsh_grid == ((5, 1), (6, 1), (7, 1), (8, 1))
    assert config.retrieval.max_candidates_per_anchor == 250_000


def test_full_config_freezes_pro_and_immutable_tokenizer_revision() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.legacy_nvidia.full.json"
    config = load_config(path)

    assert config.profile("full").formal_v0 is True
    assert config.profile("full").anchor_count == 1_000
    assert config.judge.model == "nvidia/deepseek-ai/deepseek-v4-pro"
    assert config.judge.structured_output_mode == "json_schema"
    assert config.tokenizer.revision == "b5968e9190ef611bbf34a7229255be88a0e937c1"


def test_flash_config_cannot_create_formal_full_run() -> None:
    path = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.legacy_nvidia.example.json"

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


def test_config_rejects_mismatched_judge_prompt_and_schema(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.legacy_nvidia.example.json"
    value = json.loads(source.read_text())
    value["judge"]["prompt_version"] = "dedup-judge-v1"
    path = tmp_path / "mismatched-judge-contract.json"
    path.write_text(json.dumps(value))

    with pytest.raises(DedupEvaluationError) as error:
        load_config(path)

    assert error.value.issue.code == "INVALID_JUDGE_CONTRACT"


def test_config_accepts_matching_v1_judge_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.legacy_nvidia.example.json"
    value = json.loads(source.read_text())
    value["judge"]["prompt_version"] = "dedup-judge-v1"
    value["judge"]["schema_version"] = "dedup-judge-output-v1"
    path = tmp_path / "v1-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == "dedup-judge-v1"
    assert config.judge.schema_version == "dedup-judge-output-v1"


def test_config_accepts_hs_minimal_ab_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_qwen.yaml")
    value["judge"]["prompt_version"] = HS_MINHASH_PROMPT_VERSION
    path = tmp_path / "hs-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_MINHASH_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v0"
    assert config.judge.runner_config.name == "hs_qwen.yaml"


def test_config_accepts_hs_v061_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v061_qwen.yaml")
    value["judge"]["prompt_version"] = HS_MINHASH_V061_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v2"
    path = tmp_path / "hs-v061-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_MINHASH_V061_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v2"
    assert config.judge.runner_config.name == "hs_v061_qwen.yaml"


def test_config_accepts_hs_v062_semantic_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v062_qwen.yaml")
    value["judge"]["prompt_version"] = HS_V062_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v062-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V062_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v062_qwen.yaml"


def test_config_accepts_hs_v0621_semantic_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0621_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0621_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v0621-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0621_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v0621_qwen_c64.yaml"


def test_config_accepts_hs_v0622_semantic_ledger_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0622_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0622_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v0622-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0622_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v0622_qwen_c64.yaml"


def test_config_accepts_hs_v0623_reviewed_boundary_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0623_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0623_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v0623-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0623_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v0623_qwen_c64.yaml"


def test_config_accepts_hs_v0624_verified_identity_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0624_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0624_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v0624-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0624_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v0624_qwen_c64.yaml"


def test_config_accepts_hs_v0625_boundary_delta_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0625_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0625_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    path = tmp_path / "hs-v0625-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0625_PROMPT_VERSION
    assert config.judge.schema_version == "dedup-judge-output-v3"
    assert config.judge.runner_config.name == "hs_v0625_qwen_c64.yaml"


def test_config_accepts_hs_v0626_span_payload_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0626_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0626_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    value["judge"]["visible_payload_version"] = "judge-visible-payload-v3"
    path = tmp_path / "hs-v0626-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0626_PROMPT_VERSION
    assert config.judge.visible_payload_version == "judge-visible-payload-v3"
    assert config.judge.runner_config.name == "hs_v0626_qwen_c64.yaml"


def test_config_accepts_hs_v0627_record_binding_critic_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0627_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0627_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    value["judge"]["visible_payload_version"] = "judge-visible-payload-v3"
    path = tmp_path / "hs-v0627-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0627_PROMPT_VERSION
    assert config.judge.visible_payload_version == "judge-visible-payload-v3"
    assert config.judge.runner_config.name == "hs_v0627_qwen_c64.yaml"


def test_config_accepts_hs_v0628_calibrated_critic_contract(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "hs_v0628_qwen_c64.yaml")
    value["judge"]["prompt_version"] = HS_V0628_PROMPT_VERSION
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    value["judge"]["visible_payload_version"] = "judge-visible-payload-v3"
    path = tmp_path / "hs-v0628-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == HS_V0628_PROMPT_VERSION
    assert config.judge.visible_payload_version == "judge-visible-payload-v3"
    assert config.judge.runner_config.name == "hs_v0628_qwen_c64.yaml"


@pytest.mark.parametrize(
    ("version", "runner"),
    [
        (HS_V0629_PROMPT_VERSION, "hs_v0629_qwen_c64.yaml"),
        (HS_V06210_PROMPT_VERSION, "hs_v06210_qwen_c64.yaml"),
        (HS_V06211_POLICY_PROMPT_VERSION, "hs_v06211_policy_qwen_c64.yaml"),
        (HS_V06211_PROMPT_VERSION, "hs_v06211_qwen_c64.yaml"),
        (HS_V06212_ROUTE_PROMPT_VERSION, "hs_v06212_route_qwen_c64.yaml"),
        (HS_V06212_PROMPT_VERSION, "hs_v06212_qwen_c64.yaml"),
        (HS_V06213_EXACT_PROMPT_VERSION, "hs_v06213_exact_qwen_c64.yaml"),
        (HS_V06213_PROMPT_VERSION, "hs_v06213_qwen_c64.yaml"),
        (HS_V06214_CONTROL_PROMPT_VERSION, "hs_v06214_arbitration_qwen_c64.yaml"),
        (HS_V06214_PROMPT_VERSION, "hs_v06214_qwen_c64.yaml"),
        (HS_V06215_CONTROL_PROMPT_VERSION, "hs_v06215_arbitration_qwen_c64.yaml"),
        (HS_V06215_PROMPT_VERSION, "hs_v06215_qwen_c64.yaml"),
    ],
)
def test_config_accepts_versioned_boundary_critic_contract(tmp_path: Path, version: str, runner: str) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / runner)
    value["judge"]["prompt_version"] = version
    value["judge"]["schema_version"] = "dedup-judge-output-v3"
    value["judge"]["visible_payload_version"] = "judge-visible-payload-v3"
    path = tmp_path / "hs-v0629-judge-contract.json"
    path.write_text(json.dumps(value))

    config = load_config(path)

    assert config.judge.prompt_version == version
    assert config.judge.visible_payload_version == "judge-visible-payload-v3"
    assert config.judge.runner_config.name == runner


def test_config_rejects_builtin_runner_prompt_mismatch(tmp_path: Path) -> None:
    source = Path(__file__).parents[3] / "eval" / "dedup" / "resources" / "v0_config.example.json"
    value = json.loads(source.read_text())
    value["judge"]["runner_config"] = str(source.parent / "local_ndd" / "sarah_minhash_qwen.yaml")
    value["judge"]["prompt_version"] = HS_MINHASH_PROMPT_VERSION
    path = tmp_path / "mismatched-local-ndd-contract.json"
    path.write_text(json.dumps(value))

    with pytest.raises(DedupEvaluationError) as error:
        load_config(path)

    assert error.value.issue.code == "INVALID_JUDGE_CONTRACT"
    assert error.value.issue.details["expected_prompt_version"] == "dedup-judge-sarah-minhash-v1"
