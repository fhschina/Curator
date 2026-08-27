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

"""Strict loading of frozen evaluation configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.dedup.validation import DedupEvaluationError, assert_exact_keys, require, sha256_json


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    dataset_version: str
    expected_rows: int
    expected_grouped_documents: int
    expected_groups: int
    expected_removals: int
    expected_singletons: int
    expected_retained: int
    embedding_rows: int
    embedding_dimensions: int
    embedding_dtype: str
    embedding_sha256: str


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    kind: str
    model_id: str
    revision: str
    cache_root: Path


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    backend: str
    base_url: str
    model: str
    api_key_env: str
    structured_output_mode: str
    thinking: bool
    temperature: float
    top_p: float
    max_output_tokens: int
    concurrency: int
    requests_per_minute: int
    timeout_seconds: float
    max_retries: int
    max_visible_tokens: int
    window_tokens: int
    window_overlap_tokens: int
    prompt_version: str
    schema_version: str


@dataclass(frozen=True, slots=True)
class LocalNddJudgeConfig:
    backend: str
    model: str
    model_path: Path
    runner_config: Path
    ray_temp_dir: Path
    checkpoint_root: Path
    num_cpus: int | None
    num_gpus: int
    max_retries: int
    max_visible_tokens: int
    window_tokens: int
    window_overlap_tokens: int
    prompt_version: str
    schema_version: str
    visible_payload_version: str


AnyJudgeConfig = JudgeConfig | LocalNddJudgeConfig


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    backend: str
    minhash_seed: int
    char_ngram_width: int
    num_hashes: int
    feature_ngram_width: int
    lsh_grid: tuple[tuple[int, int], ...]
    pilot_target_min: int
    pilot_target_max: int
    pilot_target_center: int
    top_k: int
    signature_chunk_rows: int
    semantic_chunk_rows: int
    max_candidates_per_anchor: int


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    name: str
    anchor_quotas: dict[str, int]
    removal_pair_budget: int
    cross_group_pair_budget: int
    qa_pair_budget: int
    minimum_diff_budget: int
    formal_v0: bool

    @property
    def anchor_count(self) -> int:
        return sum(self.anchor_quotas.values())


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    schema_version: int
    handoff_root: Path
    output_root: Path
    cache_root: Path
    verify_checksums: bool
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    judge: AnyJudgeConfig
    retrieval: RetrievalConfig
    seeds: dict[str, int]
    canonical_pair_id_version: str
    profiles: dict[str, ProfileConfig]
    source_path: Path
    raw: dict[str, Any]

    @property
    def digest(self) -> str:
        return sha256_json(self.raw)

    def profile(self, name: str) -> ProfileConfig:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise DedupEvaluationError(
                "UNKNOWN_PROFILE", "requested profile is not defined", profile=name, available=sorted(self.profiles)
            ) from exc


_TOP_LEVEL = {
    "schema_version",
    "handoff_root",
    "output_root",
    "cache_root",
    "verify_checksums",
    "dataset",
    "tokenizer",
    "judge",
    "retrieval",
    "seeds",
    "canonical_pair_id_version",
    "profiles",
}


def _object(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), "INVALID_CONFIG_TYPE", "configuration section must be an object", context=context)
    return value


def _load_dataset(value: dict[str, Any]) -> DatasetConfig:
    fields = set(DatasetConfig.__dataclass_fields__)
    assert_exact_keys(value, fields, context="dataset")
    return DatasetConfig(**value)


def _load_tokenizer(value: dict[str, Any], base: Path) -> TokenizerConfig:
    fields = set(TokenizerConfig.__dataclass_fields__) - {"cache_root"}
    assert_exact_keys(value, fields | {"cache_root"}, context="tokenizer")
    return TokenizerConfig(
        kind=str(value["kind"]),
        model_id=str(value["model_id"]),
        revision=str(value["revision"]),
        cache_root=_resolve_path(value["cache_root"], base),
    )


def _load_judge(value: dict[str, Any], base: Path) -> AnyJudgeConfig:
    if value.get("backend") == "local_ndd":
        fields = set(LocalNddJudgeConfig.__dataclass_fields__)
        assert_exact_keys(value, fields, context="judge")
        config = LocalNddJudgeConfig(
            **{
                **value,
                "model_path": _resolve_path(value["model_path"], base),
                "runner_config": _resolve_path(value["runner_config"], base),
                "ray_temp_dir": _resolve_path(value["ray_temp_dir"], base),
                "checkpoint_root": _resolve_path(value["checkpoint_root"], base),
            }
        )
        require(
            (config.prompt_version, config.schema_version)
            == ("dedup-judge-sarah-minhash-v1", "dedup-judge-output-v0"),
            "INVALID_JUDGE_CONTRACT",
            "local_ndd requires the Sarah MinHash prompt and v0-compatible output contract",
        )
        require(
            config.visible_payload_version == "judge-visible-payload-v2",
            "INVALID_JUDGE_CONFIG",
            "local_ndd must use the metadata-free v2 visible payload",
        )
        require(config.num_gpus == 1, "INVALID_JUDGE_CONFIG", "the default local_ndd contract uses one GPU")
    else:
        fields = set(JudgeConfig.__dataclass_fields__)
        assert_exact_keys(value, fields, context="judge")
        config = JudgeConfig(**value)
    supported_contracts = {
        ("dedup-judge-v0", "dedup-judge-output-v0"),
        ("dedup-judge-v1", "dedup-judge-output-v1"),
        ("dedup-judge-sarah-minhash-v1", "dedup-judge-output-v0"),
    }
    require(
        (config.prompt_version, config.schema_version) in supported_contracts,
        "INVALID_JUDGE_CONTRACT",
        "judge prompt and output schema versions must be a supported matching pair",
        prompt_version=config.prompt_version,
        schema_version=config.schema_version,
    )
    require(config.max_retries == 2, "INVALID_JUDGE_CONFIG", "judge retry count must be exactly two")
    require(config.backend in {"stub", "nvidia_openai", "local_ndd"}, "INVALID_JUDGE_CONFIG", "unknown judge backend")
    require(
        config.window_overlap_tokens < config.window_tokens, "INVALID_JUDGE_CONFIG", "window overlap must be smaller"
    )
    return config


def _load_retrieval(value: dict[str, Any]) -> RetrievalConfig:
    fields = set(RetrievalConfig.__dataclass_fields__)
    assert_exact_keys(value, fields, context="retrieval")
    grid = tuple(tuple(int(item) for item in row) for row in value["lsh_grid"])
    require(all(len(row) == 2 for row in grid), "INVALID_LSH_GRID", "each LSH grid item must contain bands and rows")
    normalized = {**value, "lsh_grid": grid}
    config = RetrievalConfig(**normalized)
    require(
        all(bands * rows <= config.num_hashes for bands, rows in grid),
        "INVALID_LSH_GRID",
        "LSH configuration uses more hashes than are available",
    )
    return config


def _load_profile(name: str, value: dict[str, Any]) -> ProfileConfig:
    required = {
        "anchor_quotas",
        "removal_pair_budget",
        "cross_group_pair_budget",
        "qa_pair_budget",
        "minimum_diff_budget",
        "formal_v0",
    }
    assert_exact_keys(value, required, context=f"profiles.{name}")
    quotas = _object(value["anchor_quotas"], f"profiles.{name}.anchor_quotas")
    expected = {"singleton", "size_2", "size_3_5", "size_6_20", "size_21_plus"}
    assert_exact_keys(quotas, expected, context=f"profiles.{name}.anchor_quotas")
    return ProfileConfig(
        name=name,
        anchor_quotas={key: int(item) for key, item in quotas.items()},
        **{key: value[key] for key in required - {"anchor_quotas"}},
    )


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path, *, path_base: Path | None = None) -> EvaluationConfig:
    source = Path(path).resolve()
    base = path_base.resolve() if path_base is not None else source.parent
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DedupEvaluationError("CONFIG_NOT_FOUND", "configuration file does not exist", path=str(source)) from exc
    except json.JSONDecodeError as exc:
        raise DedupEvaluationError("INVALID_CONFIG_JSON", "configuration JSON is invalid", path=str(source)) from exc
    raw = _object(raw, "root")
    assert_exact_keys(raw, _TOP_LEVEL, context="root")
    require(
        raw["schema_version"] == 1, "UNSUPPORTED_CONFIG_VERSION", "only configuration schema version 1 is supported"
    )
    seeds = _object(raw["seeds"], "seeds")
    seed_fields = {"pilot_seed", "anchor_seed", "pair_seed", "judge_order_seed", "qa_seed"}
    assert_exact_keys(seeds, seed_fields, context="seeds")
    profiles_raw = _object(raw["profiles"], "profiles")
    require({"smoke", "full"}.issubset(profiles_raw), "MISSING_PROFILE", "smoke and full profiles are required")
    config = EvaluationConfig(
        schema_version=1,
        handoff_root=_resolve_path(raw["handoff_root"], base),
        output_root=_resolve_path(raw["output_root"], base),
        cache_root=_resolve_path(raw["cache_root"], base),
        verify_checksums=bool(raw["verify_checksums"]),
        dataset=_load_dataset(_object(raw["dataset"], "dataset")),
        tokenizer=_load_tokenizer(_object(raw["tokenizer"], "tokenizer"), base),
        judge=_load_judge(_object(raw["judge"], "judge"), base),
        retrieval=_load_retrieval(_object(raw["retrieval"], "retrieval")),
        seeds={key: int(value) for key, value in seeds.items()},
        canonical_pair_id_version=str(raw["canonical_pair_id_version"]),
        profiles={
            name: _load_profile(name, _object(value, f"profiles.{name}")) for name, value in profiles_raw.items()
        },
        source_path=source,
        raw=raw,
    )
    require(
        config.dataset.expected_grouped_documents == config.dataset.expected_groups + config.dataset.expected_removals
        and config.dataset.expected_retained == config.dataset.expected_singletons + config.dataset.expected_groups
        and config.dataset.expected_rows
        == config.dataset.expected_singletons + config.dataset.expected_grouped_documents,
        "DATASET_ACCOUNTING_INCONSISTENT",
        "dataset counts do not satisfy the one-keeper-per-group accounting identities",
    )
    require(config.canonical_pair_id_version == "cp1", "UNSUPPORTED_PAIR_ID_VERSION", "V0 requires cp1")
    if config.retrieval.backend == "fixture_cpu":
        require(
            config.dataset.dataset_version.startswith("fixture-")
            and config.dataset.expected_rows <= 10_000
            and config.judge.backend in {"local_ndd", "stub"}
            and config.tokenizer.kind == "whitespace",
            "FIXTURE_BACKEND_SCOPE_INVALID",
            "fixture_cpu is restricted to small stub/local_ndd judge test corpora",
        )
    if config.retrieval.backend != "fixture_cpu":
        require(
            config.handoff_root == Path("/raid/vjawa/dedup_eval_10M")
            and config.output_root == Path("/raid/hfang/dedup_eval_runs")
            and config.cache_root == Path("/raid/hfang/dedup_eval_cache")
            and str(config.tokenizer.cache_root).startswith("/raid/hfang/"),
            "V0_RUNTIME_PATH_MISMATCH",
            "production handoff, output, and cache roots are frozen under /raid",
        )
        expected_seeds = {
            "pilot_seed": 26081200,
            "anchor_seed": 26081201,
            "pair_seed": 26081202,
            "judge_order_seed": 26081203,
            "qa_seed": 26081204,
        }
        require(config.seeds == expected_seeds, "V0_SEED_MISMATCH", "production V0 seeds are frozen")
        require(
            set(config.profiles) == {"smoke", "full"},
            "V0_PROFILE_MISMATCH",
            "only smoke and full profiles are allowed",
        )
        expected_profiles = {
            "smoke": (
                {"singleton": 10, "size_2": 2, "size_3_5": 2, "size_6_20": 3, "size_21_plus": 3},
                50,
                50,
                100,
                False,
            ),
            "full": (
                {"singleton": 500, "size_2": 125, "size_3_5": 125, "size_6_20": 125, "size_21_plus": 125},
                10_000,
                10_000,
                200,
                config.judge.backend == "nvidia_openai",
            ),
        }
        for name, (quotas, removal, cross_group, qa, formal) in expected_profiles.items():
            profile = config.profiles[name]
            actual = (
                profile.anchor_quotas,
                profile.removal_pair_budget,
                profile.cross_group_pair_budget,
                profile.qa_pair_budget,
                profile.minimum_diff_budget,
                profile.formal_v0,
            )
            require(
                actual == (quotas, removal, cross_group, qa, 0, formal),
                "V0_PROFILE_MISMATCH",
                "production V0 profile is not the frozen proposal profile",
                profile=name,
            )
        dataset_actual = (
            config.dataset.expected_rows,
            config.dataset.expected_grouped_documents,
            config.dataset.expected_groups,
            config.dataset.expected_removals,
            config.dataset.expected_singletons,
            config.dataset.expected_retained,
            config.dataset.embedding_rows,
            config.dataset.embedding_dimensions,
            config.dataset.embedding_dtype,
        )
        require(
            dataset_actual
            == (10_008_061, 2_021_220, 352_601, 1_668_619, 7_986_841, 8_339_442, 10_008_061, 768, "float32"),
            "V0_DATASET_CONTRACT_MISMATCH",
            "production V0 dataset accounting is frozen",
        )
        require(
            config.dataset.dataset_version == "CC-MAIN-2025-26-dense-10m-v0"
            and config.dataset.embedding_sha256 == "d2830786a29c8e3c92b8549e031abf0dd6c302b142a448404a4fb07be951ca6a",
            "V0_DATASET_CONTRACT_MISMATCH",
            "production dataset identity and embedding checksum are frozen",
        )
        require(config.verify_checksums, "V0_CHECKSUMS_DISABLED", "production V0 requires checksum verification")
        require(
            config.tokenizer.kind == "huggingface" and config.tokenizer.model_id == "deepseek-ai/DeepSeek-V4-Pro",
            "V0_TOKENIZER_MISMATCH",
            "production V0 requires the official DeepSeek V4 Pro tokenizer",
        )
        if isinstance(config.judge, JudgeConfig):
            require(
                config.judge.backend == "nvidia_openai"
                and config.judge.base_url == "https://inference-api.nvidia.com/v1"
                and config.judge.model
                in {
                    "nvidia/deepseek-ai/deepseek-v4-pro",
                    "nvidia/deepseek-ai/deepseek-v4-flash",
                }
                and config.judge.api_key_env == "NVIDIA_API_KEY",
                "V0_JUDGE_MISMATCH",
                "production judge provider contract is not an approved DeepSeek V4 configuration",
            )
            require(
                config.judge.structured_output_mode in {"json_schema", "json_object_plus_local_schema"}
                and not config.judge.thinking
                and config.judge.temperature == 0
                and config.judge.max_output_tokens == 1024
                and config.judge.concurrency == 8
                and config.judge.requests_per_minute == 60
                and config.judge.timeout_seconds == 180
                and config.judge.max_visible_tokens == 65_536
                and config.judge.window_tokens == 8_192
                and config.judge.window_overlap_tokens == 1_024,
                "V0_JUDGE_MISMATCH",
                "production V0 judge execution parameters are frozen",
            )
        else:
            require(
                config.judge.backend == "local_ndd"
                and config.judge.model == "Qwen/Qwen3.8-27B"
                and config.judge.runner_config.is_file()
                and str(config.judge.model_path).startswith("/raid/hfang/")
                and str(config.judge.ray_temp_dir).startswith("/raid/hfang/")
                and str(config.judge.checkpoint_root).startswith("/raid/hfang/")
                and config.judge.max_visible_tokens == 65_536
                and config.judge.window_tokens == 8_192
                and config.judge.window_overlap_tokens == 1_024,
                "LOCAL_NDD_JUDGE_MISMATCH",
                "production local_ndd configuration differs from the approved Sarah/Qwen contract",
            )
        require(
            config.retrieval.backend == "gpu_cudf"
            and config.retrieval.minhash_seed == 42
            and config.retrieval.char_ngram_width == 24
            and config.retrieval.lsh_grid == ((5, 1), (6, 1), (7, 1), (8, 1))
            and config.retrieval.pilot_target_min == 20
            and config.retrieval.pilot_target_max == 50
            and config.retrieval.pilot_target_center == 35
            and config.retrieval.num_hashes == 260
            and config.retrieval.feature_ngram_width == 5
            and config.retrieval.top_k == 50
            and config.retrieval.max_candidates_per_anchor == 250_000,
            "V0_RETRIEVAL_MISMATCH",
            "production V0 retrieval contract is frozen",
        )
    return config
