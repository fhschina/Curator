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

"""Step 6: review saved main results with coverage and optional subject verification.

Run from the repository root:
    python tutorials/eval/dedup/6_run_critics.py \
        --input-path output/dedup_eval/judged_pairs \
        --output-path output/dedup_eval/reviewed_pairs

Reuses the main judge YAML's serving/model settings, but does not run its judges
or score filters. Main results must already be present in the input records.
Add --subject to check fixed subject-conflict proposals after coverage, sharing
the same service and Pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import critics
import ray
from critics.coverage import CoverageCritic
from critics.stages import CriticApplyStage, CriticPrepareStage
from critics.subject import SubjectCritic, SubjectVerifier

from nemo_curator.core.client import RayClient
from nemo_curator.eval.llm_judge.workflow import LLMJudgeWorkflow, build_config_builder
from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage

if TYPE_CHECKING:
    import data_designer.config as dd

_CONFIG_DIR = Path(__file__).resolve().parent / "judge_config"
_PROMPT_DIR = _CONFIG_DIR / "critics" / "coverage"


@dataclass
class CoverageWorkflow(LLMJudgeWorkflow):
    """Use the shared workflow lifecycle with a tutorial-specific structured column."""

    source_judge: str = "pair_semantic_judgment"
    model_alias: str | None = None
    subject: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        models = self.config["models"]
        if self.model_alias is None:
            self.model_alias = str(models[0]["alias"])
        if self.model_alias not in {model["alias"] for model in models}:
            message = f"Unknown coverage model alias: {self.model_alias!r}"
            raise ValueError(message)
        self._source_stage = next(
            (
                stage
                for stage in self.config["execution"]["stages"]
                if any(judge["name"] == self.source_judge for judge in stage["judges"])
            ),
            None,
        )
        if self._source_stage is None:
            message = f"Unknown source judge: {self.source_judge!r}"
            raise ValueError(message)
        self._critic = CoverageCritic(self.source_judge)
        self._prompt = (_PROMPT_DIR / "pair.jinja").read_text(encoding="utf-8")
        self._system_prompt = (_PROMPT_DIR / "system.jinja").read_text(encoding="utf-8")
        self._subject_prompts = []
        if self.subject:
            directory = _CONFIG_DIR / "critics" / "subject"
            for prefix in ("", "verifier_"):
                self._subject_prompts.append(
                    (
                        (directory / f"{prefix}pair.jinja").read_text(encoding="utf-8"),
                        (directory / f"{prefix}system.jinja").read_text(encoding="utf-8"),
                    )
                )
        self._user_postprocessing = list(self.postprocessing_stages)
        self.preprocessing_stages = [*self.preprocessing_stages, CriticPrepareStage(self._critic)]
        self.postprocessing_stages = [CriticApplyStage(self._critic), *self.postprocessing_stages]

    def _build_judge_stages(
        self, *, endpoint: str
    ) -> list[
        tuple[
            str,
            dd.DataDesignerConfigBuilder,
            list[dd.ModelProvider],
            dict[str, object] | None,
            int | None,
            list[dict[str, object]],
        ]
    ]:
        builder, providers = build_config_builder(
            self.config_path, endpoint=endpoint, models=self.config["models"], judges=[]
        )
        builder.add_column(
            self._critic.build_column(
                model_alias=self.model_alias, prompt=self._prompt, system_prompt=self._system_prompt
            )
        )
        if self.subject:
            subject_critic, verifier = SubjectCritic(), SubjectVerifier()
            subject_stages = [CriticPrepareStage(subject_critic)]
            for critic, (prompt, system_prompt) in zip((subject_critic, verifier), self._subject_prompts, strict=True):
                subject_builder, subject_providers = build_config_builder(
                    self.config_path, endpoint=endpoint, models=self.config["models"], judges=[]
                )
                subject_builder.add_column(
                    critic.build_column(model_alias=self.model_alias, prompt=prompt, system_prompt=system_prompt)
                )
                subject_stages.extend(
                    [
                        DataDesignerStage(config_builder=subject_builder, model_providers=subject_providers).with_(
                            name=f"ndd_{critic.name}",
                            runtime_env=self._source_stage.get("runtime_env"),
                            num_workers=self._source_stage.get("num_workers"),
                        ),
                        CriticApplyStage(critic),
                    ]
                )
            self.postprocessing_stages = [
                CriticApplyStage(self._critic),
                *subject_stages,
                *self._user_postprocessing,
            ]
        # Dynamo startup detaches from Ray; distribute helpers in the subsequent pipeline job.
        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": [critics]})
        return [
            (
                "coverage",
                builder,
                providers,
                self._source_stage.get("runtime_env"),
                self._source_stage.get("num_workers"),
                [],
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-config", default=str(_CONFIG_DIR / "fuzzy_pair_judge.yaml"))
    parser.add_argument("--input-path", required=True, help="Saved main-judge output from step 5.")
    parser.add_argument("--output-path", required=True, help="A separate directory for reviewed pairs.")
    parser.add_argument("--input-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--output-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--source-judge", default="pair_semantic_judgment")
    parser.add_argument("--model-alias", default=None, help="Model alias from the YAML; defaults to the first model.")
    parser.add_argument("--subject", action="store_true", help="Verify subject-conflict proposals after coverage.")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--ray-temp-dir", default="/tmp/ray")  # noqa: S108
    args = parser.parse_args()
    workflow = CoverageWorkflow(
        judge_config=args.judge_config,
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.input_format,
        output_format=args.output_format,
        source_judge=args.source_judge,
        model_alias=args.model_alias,
        subject=args.subject,
        checkpoint_path=args.checkpoint_path,
    )
    with RayClient(ray_temp_dir=args.ray_temp_dir):
        try:
            workflow.run()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
