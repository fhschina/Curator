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

"""Run conditional text generation in a Curator Pipeline using an existing model endpoint."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import data_designer.config as dd

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer.data_designer import DataDesignerStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter

SEED_RECORDS = [
    {"id": "row_1", "text": "The library opens at nine and closes at six on weekdays.", "should_generate": True},
    {"id": "row_2", "text": "The park has a walking trail and a small pond.", "should_generate": False},
    {
        "id": "row_3",
        "text": "The train leaves at noon and arrives in the city two hours later.",
        "should_generate": True,
    },
    {"id": "row_4", "text": "The cafe serves breakfast all day and closes on Sundays.", "should_generate": False},
]


def parse_args() -> argparse.Namespace:
    """Read the endpoint, model, and local output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible API base URL, including /v1")
    parser.add_argument("--model", required=True, help="Model identifier served by the endpoint")
    parser.add_argument("--output-path", required=True, type=Path, help="Local directory for output JSONL files")
    return parser.parse_args()


def build_config(model: str) -> dd.DataDesignerConfigBuilder:
    """Generate a summary only for rows whose should_generate flag is true."""
    builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias="example_model",
                model=model,
                provider="example_provider",
                skip_health_check=True,
                inference_parameters=dd.ChatCompletionInferenceParams(temperature=0.0, max_tokens=2048),
            )
        ]
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="generated_text",
            model_alias="example_model",
            prompt="Summarize this text in one sentence: {{ text }}",
            # A true condition skips this cell, leaving its default value of None.
            skip=dd.SkipConfig(when="{{ not should_generate }}"),
        )
    )
    return builder


def validate_output(records: list[dict]) -> None:
    """Check row preservation and conditional generation without assuming output order."""
    expected_ids = {record["id"] for record in SEED_RECORDS}
    if len(records) != len(SEED_RECORDS) or {record.get("id") for record in records} != expected_ids:
        msg = "Output must contain every input ID exactly once."
        raise ValueError(msg)

    by_id = {record["id"]: record for record in records}
    for seed in SEED_RECORDS:
        record = by_id[seed["id"]]
        if record.get("text") != seed["text"] or record.get("should_generate") is not seed["should_generate"]:
            msg = f"Original fields changed for {seed['id']}."
            raise ValueError(msg)
        if "generated_text" not in record:
            msg = f"Missing generated_text column for {seed['id']}."
            raise ValueError(msg)
        generated = record["generated_text"]
        if seed["should_generate"]:
            if not isinstance(generated, str) or not generated.strip():
                msg = f"Expected nonempty generated text for {seed['id']}."
                raise ValueError(msg)
        elif generated is not None:
            msg = f"Expected JSON null for skipped row {seed['id']}."
            raise ValueError(msg)


def main() -> None:
    """Run the reader, conditional generator, and writer, then validate their output."""
    args = parse_args()
    provider = dd.ModelProvider(
        name="example_provider",
        endpoint=args.endpoint,
        provider_type="openai",
        api_key=os.environ.get("OPENAI_API_KEY") or "unused",
    )

    with TemporaryDirectory(prefix="ndd_conditional_seed_") as seed_dir:
        seed_path = Path(seed_dir) / "seed.jsonl"
        seed_path.write_text("".join(json.dumps(record) + "\n" for record in SEED_RECORDS), encoding="utf-8")
        pipeline = Pipeline(
            name="ndd_conditional_generation",
            stages=[
                JsonlReader(file_paths=str(seed_path), fields=["id", "text", "should_generate"]),
                DataDesignerStage(config_builder=build_config(args.model), model_providers=[provider]),
                JsonlWriter(path=str(args.output_path.resolve())),
            ],
        )
        print(pipeline.describe())
        with RayClient(num_cpus=4, num_gpus=0, include_dashboard=False):
            output_tasks = pipeline.run(executor=RayDataExecutor())

    # Read only files returned by this run, so previous output cannot affect validation.
    records = []
    for task in output_tasks or []:
        for file_path in task.data:
            print(f"Output file: {file_path}")
            with Path(file_path).open(encoding="utf-8") as output_file:
                records.extend(json.loads(line) for line in output_file if line.strip())
    validate_output(records)
    for record in sorted(records, key=lambda row: row["id"]):
        print(json.dumps(record, ensure_ascii=False))
    selected = sum(record["should_generate"] for record in SEED_RECORDS)
    print(
        f"Validation passed: input={len(SEED_RECORDS)}, output={len(records)}, "
        f"selected={selected}, skipped={len(SEED_RECORDS) - selected} (row counts, not HTTP request counts)."
    )


if __name__ == "__main__":
    main()
