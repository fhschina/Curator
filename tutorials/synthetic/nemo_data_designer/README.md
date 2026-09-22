# NeMo Data Designer Synthetic Data Generation

This tutorial shows how to use NeMo Data Designer with NeMo Curator to generate synthetic medical notes from seed symptom and diagnosis data. It downloads a small CSV dataset, converts it into JSONL seed records, builds a Data Designer configuration with prompt templates and samplers, and runs the generation workflow through a Curator pipeline.

The tutorial supports both local and remote inference. By default, it starts a local Ray Serve + vLLM `InferenceServer` for `openai/gpt-oss-20b`; users can also set a remote provider such as NVIDIA NIM and point Data Designer at that endpoint instead.

Use `ndd_data_generation_example.ipynb` for the notebook walkthrough or `ndd_data_generation_example.py` for the script version of the same workflow.

## Conditional Generation with SkipConfig

`ndd_conditional_generation_example.py` is a minimal example of conditional generation in a generic Curator `Pipeline`. It uses the existing `DataDesignerStage` with Data Designer's `SkipConfig`, independently of LLM judge workflows or deduplication evaluation.

Use an existing environment with Curator's `sdg_cpu` extra (Data Designer 0.9.1) and an already deployed OpenAI-compatible model endpoint. Activate that environment and run from the Curator repository root:

```bash
# If the endpoint requires authentication, set OPENAI_API_KEY in your environment.
# PYTHONPATH ensures the driver and local Ray workers use this checkout.
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
    tutorials/synthetic/nemo_data_designer/ndd_conditional_generation_example.py \
    --endpoint http://localhost:8000/v1 \
    --model YOUR_SERVED_MODEL_NAME \
    --output-path /tmp/ndd_conditional_output
```

All three arguments are required. `--endpoint` is the API base URL, including `/v1`; `--model` is the endpoint's served model identifier; `--output-path` is a local output directory. The script reads `OPENAI_API_KEY`, using a placeholder when authentication is unnecessary. It starts a CPU Ray cluster when `RAY_ADDRESS` is unset; run with `RAY_ADDRESS` unset for a standalone local demonstration. The script does not start an inference server or request GPUs for its stages.

The script writes four temporary seed records with `id`, `text`, and a boolean `should_generate` field, then runs:

```text
JsonlReader → DataDesignerStage → JsonlWriter
```

The `generated_text` column uses `skip=dd.SkipConfig(when="{{ not should_generate }}")`. A true `when` expression skips generation for that cell; its default value is `None`, serialized as JSON `null`. All four rows pass through the pipeline, with their original fields preserved:

| id | should_generate | generated_text |
|---|---|---|
| row_1 | true | A generated summary |
| row_2 | false | `null` |
| row_3 | true | A generated summary |
| row_4 | false | `null` |

The script validates the current run's output by ID, prints each record, and reports:

```text
Validation passed: input=4, output=4, selected=2, skipped=2 (row counts, not HTTP request counts).
```

Missing or duplicate IDs, changed seed fields, empty selected results, and non-null skipped results fail validation. Model failures that prevent valid output also fail the run. These are row counts, not measured endpoint request counts; retries can increase requests. Temporary seed data and the Ray cluster started by the script are cleaned up, while output JSONL files remain available for inspection.

## Local InferenceServer Notes

When using the local server, the tutorial detects the number of Ray-visible GPUs and uses that value as vLLM `tensor_parallel_size`.

On some PCIe-only multi-GPU systems, NCCL peer-to-peer initialization can hang during vLLM startup. This is most likely to appear when `tensor_parallel_size > 1`.

If the tutorial hangs while starting the local inference server, try one of the following:

```bash
export NCCL_P2P_DISABLE=1
```

or edit the inference-server cell to use single-GPU tensor parallelism:

```python
tensor_parallel_size = 1
```

Then pass that value into `engine_kwargs`:

```python
engine_kwargs={
    "tensor_parallel_size": tensor_parallel_size,
}
```

`NCCL_P2P_DISABLE=1` allows multi-GPU serving to continue but may reduce communication performance. Setting `tensor_parallel_size=1` avoids cross-GPU NCCL collectives for vLLM while leaving the remaining GPUs visible to Ray and the tutorial. Restart Ray or the tutorial kernel after changing NCCL environment variables.
