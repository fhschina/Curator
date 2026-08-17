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

"""Stable command-line interface for the operational dedup evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from eval.dedup.config import EvaluationConfig, load_config
from eval.dedup.handoff.corpus import TokenCounter
from eval.dedup.handoff.manifests import register_corpus_handoff, register_sut_handoff
from eval.dedup.judging.client import create_judge_client
from eval.dedup.judging.payload import align_evidence_offsets, validate_evidence_offsets
from eval.dedup.judging.schema import parse_judge_json, validate_judge_output
from eval.dedup.report import (
    import_human_qa,
    pair_explorer_destination,
    publish_human_qa_report,
    publish_report,
)
from eval.dedup.run import create_run, load_run, run_pipeline, run_status, validate_run
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, write_json_atomic

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPORT_EXPORT_ROOT = _REPOSITORY_ROOT.parent / "dedup_eval_runs"


def _load_repository_env(dotenv_path: Path | None = None) -> None:
    """Load repo-local secrets without overriding an explicit process environment."""

    path = dotenv_path if dotenv_path is not None else _REPOSITORY_ROOT / ".env"
    if path.is_file():
        load_dotenv(dotenv_path=path, override=False, verbose=False)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _probe_judge(
    client: Any,
    *,
    prompt: str,
    payload: dict[str, Any],
    max_retries: int,
) -> int:
    """Apply the frozen judge retry budget to the provider capability probe."""

    errors = []
    request_prompt = prompt
    for attempt in range(max_retries + 1):
        try:
            raw = client.judge(system_prompt=request_prompt, payload=payload)
            aligned, _ = align_evidence_offsets(parse_judge_json(raw), payload)
            result = validate_judge_output(aligned)
            validate_evidence_offsets(result, payload)
            return attempt + 1
        except Exception as exc:  # noqa: BLE001 - provider and schema failures share one retry policy
            errors.append(
                {
                    "attempt": attempt + 1,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc)[:1000],
                }
            )
            failure = str(exc).replace("\n", " ")[:400]
            request_prompt = (
                f"{prompt}\n\nREPAIR RETRY: the previous response failed strict local validation: {failure}. "
                "Return a fresh JSON object with every required key exactly once and exact evidence offsets."
            )
    raise DedupEvaluationError(
        "STRUCTURED_OUTPUT_ATTEMPTS_EXHAUSTED",
        "judge probe exhausted the frozen schema-validation retry budget",
        attempts=max_retries + 1,
        errors=errors,
    )


def preflight(config: EvaluationConfig, profile_name: str) -> dict[str, Any]:
    """Perform all blocking local and provider checks without creating a run."""

    profile = config.profile(profile_name)
    require(
        config.retrieval.backend == "fixture_cpu"
        or not profile.formal_v0
        or config.judge.model == "nvidia/deepseek-ai/deepseek-v4-pro",
        "FORMAL_V0_JUDGE_MODEL_MISMATCH",
        "formal V0 requires the proposal judge model; V4 Flash is allowed only for NON-V0 smoke",
        judge_model=config.judge.model,
    )
    require(
        config.handoff_root.is_dir(),
        "HANDOFF_ROOT_NOT_FOUND",
        "handoff root does not exist",
        path=str(config.handoff_root),
    )
    output_parent = config.output_root
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    require(
        os.access(output_parent, os.W_OK),
        "OUTPUT_ROOT_NOT_WRITABLE",
        "nearest output parent is not writable",
        path=str(output_parent),
    )
    cache_parent = config.cache_root
    while not cache_parent.exists() and cache_parent != cache_parent.parent:
        cache_parent = cache_parent.parent
    require(
        os.access(cache_parent, os.W_OK),
        "CACHE_ROOT_NOT_WRITABLE",
        "nearest cache parent is not writable",
        path=str(cache_parent),
    )
    output_free = shutil.disk_usage(output_parent).free
    cache_free = shutil.disk_usage(cache_parent).free
    minimum_free = 500 * 1024**3
    require(
        config.retrieval.backend == "fixture_cpu" or (output_free >= minimum_free and cache_free >= minimum_free),
        "INSUFFICIENT_RUNTIME_DISK",
        "production preflight requires at least 500 GiB free on output and cache filesystems",
        output_free_bytes=output_free,
        cache_free_bytes=cache_free,
    )
    with tempfile.TemporaryDirectory(prefix="dedup-preflight-") as directory:
        temporary = Path(directory)
        corpus_path = temporary / "corpus_manifest.json"
        sut_path = temporary / "sut_run_manifest.json"
        corpus = register_corpus_handoff(config, corpus_path)
        sut = register_sut_handoff(config, corpus, sut_path)
    tokenizer = TokenCounter(config.tokenizer)
    tokenizer_contract = tokenizer.contract()
    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    require(
        gpu_result.returncode == 0,
        "GPU_PREFLIGHT_FAILED",
        "nvidia-smi could not enumerate GPUs",
        stderr=gpu_result.stderr,
    )
    gpus = [line for line in gpu_result.stdout.splitlines() if line.strip()]
    require(
        gpus or config.retrieval.backend == "fixture_cpu",
        "NO_GPU_AVAILABLE",
        "production retrieval requires at least one GPU",
    )
    if config.retrieval.backend != "fixture_cpu":
        require(
            len(gpus) == 8 and all("B200" in gpu for gpu in gpus),
            "GPU_TOPOLOGY_MISMATCH",
            "production V0 requires exactly eight visible NVIDIA B200 GPUs",
            visible_gpus=gpus,
        )
    judge_probe = {"backend": config.judge.backend, "structured_output_mode": config.judge.structured_output_mode}
    if config.judge.backend != "stub":
        require(
            os.environ.get(config.judge.api_key_env, "").strip(),
            "MISSING_API_KEY",
            "judge API key is unavailable",
            local_checks_passed=True,
            handoff_rows=corpus["dataset_row_count"],
            shard_count=len(corpus["shards"]),
            grouped_documents=sut["duplicate_groups"]["rows"],
            group_count=sut["duplicate_groups"]["groups"],
            removal_count=sut["removal_ids"]["rows"],
            tokenizer=tokenizer_contract,
            visible_gpus=gpus,
            output_free_bytes=output_free,
            cache_free_bytes=cache_free,
        )
        client = create_judge_client(config.judge)
        fixture_payload = {
            "payload_schema_version": "judge-visible-payload-v1",
            "document_a": {
                "metadata": {"url": None, "crawl_timestamp": None, "language": "en", "character_count": 5},
                "text": "alpha",
            },
            "document_b": {
                "metadata": {"url": None, "crawl_timestamp": None, "language": "en", "character_count": 5},
                "text": "alpha",
            },
            "long_document_evidence": {"truncated": False, "token_counts": {"A": 1, "B": 1}, "windows": []},
        }
        prompt = (Path(__file__).resolve().parent / "resources" / "judge_prompt_v0.txt").read_text(encoding="utf-8")
        try:
            probe_attempts = _probe_judge(
                client,
                prompt=prompt,
                payload=fixture_payload,
                max_retries=config.judge.max_retries,
            )
        except Exception as primary_error:
            if config.judge.structured_output_mode != "json_schema":
                raise DedupEvaluationError(
                    "STRUCTURED_OUTPUT_PROBE_FAILED",
                    "configured judge structured-output mode failed its local schema probe",
                    mode=config.judge.structured_output_mode,
                    error_type=primary_error.__class__.__name__,
                ) from primary_error
            fallback = replace(config.judge, structured_output_mode="json_object_plus_local_schema")
            try:
                fallback_client = create_judge_client(fallback)
                _probe_judge(
                    fallback_client,
                    prompt=prompt,
                    payload=fixture_payload,
                    max_retries=config.judge.max_retries,
                )
            except Exception as fallback_error:
                raise DedupEvaluationError(
                    "STRUCTURED_OUTPUT_PROBE_FAILED",
                    "both json_schema and JSON mode failed the local schema probe",
                    json_schema_error_type=primary_error.__class__.__name__,
                    json_mode_error_type=fallback_error.__class__.__name__,
                ) from fallback_error
            raise DedupEvaluationError(
                "STRUCTURED_OUTPUT_FALLBACK_REQUIRED",
                "provider JSON mode passed; freeze judge.structured_output_mode as "
                "json_object_plus_local_schema and rerun preflight",
                failed_mode="json_schema",
                supported_mode="json_object_plus_local_schema",
            ) from primary_error
        judge_probe["probe"] = "passed"
        judge_probe["attempts"] = probe_attempts
    return {
        "status": "passed",
        "profile": profile.name,
        "formal_v0": profile.formal_v0,
        "handoff": {
            "rows": corpus["dataset_row_count"],
            "shards": len(corpus["shards"]),
            "sut_run_id": sut["sut_run_id"],
            "removals": sut["removal_ids"]["rows"],
        },
        "disk": {"output_free_bytes": output_free, "cache_free_bytes": cache_free},
        "tokenizer": tokenizer_contract,
        "gpus": gpus,
        "judge": judge_probe,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval.dedup", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate all blocking inputs and external capabilities"
    )
    preflight_parser.add_argument("--config", type=Path, required=True)
    preflight_parser.add_argument("--profile", choices=("smoke", "full"), required=True)

    run_parser = subparsers.add_parser("run", help="create and execute a new immutable evaluation run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    run_parser.add_argument("--through-step", type=int, default=10)

    resume_parser = subparsers.add_parser("resume", help="validate and resume the first incomplete stage")
    resume_parser.add_argument("--run-root", type=Path, required=True)
    resume_parser.add_argument("--through-step", type=int, default=10)

    for name in ("status", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-root", type=Path, required=True)

    export_parser = subparsers.add_parser("qa-export", help="create or report the frozen blind human-QA packet")
    export_parser.add_argument("--run-root", type=Path, required=True)

    import_parser = subparsers.add_parser("qa-import", help="validate and import completed human-QA labels")
    import_parser.add_argument("--run-root", type=Path, required=True)
    import_parser.add_argument("--labels", type=Path, required=True)
    import_parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_REPORT_EXPORT_ROOT,
        help="root for run-scoped human-QA report exports",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="render a versioned automated report from completed immutable artifacts",
    )
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_REPORT_EXPORT_ROOT,
        help="root for run-scoped report exports",
    )
    report_parser.add_argument(
        "--output-label",
        default="automated_v3",
        help="safe filename label for a derived report revision",
    )
    return parser


def _qa_export(context: Any) -> dict[str, Any]:
    packet = context.data / "human_qa_packet.jsonl"
    template = context.data / "human_qa_labels.csv"
    require(
        packet.is_file() and template.is_file(),
        "QA_PACKET_NOT_AVAILABLE",
        "run through Step 6 before exporting the frozen QA packet",
    )
    run_status(context)
    counts = {"qa_pairs": sum(1 for line in packet.open(encoding="utf-8") if line.strip())}
    return {**counts, "packet": str(packet), "labels_template": str(template)}


def _report_export_directory(context: Any, output_root: Path) -> Path:
    return output_root.expanduser().resolve() / context.evaluation_run_id / "v0_run" / "reports"


def main(argv: list[str] | None = None) -> int:
    _load_repository_env()
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            _print(preflight(load_config(args.config), args.profile))
        elif args.command == "run":
            context = create_run(load_config(args.config), args.profile)
            _print(run_pipeline(context, through_step=args.through_step))
        elif args.command == "resume":
            _print(run_pipeline(load_run(args.run_root), through_step=args.through_step))
        elif args.command == "status":
            _print(run_status(load_run(args.run_root)))
        elif args.command == "validate":
            result = validate_run(load_run(args.run_root))
            _print(result)
            return 0 if result["valid"] else 1
        elif args.command == "qa-export":
            _print(_qa_export(load_run(args.run_root)))
        elif args.command == "report":
            context = load_run(args.run_root)
            require(
                re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}", args.output_label) is not None,
                "REPORT_OUTPUT_LABEL_INVALID",
                "report output label must contain only letters, numbers, dot, dash, or underscore",
            )
            status = run_status(context)
            require(
                all(item["status"] == "complete" for item in status["stages"][:9]),
                "REPORT_TOO_EARLY",
                "Steps 1-9 must be complete before report rendering",
            )
            report_directory = _report_export_directory(context, args.output_root)
            report_path = report_directory / f"final_report.{args.output_label}.md"
            recommendations_path = report_directory / f"recommendations.{args.output_label}.json"
            manifest_path = report_directory / f"report_generation_manifest.{args.output_label}.json"
            dashboard_path = pair_explorer_destination(report_path)
            require(
                not any(path.exists() for path in (report_path, recommendations_path, manifest_path, dashboard_path)),
                "REPORT_OUTPUT_EXISTS",
                "versioned automated report output already exists",
            )
            result = publish_report(
                profile=context.profile,
                run_root=context.run_root,
                recommendation_judge=context.config.judge,
                final_destination=report_path,
                recommendations_destination=recommendations_path,
                manifest_destination=manifest_path,
            )
            _print(
                {
                    **result,
                    "report": str(report_path),
                    "recommendations": str(recommendations_path),
                    "manifest": str(manifest_path),
                    "pair_explorer": str(dashboard_path),
                }
            )
        elif args.command == "qa-import":
            context = load_run(args.run_root)
            require(context.profile.formal_v0, "QA_IMPORT_PROFILE_INVALID", "qa-import is only valid for full V0 runs")
            status = run_status(context)
            require(
                all(item["status"] == "complete" for item in status["stages"][:9]),
                "QA_IMPORT_TOO_EARLY",
                "Steps 1-9 must be complete before QA import",
            )
            counts = import_human_qa(
                packet_path=context.data / "human_qa_packet.jsonl",
                labels_path=args.labels,
                destination=context.data / "human_qa_results.csv",
            )
            qa_results = context.data / "human_qa_results.csv"
            write_json_atomic(
                context.manifests / "human_qa_import.json",
                {
                    "schema_version": "human-qa-import-v1",
                    "evaluation_run_id": context.evaluation_run_id,
                    "source_labels_path": str(args.labels.resolve()),
                    "source_labels_sha256": sha256_file(args.labels),
                    "results_path": str(qa_results),
                    "results_sha256": sha256_file(qa_results),
                    "counts": counts,
                },
            )
            qa_report = publish_human_qa_report(
                packet_path=context.data / "human_qa_packet.jsonl",
                labels_path=qa_results,
                judge_results_path=context.data / "judge_results.jsonl",
                metrics_destination=_report_export_directory(context, args.output_root) / "human_qa_metrics.json",
                report_destination=_report_export_directory(context, args.output_root) / "human_qa_report.md",
            )
            _print(
                {
                    **counts,
                    "destination": str(context.data / "human_qa_results.csv"),
                    "qa_report": qa_report,
                }
            )
        return 0
    except DedupEvaluationError as exc:
        _print({"status": "error", "issue": asdict(exc.issue)})
        return 2
