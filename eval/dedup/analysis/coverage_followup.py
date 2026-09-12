# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V0.6.2.21 paired coverage evaluation with verified native requests on both arms."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis import coverage_diagnostic as previous
from eval.dedup.analysis.critic_diagnostic import bind_fixed_main
from eval.dedup.judging.payload_transport import bind_transported_payload, encode_repair_feedback
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

ANALYSIS = previous.ANALYSIS
VARIANTS = ("control", "coverage")
CONFIG = previous.RESOURCES / "hs_v06221_coverage_qwen_c64.yaml"
CONTROL_CONFIG = previous.RESOURCES / "hs_v06221_control_qwen_c64.yaml"


def validate_control() -> None:
    old = yaml.safe_load(previous.CONTROL_CONFIG.read_text())
    new = yaml.safe_load(CONTROL_CONFIG.read_text())
    stage = new["execution"]["stages"][0]
    require(
        stage["judges"][0].pop("with_trace") == "all_messages", "FOLLOWUP_CONTROL_TRACE", "control needs raw trace"
    )
    stage["name"] = old["execution"]["stages"][0]["name"]
    require(new == old, "FOLLOWUP_CONTROL_CHANGED", "control may add trace only, not change prompts or model settings")


@lru_cache(maxsize=2)
def renderer(variant: str) -> Any:
    require(variant in VARIANTS, "FOLLOWUP_VARIANT", "unknown arm")
    return previous.coverage_renderer(CONFIG) if variant == "coverage" else previous.message_renderer("control")


@lru_cache(maxsize=1)
def control_schema() -> dict:
    from data_designer.engine.column_generators.utils.prompt_renderer import create_response_recipe

    from eval.llm_judge.run_llm_judge import build_config_builder

    config = yaml.safe_load(CONTROL_CONFIG.read_text())
    builder, _ = build_config_builder(
        CONTROL_CONFIG,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=config["execution"]["stages"][0]["judges"],
    )
    return json.loads(create_response_recipe(builder.get_column_configs()[0]).schema)


def trace_text(content: Any) -> str:
    if isinstance(content, list):
        require(
            bool(content)
            and all(
                isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) for b in content
            ),
            "FOLLOWUP_BOUNDARY_TRACE",
            "native content must contain only text blocks",
        )
        content = "".join(b["text"] for b in content)
    require(isinstance(content, str), "FOLLOWUP_BOUNDARY_TRACE", "native message text is missing")
    return content


def bind(inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
    from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

    packets, raw = previous._index(inputs, "followup inputs"), previous._index(outputs, "followup outputs")
    require(packets.keys() == raw.keys(), "FOLLOWUP_OUTPUT_MISSING", "every input needs exactly one response")
    column = previous.COLUMN if variant == "coverage" else previous.RECORD_BINDING_CRITIC_COLUMN
    predictions = []
    for pid, row in raw.items():
        packet = packets[pid]
        bound, audit = bind_transported_payload(packet, row)
        require(
            row.get("repair_feedback") == packet["repair_feedback"] and isinstance(packet["repair_feedback"], str),
            "FOLLOWUP_BOUNDARY_FEEDBACK",
            "retry feedback changed in transport",
        )
        trace = row.get(column + "__trace")
        require(
            isinstance(trace, list) and len(trace) >= 3, "FOLLOWUP_BOUNDARY_TRACE", "full native conversation required"
        )
        messages = [{"role": m.get("role"), "content": trace_text(m.get("content"))} for m in trace[:2]]
        require(
            messages == renderer(variant)(packet),
            "FOLLOWUP_BOUNDARY_PROMPT",
            "actual initial messages differ from frozen renderer",
        )
        assistants = [m for m in trace if m.get("role") == "assistant"]
        require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "assistant response missing")
        if variant == "coverage":
            prediction = previous.bind_coverage([packet], [bound], mains)[0]
        else:
            require(previous.COLUMN not in row, "FOLLOWUP_BOUNDARY_COLUMN", "control cannot observe coverage output")
            parsed = StructuredResponseRecipe(control_schema(), pruning=False).parse(
                trace_text(assistants[-1].get("content"))
            )
            require(
                parsed == row.get(column),
                "FOLLOWUP_RAW_PRUNED",
                "original control response differs from parsed column",
            )
            prediction = bind_fixed_main([packet], [bound], mains)[0]
        predictions.append(
            {
                **prediction,
                "diagnostic_version": "v0.6.2.21",
                "payload_transport": audit,
                "native_request_messages_sha256": sha256_json(messages),
                "native_trace_sha256": sha256_json(trace),
                "raw_output_sha256": sha256_json(row),
                "native_corrections": len(assistants) - 1,
            }
        )
    return predictions


def fatal_boundary(exc: Exception) -> bool:
    return isinstance(exc, DedupEvaluationError) and exc.issue.code.startswith(
        ("PAYLOAD_TRANSPORT_", "FOLLOWUP_BOUNDARY_", "COVERAGE_COLUMN_LEAK", "CRITIC_MAIN_CALLED")
    )


def run_cell(
    root: Path, inputs: list[dict], mains: dict, runtime: Any, relay: Any, settings: dict, variant: str
) -> dict:
    from eval.dedup.judging.request_relay import RelayContext

    require(not root.exists(), "FOLLOWUP_CELL_EXISTS", "never reuse an observed cell")
    pending, accepted, retried, corrections, fatal = inputs, [], set(), [], False
    column = previous.COLUMN if variant == "coverage" else previous.RECORD_BINDING_CRITIC_COLUMN
    for attempt in range(1, settings["max_retries"] + 2):
        folder = root / f"attempt_{attempt:02d}"
        previous._write_jsonl(folder / "input.jsonl", pending)
        relay.set_context(RelayContext(root.name, attempt, folder / "events.jsonl"))
        runtime.run(
            input_path=str(folder / "input.jsonl"),
            input_format="jsonl",
            output_path=str(folder / "output"),
            output_format="jsonl",
            checkpoint_path=str(folder / "checkpoints"),
            files_per_partition=1,
            inference_parameter_overrides={
                "judge": {
                    "temperature": settings["temperature"],
                    "top_p": settings["top_p"],
                    "max_tokens": settings["max_output_tokens"],
                    "timeout": settings["timeout_seconds"],
                    "max_parallel_requests": settings["max_parallel_requests"],
                }
            },
        )
        outputs = previous._index(previous._read_output_rows(folder / "output"), "followup native outputs")
        require(
            outputs.keys() <= previous._index(pending, "pending").keys(),
            "FOLLOWUP_BOUNDARY_MEMBERSHIP",
            "foreign response pair",
        )
        errors, next_pending = [], []
        for packet in pending:
            pid = packet["canonical_pair_id"]
            row = outputs.get(pid)
            if row is not None:
                trace = row.get(column + "__trace", [])
                count = (
                    sum(isinstance(m, dict) and m.get("role") == "assistant" for m in trace)
                    if isinstance(trace, list)
                    else 0
                )
                corrections.append(
                    {
                        "canonical_pair_id": pid,
                        "outer_attempt": attempt,
                        "assistant_messages": count,
                        "native_corrections": max(0, count - 1),
                    }
                )
            try:
                prediction = bind([packet], [row] if row is not None else [], mains, variant)[0]
                accepted.append({**prediction, "attempts": attempt, "retried": attempt > 1})
            except Exception as exc:  # noqa: BLE001 - preserve every failed pair under the same bounded budget
                feedback = previous._safe_retry_feedback(exc)
                is_fatal = fatal_boundary(exc)
                fatal = fatal or is_fatal
                errors.append({"canonical_pair_id": pid, "issue": feedback, "fatal_boundary": is_fatal})
                next_pending.append({**packet, "repair_feedback": encode_repair_feedback(feedback)})
        write_json_atomic(folder / "validation_errors.json", errors)
        pending = next_pending
        if not pending or fatal:
            break
        if attempt <= settings["max_retries"]:
            retried.update(r["canonical_pair_id"] for r in pending)
    previous._write_jsonl(root / "predictions.jsonl", accepted)
    write_json_atomic(root / "native_corrections.json", corrections)
    write_json_atomic(root / "terminal_pair_ids.json", [p["canonical_pair_id"] for p in pending])
    events = [e for p in root.glob("attempt_*/events.jsonl") for e in previous._jsonl(p)]
    native_pairs = {r["canonical_pair_id"] for r in corrections if r["native_corrections"]}
    complete = {
        "requested": len(inputs),
        "valid": len(accepted),
        "errors": len(pending),
        "retried": len(retried),
        "fatal_boundary": fatal,
        "native_corrections": sum(r["native_corrections"] for r in corrections),
        "native_corrected_pairs": len(native_pairs),
        "judge_retried_pairs": len(retried | native_pairs),
        "http_statuses": dict(Counter(str(e["http_status"]) for e in events)),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {str(p): sha256_file(p) for p in sorted(root.rglob("*.json*")) if p.is_file()},
    }
    write_json_atomic(root / "complete.json", complete)
    return complete


def replay_cell(root: Path, name: str, variant: str, inputs: list[dict], mains: dict) -> tuple[list[dict], dict]:
    folder = root / name
    complete = json.loads((folder / "complete.json").read_text())
    require(
        all(sha256_file(p) == d for p, d in complete["artifacts"].items()),
        "FOLLOWUP_RESULTS_CHANGED",
        "cell artifacts changed",
    )
    predictions = previous._jsonl(folder / "predictions.jsonl")
    packets = previous._index(inputs, "replay inputs")
    published = previous._index(predictions, "replay predictions")
    terminal = json.loads((folder / "terminal_pair_ids.json").read_text())
    require(
        len(predictions) == complete["valid"]
        and len(terminal) == len(set(terminal)) == complete["errors"]
        and not (set(terminal) & published.keys())
        and set(terminal) | published.keys() == packets.keys()
        and complete["requested"] == len(inputs),
        "FOLLOWUP_REPLAY_MEMBERSHIP",
        "all accepted and terminal pairs must be accounted for",
    )
    raw = {}
    for attempt in sorted(folder.glob("attempt_*")):
        attempted = previous._index(previous._jsonl(attempt / "input.jsonl"), "attempt inputs")
        for p in sorted((attempt / "output").glob("*.jsonl")):
            for row in previous._jsonl(p):
                pid = row["canonical_pair_id"]
                raw[(pid, int(attempt.name.split("_")[1]), sha256_json(row))] = (attempted[pid], row)
    for prediction in predictions:
        pid = prediction["canonical_pair_id"]
        key = (pid, prediction["attempts"], prediction["raw_output_sha256"])
        require(key in raw, "FOLLOWUP_RAW_MISSING", "accepted response has no exact attempt/digest match")
        packet, row = raw[key]
        require(
            packet["payload"] == packets[pid]["payload"], "FOLLOWUP_REPLAY_PAYLOAD", "retry changed original payload"
        )
        replay = bind([packet], [row], mains, variant)[0]
        require(
            all(prediction[k] == v for k, v in replay.items()),
            "FOLLOWUP_REPLAY_CHANGED",
            "public result or evidence changed",
        )
    return predictions, {k: v for k, v in complete.items() if k != "artifacts"}


def token_preflight(inputs: list[dict], tokenizer: Any) -> dict:
    rows = []
    for variant in VARIANTS:
        for row in inputs:
            messages = renderer(variant)(row)
            count = len(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(
                count + 4096 + 2048 <= 32768, "FOLLOWUP_TOKEN_BUDGET", "do not truncate or exclude oversized pairs"
            )
            rows.append(
                {
                    "variant": variant,
                    "canonical_pair_id": row["canonical_pair_id"],
                    "input_tokens": count,
                    "message_sha256": sha256_json(messages),
                }
            )
    return {
        "context_budget": 32768,
        "output_budget": 4096,
        "safety_margin": 2048,
        "max_input_tokens": {v: max(r["input_tokens"] for r in rows if r["variant"] == v) for v in VARIANTS},
        "rows": rows,
    }


def prepare(root: Path, parent: Path, boundary: Path) -> dict:
    from transformers import AutoTokenizer

    require(not root.exists(), "FOLLOWUP_ROOT_EXISTS", "use a fresh run root")
    old = previous.validate_freeze(parent)
    require(old["schema_version"] == "dedup-coverage-diagnostic-v1", "FOLLOWUP_PARENT", "expected .20 frozen inputs")
    validate_control()
    report = json.loads(boundary.read_text())
    required_files = {
        str(Path(__file__).resolve()),
        str((ANALYSIS.parents[2] / "tests/eval/dedup/test_coverage_followup.py").resolve()),
    }
    require(
        report["passed"]
        and report["external_model_calls"] == 0
        and report["full_boundary_tests_passed"] == 4
        and required_files <= report["artifacts"].keys()
        and all(sha256_file(p) == h for p, h in report["artifacts"].items()),
        "FOLLOWUP_BOUNDARY_UNVERIFIED",
        "both arms need current full-pipeline initial/retry boundary evidence",
    )
    labels = previous._read_csv(Path(old["paths"]["labels"]))
    previous.validate_projection(labels, previous._read_csv(Path(old["paths"]["reference"])))
    inputs = {
        r: [
            {**p, "repair_feedback": ""}
            for p in previous.blind_rows(previous._jsonl(parent / f"input_repeat_{r}.jsonl"), repeat=r)
        ]
        for r in (1, 2)
    }
    require(
        all(
            len(rows) == 258 and previous._index(rows, "inputs").keys() == previous._index(labels, "labels").keys()
            for rows in inputs.values()
        ),
        "FOLLOWUP_MEMBERSHIP",
        "all frozen pairs required",
    )
    tokens = token_preflight(inputs[1], AutoTokenizer.from_pretrained(previous.TOKENIZER, local_files_only=True))
    files = [
        Path(__file__),
        boundary,
        CONFIG,
        CONTROL_CONFIG,
        parent / "manifest.json",
        parent / "input_freeze.json",
        ANALYSIS / "v06221_design.md",
        ANALYSIS / "v06221_run_protocol.md",
        ANALYSIS.parent / "judging/payload_transport.py",
        previous.RESOURCES / "hs_v06221_coverage_system.jinja",
        previous.RESOURCES / "hs_v06221_coverage_pair.jinja",
        ANALYSIS.parents[2] / "tests/eval/dedup/test_coverage_followup.py",
    ]
    manifest = {
        **{k: v for k, v in old.items() if k not in {"diagnostic_contract_digest", "token_preflight_sha256"}},
        "schema_version": "dedup-coverage-followup-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_diagnostic": str(parent),
        "parent_contract_digest": old["diagnostic_contract_digest"],
        "settings": {**old["settings"], "ray_temp_dir": "/raid/hfang/ihb/r21diag"},
        "paths": {**old["paths"], "protocol": str(ANALYSIS / "v06221_run_protocol.md")},
        "frozen_files": old["frozen_files"] | {str(p.resolve()): sha256_file(p) for p in files},
        "token_preflight_sha256": sha256_json(tokens),
        "variant_contracts": {
            "control": {"runner_config": str(CONTROL_CONFIG), "adapter_policy": "v8", "with_trace": "all_messages"},
            "coverage": {"runner_config": str(CONFIG), "adapter_policy": "dedup-retained-coverage-v1"},
        },
    }
    manifest["diagnostic_contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "fixed_main.json", json.loads((parent / "fixed_main.json").read_text()))
    write_json_atomic(root / "coverage_schema.json", previous.coverage_schema())
    write_json_atomic(root / "token_preflight.json", tokens)
    for r, rows in inputs.items():
        previous._write_jsonl(root / f"input_repeat_{r}.jsonl", rows)
    write_json_atomic(
        root / "input_freeze.json", {str(p): sha256_file(p) for p in root.glob("*.json*") if p.name != "manifest.json"}
    )
    return manifest


def runtime_for(variant: str, endpoint: str, settings: dict) -> Any:
    from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

    if variant == "coverage":
        return previous.CoverageWitnessRuntime(
            CONFIG, endpoint=endpoint, model_name=settings["logical_model"], ray_temp_dir=settings["ray_temp_dir"]
        )
    return ExternalJudgeRuntime(
        CONTROL_CONFIG,
        endpoint=endpoint,
        provider_api_key="unused",
        served_model_overrides={"judge": settings["logical_model"]},
        ray_temp_dir=settings["ray_temp_dir"],
        num_cpus=None,
    )


def run(root: Path) -> None:
    from eval.dedup.judging.request_relay import RequestRelay

    manifest = previous.validate_freeze(root)
    require(not (root / "started.json").exists(), "FOLLOWUP_ALREADY_STARTED", "never restart an observed experiment")
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(bool(key), "FOLLOWUP_CREDENTIAL_MISSING", "inference credential unavailable")
    inputs = {r: previous._jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    write_json_atomic(
        root / "started.json",
        {"at_utc": datetime.now(UTC).isoformat(), "contract": manifest["diagnostic_contract_digest"]},
    )
    try:
        with (
            RequestRelay(
                logical_model=settings["logical_model"],
                upstream_base_url=settings["hub_base_url"],
                upstream_model=settings["hub_model"],
                upstream_api_key=key,
                timeout_seconds=settings["timeout_seconds"],
                expected_generation_parameters={
                    "temperature": settings["temperature"],
                    "top_p": settings["top_p"],
                    "max_tokens": settings["max_output_tokens"],
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            ) as relay,
            runtime_for("control", relay.endpoint, settings),
        ):
            preflights = {}
            for spec in [{"repeat": 0, "variant": v} for v in VARIANTS] + manifest["schedule"]:
                previous.validate_freeze(root)
                repeat, variant = spec["repeat"], spec["variant"]
                if repeat:
                    require(
                        len(preflights) == 2 and all(previous.technical_passed(c) for c in preflights.values()),
                        "FOLLOWUP_PREFLIGHT_FAILED",
                        "failed technical preflight stops formal submission",
                    )
                name = f"repeat_{repeat}_{variant}" if repeat else f"preflight_{variant}"
                batch = inputs[repeat] if repeat else inputs[1][:8]
                with runtime_for(variant, relay.endpoint, settings) as runtime:
                    complete = run_cell(root / name, batch, mains, runtime, relay, settings, variant)
                print(
                    json.dumps({"cell": name, **{k: v for k, v in complete.items() if k != "artifacts"}}),
                    flush=True,
                )
                require(
                    not complete["fatal_boundary"],
                    "FOLLOWUP_BOUNDARY_FAILED",
                    "deterministic boundary failure cannot be repaired by model retries",
                )
                if not repeat:
                    preflights[variant] = complete
                else:
                    require(
                        complete["valid"] == len(batch) and complete["errors"] == 0,
                        "FOLLOWUP_CELL_FAILED",
                        "incomplete formal cell stops the schedule",
                    )
    except Exception as exc:
        write_json_atomic(
            root / "stopped.json",
            {
                "at_utc": datetime.now(UTC).isoformat(),
                "issue": previous._safe_retry_feedback(exc),
                "eligible_for_release": False,
            },
        )
        raise
    previous.validate_freeze(root)
    write_json_atomic(
        root / "run_complete.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "formal_requests": 1032,
            "preflight_requests": 16,
            "eligible_for_release": False,
        },
    )


def summarize(root: Path, output: Path) -> dict:
    manifest = previous.validate_freeze(root)
    inputs = {r: previous._jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    result = {
        "schema_version": "dedup-coverage-followup-assessment-v1",
        "diagnostic_only": True,
        "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
        "eligible_for_release": False,
        "reference_changed": False,
        "supported_for_separate_fresh_local_validation": False,
        "preflights": {},
    }
    for variant in VARIANTS:
        name = f"preflight_{variant}"
        if (root / name / "complete.json").is_file():
            rows, ops = replay_cell(root, name, variant, inputs[1][:8], mains)
            result["preflights"][variant] = {
                "operations": ops,
                "technical_passed": previous.technical_passed(ops),
                "primary_predictions": [
                    {k: row[k] for k in ("canonical_pair_id", *previous.PRIMARY_FIELDS)} for row in rows
                ],
            }
    if not (root / "run_complete.json").is_file():
        require((root / "stopped.json").is_file(), "FOLLOWUP_STILL_RUNNING", "do not score an active or unknown run")
        result.update(
            status="STOPPED_BEFORE_COMPLETE_FORMAL_SCHEDULE",
            stop=json.loads((root / "stopped.json").read_text()),
            completed_formal_cells={},
        )
        for spec in manifest["schedule"]:
            name = f"repeat_{spec['repeat']}_{spec['variant']}"
            if (root / name / "complete.json").is_file():
                _, ops = replay_cell(root, name, spec["variant"], inputs[spec["repeat"]], mains)
                result["completed_formal_cells"][name] = ops
        write_json_atomic(output, result)
        return result
    labels = previous._read_csv(Path(manifest["paths"]["labels"]))
    _, _, baseline, _ = previous.load_run(Path(manifest["baseline_root"]))
    ids = previous._index(labels, "labels").keys()
    baseline = [r for r in baseline if r["canonical_pair_id"] in ids]
    baseline_metrics = previous.evaluate_predictions(labels, baseline)
    negatives = json.loads(Path(manifest["paths"]["negative_guards"]).read_text())["critic_repaired_negative_guards"]
    additional = json.loads(Path(manifest["paths"]["additional_guards"]).read_text())
    targets = [
        {**r, "final": {k: r[f"human_{k}"] for k in previous.PRIMARY_FIELDS}}
        for r in labels
        if r["review_id"] in previous.TARGETS
    ]
    require(
        len(negatives) == 38 and len(additional) == 22 and len(targets) == 3,
        "FOLLOWUP_GUARDS",
        "all fixed guards required",
    )
    cells, predictions = {}, {}
    for spec in manifest["schedule"]:
        name = f"repeat_{spec['repeat']}_{spec['variant']}"
        rows, ops = replay_cell(root, name, spec["variant"], inputs[spec["repeat"]], mains)
        require(
            ops["requested"] == ops["valid"] == 258 and ops["errors"] == 0,
            "FOLLOWUP_INCOMPLETE_FORMAL",
            "all pairs required",
        )
        predictions[name] = rows
        metrics = previous.evaluate_predictions(labels, rows)
        guards = previous.guard_report(labels, rows, baseline, negatives)
        guards["additional_critic_repair_guards"] = previous.evaluate_primary_guards(additional, rows)
        gates = previous.local_gates(metrics, baseline_metrics, guards, ops, 258)
        gates["checks"]["all_judge_retried_pairs_at_most_one_percent"] = ops["judge_retried_pairs"] / 258 <= 0.01
        gates["passed"] = all(gates["checks"].values())
        cells[name] = {
            "metrics": metrics,
            "operations": ops,
            "guards": guards,
            "targets": previous.evaluate_primary_guards(targets, rows),
            "local_gates": gates,
        }
    checks = {
        f"repeat_{r}_coverage": previous.diagnostic_checks(
            cells[f"repeat_{r}_coverage"], cells[f"repeat_{r}_control"], cells[f"repeat_{r}_coverage"]["targets"]
        )
        for r in (1, 2)
    }
    repeats = {
        v: previous.primary_changes(labels, predictions[f"repeat_1_{v}"], predictions[f"repeat_2_{v}"])
        for v in VARIANTS
    }
    stable = len(repeats["coverage"]) <= len(repeats["control"])
    result.update(
        status="FORMAL_SCHEDULE_COMPLETE",
        cells=cells,
        checks=checks,
        within_arm_repeat_changes=repeats,
        repeat_primary_changes_not_above_control=stable,
        paired_changes={
            str(r): previous.primary_changes(
                labels, predictions[f"repeat_{r}_control"], predictions[f"repeat_{r}_coverage"]
            )
            for r in (1, 2)
        },
        supported_for_separate_fresh_local_validation=stable and all(all(c.values()) for c in checks.values()),
    )
    write_json_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.20-coverage-diagnostic"))
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.boundary is not None, "FOLLOWUP_BOUNDARY_REQUIRED", "supply current boundary evidence")
        print(json.dumps({"contract": prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}))
    elif args.action == "run":
        run(args.root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "FOLLOWUP_OUTPUT_REQUIRED",
            "supply a new report path",
        )
        result = summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
