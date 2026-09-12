# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen V0.6.2.20 coverage-critic evaluation with digest-matched raw-response replay."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.analysis.critic_diagnostic import (
    ANALYSIS,
    RESOURCES,
    _jsonl,
    _write_jsonl,
    bind_fixed_main,
    blind_rows,
    primary_changes,
    validate_freeze,
)
from eval.dedup.analysis.critic_diagnostic import (
    run_cell as run_control_cell,
)
from eval.dedup.analysis.development_diagnostic import _index, evaluate_primary_guards, load_run
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, evaluate_predictions
from eval.dedup.analysis.local_context_diagnostic import TOKENIZER
from eval.dedup.analysis.local_iteration import guard_report, local_gates, validate_projection
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.analysis.presentation_diagnostic import TARGETS, diagnostic_checks, message_renderer
from eval.dedup.judging.coverage_runtime import CoverageWitnessRuntime, coverage_renderer, validate_native_coverage_row
from eval.dedup.judging.coverage_witness import COLUMN, adapt_coverage_witness, coverage_schema
from eval.dedup.judging.local_ndd import RECORD_BINDING_CRITIC_COLUMN, _read_output_rows, _safe_retry_feedback
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

VARIANTS = ("control", "coverage")
CONFIG = RESOURCES / "hs_v06220_coverage_qwen_c64.yaml"
CONTROL_CONFIG = RESOURCES / "hs_v06216_control_qwen_c64.yaml"


def bind_coverage(inputs: list[dict], outputs: list[dict], mains: dict) -> list[dict]:
    expected, actual = _index(inputs, "coverage inputs"), _index(outputs, "coverage outputs")
    require(expected.keys() == actual.keys(), "COVERAGE_MEMBERSHIP", "every input requires exactly one output")
    predictions = []
    for pid, row in actual.items():
        packet = expected[pid]
        require(
            row.get("payload") == packet["payload"]
            and row.get("judge_payload_hash") == packet["judge_payload_hash"] == sha256_json(packet["payload"]),
            "COVERAGE_PAYLOAD_CHANGED",
            "output belongs to another payload",
        )
        require(
            not ({"qwen_dedup_semantic_judge", RECORD_BINDING_CRITIC_COLUMN} & row.keys()),
            "COVERAGE_COLUMN_LEAK",
            "coverage must not call or observe main or legacy critic",
        )
        value = validate_native_coverage_row(row)
        public = adapt_coverage_witness(mains[pid], value, packet["payload"])
        validate_evidence_offsets(public, packet["payload"])
        trace = row[COLUMN + "__trace"]
        predictions.append(
            {
                "canonical_pair_id": pid,
                "diagnostic_only": True,
                "fixed_main_sha256": sha256_json(mains[pid]),
                "coverage_response_sha256": sha256_json(value),
                "coverage_trace_sha256": sha256_json(trace),
                "native_corrections": max(0, sum(m.get("role") == "assistant" for m in trace) - 1),
                **public,
            }
        )
    return predictions


def run_coverage_cell(root: Path, inputs: list[dict], mains: dict, runtime: Any, relay: Any, settings: dict) -> dict:
    from eval.dedup.judging.request_relay import RelayContext

    require(not root.exists(), "COVERAGE_CELL_EXISTS", "never reuse an observed cell")
    pending, accepted, retried, native_events = inputs, [], set(), []
    for attempt in range(1, settings["max_retries"] + 2):
        folder = root / f"attempt_{attempt:02d}"
        _write_jsonl(folder / "input.jsonl", pending)
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
        outputs = _index(_read_output_rows(folder / "output"), "coverage output")
        require(outputs.keys() <= _index(pending, "pending").keys(), "COVERAGE_MEMBERSHIP", "unexpected output pair")
        errors, next_pending = [], []
        for packet in pending:
            pid = packet["canonical_pair_id"]
            row = outputs.get(pid)
            if row is not None:
                trace = row.get(COLUMN + "__trace", [])
                messages = sum(m.get("role") == "assistant" for m in trace) if isinstance(trace, list) else 0
                native_events.append(
                    {
                        "canonical_pair_id": pid,
                        "outer_attempt": attempt,
                        "assistant_messages": messages,
                        "native_corrections": max(0, messages - 1),
                    }
                )
            try:
                result = bind_coverage([packet], [row] if row is not None else [], mains)[0]
                accepted.append({**result, "attempts": attempt, "retried": attempt > 1})
            except Exception as exc:  # noqa: BLE001 - all schema/evidence failures share a bounded retry budget
                feedback = _safe_retry_feedback(exc)
                errors.append({"canonical_pair_id": pid, "issue": feedback})
                next_pending.append({**packet, "repair_feedback": feedback})
                if attempt <= settings["max_retries"]:
                    retried.add(pid)
        write_json_atomic(folder / "validation_errors.json", errors)
        pending = next_pending
        if not pending:
            break
    _write_jsonl(root / "predictions.jsonl", accepted)
    write_json_atomic(root / "native_corrections.json", native_events)
    write_json_atomic(root / "terminal_pair_ids.json", [r["canonical_pair_id"] for r in pending])
    events = [e for path in root.glob("attempt_*/events.jsonl") for e in _jsonl(path)]
    complete = {
        "requested": len(inputs),
        "valid": len(accepted),
        "errors": len(pending),
        "retried": len(retried),
        "native_corrections": sum(e["native_corrections"] for e in native_events),
        "native_corrected_pairs": len({e["canonical_pair_id"] for e in native_events if e["native_corrections"]}),
        "judge_retried_pairs": len(
            retried | {e["canonical_pair_id"] for e in native_events if e["native_corrections"]}
        ),
        "http_statuses": dict(Counter(str(e["http_status"]) for e in events)),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {str(p): sha256_file(p) for p in sorted(root.rglob("*.json*")) if p.is_file()},
    }
    write_json_atomic(root / "complete.json", complete)
    return complete


def token_preflight(inputs: list[dict], tokenizer: Any) -> dict:
    rows = []
    for variant, render in (("control", message_renderer("control")), ("coverage", coverage_renderer(CONFIG))):
        for row in inputs:
            messages = render(row)
            count = len(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(
                count + 4096 + 2048 <= 32768,
                "COVERAGE_TOKEN_BUDGET",
                "do not truncate evidence or remove oversized pairs",
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
        "budget_source": "client safety budget, not Hub context limit",
        "max_input_tokens": {v: max(r["input_tokens"] for r in rows if r["variant"] == v) for v in VARIANTS},
        "rows": rows,
    }


def prepare(root: Path, parent: Path, boundary: Path) -> dict:
    from transformers import AutoTokenizer

    require(not root.exists(), "COVERAGE_ROOT_EXISTS", "use a new run root with no reused cache")
    old = validate_freeze(parent)
    require(
        old["schema_version"] == "dedup-local-context-diagnostic-v1",
        "COVERAGE_PARENT",
        "expected complete .18 frozen inputs",
    )
    labels = _read_csv(Path(old["paths"]["labels"]))
    validate_projection(labels, _read_csv(Path(old["paths"]["reference"])))
    boundary_result = json.loads(boundary.read_text())
    required_boundary_files = {
        str((ANALYSIS.parent / "judging/coverage_runtime.py").resolve()),
        str((ANALYSIS.parents[2] / "tests/eval/dedup/test_coverage_runtime.py").resolve()),
    }
    require(
        boundary_result["passed"]
        and boundary_result["external_model_calls"] == 0
        and required_boundary_files <= boundary_result["artifacts"].keys()
        and all(sha256_file(p) == h for p, h in boundary_result["artifacts"].items()),
        "COVERAGE_BOUNDARY_UNVERIFIED",
        "native boundary check must cover the current runtime",
    )
    inputs = {r: blind_rows(_jsonl(parent / f"input_repeat_{r}.jsonl"), repeat=r) for r in (1, 2)}
    require(
        len(inputs[1]) == 258 and _index(inputs[1], "inputs").keys() == _index(labels, "labels").keys(),
        "COVERAGE_MEMBERSHIP",
        "all 258 frozen pairs and original weights are required",
    )
    source_manifest = json.loads((Path(old["source_root"]) / "run_manifest.json").read_text())
    tokenizer_files = {
        str(TOKENIZER / name): digest for name, digest in source_manifest["tokenizer"]["asset_checksums"].items()
    }
    require(
        all(sha256_file(p) == d for p, d in tokenizer_files.items()),
        "COVERAGE_TOKENIZER_CHANGED",
        "tokenizer differs from the frozen source",
    )
    token_audit = token_preflight(inputs[1], AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True))
    frozen = dict(old["frozen_files"]) | tokenizer_files
    files = [
        Path(__file__),
        parent / "manifest.json",
        parent / "input_freeze.json",
        boundary,
        ANALYSIS / "v06220_design.md",
        ANALYSIS / "v06220_run_protocol.md",
        ANALYSIS.parent / "judging/coverage_witness.py",
        ANALYSIS.parent / "judging/coverage_runtime.py",
        CONFIG,
        RESOURCES / "hs_v06220_coverage_pair.jinja",
        RESOURCES / "hs_v06220_coverage_system.jinja",
        ANALYSIS.parents[2] / "tests/eval/dedup/test_coverage_diagnostic.py",
        ANALYSIS.parents[2] / "tests/eval/dedup/test_coverage_runtime.py",
    ]
    frozen.update({str(p.resolve()): sha256_file(p) for p in files})
    manifest = {
        "schema_version": "dedup-coverage-diagnostic-v1",
        "diagnostic_only": True,
        "eligible_for_release": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_diagnostic": str(parent),
        "parent_contract_digest": old["diagnostic_contract_digest"],
        "source_root": old["source_root"],
        "baseline_root": old["baseline_root"],
        "settings": {**old["settings"], "ray_temp_dir": "/raid/hfang/ihb/r20diag"},
        "paths": {**old["paths"], "protocol": str(ANALYSIS / "v06220_run_protocol.md")},
        "source_implementation_sha256": old["source_implementation_sha256"],
        "frozen_files": frozen,
        "fixed_main_sha256": old["fixed_main_sha256"],
        "formal_pairs_per_cell": 258,
        "formal_requests": 1032,
        "preflight_pairs_per_arm": 8,
        "schedule": [
            {"repeat": r, "variant": v} for r, arms in ((1, VARIANTS), (2, tuple(reversed(VARIANTS)))) for v in arms
        ],
        "token_preflight_sha256": sha256_json(token_audit),
        "coverage_schema_sha256": sha256_json(coverage_schema()),
        "variant_contracts": {
            "control": {"runner_config": str(CONTROL_CONFIG), "adapter_policy": "v8"},
            "coverage": {"runner_config": str(CONFIG), "adapter_policy": "dedup-retained-coverage-v1"},
        },
    }
    manifest["diagnostic_contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "fixed_main.json", json.loads((parent / "fixed_main.json").read_text()))
    write_json_atomic(root / "coverage_schema.json", coverage_schema())
    write_json_atomic(root / "token_preflight.json", token_audit)
    for r, rows in inputs.items():
        _write_jsonl(root / f"input_repeat_{r}.jsonl", rows)
    write_json_atomic(
        root / "input_freeze.json",
        {
            str(p): sha256_file(p)
            for p in (
                root / "fixed_main.json",
                root / "coverage_schema.json",
                root / "token_preflight.json",
                root / "input_repeat_1.jsonl",
                root / "input_repeat_2.jsonl",
            )
        },
    )
    return manifest


def technical_passed(complete: dict) -> bool:
    return (
        complete["requested"] == complete["valid"]
        and complete["errors"] == 0
        and complete["retried"] == 0
        and complete.get("native_corrections", 0) == 0
        and complete["http_statuses"] == {"200": complete["requested"]}
    )


def run(root: Path) -> None:
    from eval.dedup.judging.request_relay import RequestRelay
    from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

    manifest = validate_freeze(root)
    require(
        not (root / "started.json").exists(),
        "COVERAGE_ALREADY_STARTED",
        "never silently restart an observed experiment",
    )
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(bool(key), "COVERAGE_CREDENTIAL_MISSING", "inference credential is unavailable")
    inputs = {r: _jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    write_json_atomic(
        root / "started.json",
        {"at_utc": datetime.now(UTC).isoformat(), "contract": manifest["diagnostic_contract_digest"]},
    )
    try:
        with RequestRelay(
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
        ) as relay:

            def runtime(variant: str) -> Any:
                if variant == "coverage":
                    return CoverageWitnessRuntime(
                        CONFIG,
                        endpoint=relay.endpoint,
                        model_name=settings["logical_model"],
                        ray_temp_dir=settings["ray_temp_dir"],
                    )
                return ExternalJudgeRuntime(
                    CONTROL_CONFIG,
                    endpoint=relay.endpoint,
                    provider_api_key="unused",
                    served_model_overrides={"judge": settings["logical_model"]},
                    ray_temp_dir=settings["ray_temp_dir"],
                    num_cpus=None,
                )

            with runtime("control"):
                preflight = {}
                for spec in [{"repeat": 0, "variant": v} for v in VARIANTS] + manifest["schedule"]:
                    validate_freeze(root)
                    r, variant = spec["repeat"], spec["variant"]
                    if r:
                        require(
                            len(preflight) == 2 and all(technical_passed(c) for c in preflight.values()),
                            "COVERAGE_PREFLIGHT_FAILED",
                            "technical instability stops formal submission; do not reuse preflight outputs",
                        )
                    name = f"repeat_{r}_{variant}" if r else f"preflight_{variant}"
                    batch = inputs[r] if r else inputs[1][:8]
                    with runtime(variant) as borrowed:
                        runner = run_coverage_cell if variant == "coverage" else run_control_cell
                        complete = runner(root / name, batch, mains, borrowed, relay, settings)
                    print(
                        json.dumps({"cell": name, **{k: v for k, v in complete.items() if k != "artifacts"}}),
                        flush=True,
                    )
                    if not r:
                        preflight[variant] = complete
                    else:
                        require(
                            complete["valid"] == len(batch) and complete["errors"] == 0,
                            "COVERAGE_CELL_FAILED",
                            "incomplete cell stops the diagnostic without removing failures",
                        )
    except Exception as exc:
        write_json_atomic(
            root / "stopped.json",
            {
                "at_utc": datetime.now(UTC).isoformat(),
                "issue": _safe_retry_feedback(exc),
                "eligible_for_release": False,
            },
        )
        raise
    validate_freeze(root)
    write_json_atomic(
        root / "run_complete.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "formal_requests": 1032,
            "preflight_requests": 16,
            "eligible_for_release": False,
        },
    )


def replay_cell(root: Path, name: str, inputs: list[dict], mains: dict) -> tuple[list[dict], dict]:
    folder = root / name
    complete = json.loads((folder / "complete.json").read_text())
    require(
        all(sha256_file(p) == d for p, d in complete["artifacts"].items()),
        "COVERAGE_RESULTS_CHANGED",
        "cell artifacts changed",
    )
    rows = _jsonl(folder / "predictions.jsonl")
    packets, published = _index(inputs, "inputs"), _index(rows, "published")
    require(
        published.keys() <= packets.keys()
        and len(rows) == complete["valid"]
        and complete["requested"] == len(inputs) == complete["valid"] + complete["errors"],
        "COVERAGE_REPLAY_MEMBERSHIP",
        "all valid and terminal pairs must be accounted for",
    )
    candidate = name.endswith("coverage")
    column = COLUMN if candidate else RECORD_BINDING_CRITIC_COLUMN
    raw = {}
    for path in folder.glob("attempt_*/output/*.jsonl"):
        for row in _jsonl(path):
            value = row.get(column)
            value = json.loads(value) if isinstance(value, str) else value
            if isinstance(value, dict):
                key = (
                    row["canonical_pair_id"],
                    sha256_json(value),
                    sha256_json(row.get(COLUMN + "__trace")) if candidate else "",
                )
                raw[key] = row
    for row in rows:
        pid = row["canonical_pair_id"]
        key = (
            (pid, row["coverage_response_sha256"], row["coverage_trace_sha256"])
            if candidate
            else (pid, row["critic_response_sha256"], "")
        )
        require(key in raw, "COVERAGE_RAW_MISSING", "published response/trace digest has no raw match")
        replay = (bind_coverage if candidate else bind_fixed_main)([packets[pid]], [raw[key]], mains)[0]
        require(
            all(row[k] == v for k, v in replay.items()),
            "COVERAGE_REPLAY_CHANGED",
            "public output or exact evidence changed",
        )
    return rows, {k: v for k, v in complete.items() if k != "artifacts"}


def summarize(root: Path, output: Path) -> dict:
    manifest = validate_freeze(root)
    inputs = {r: _jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    preflights = {}
    for variant in VARIANTS:
        name = f"preflight_{variant}"
        if (root / name / "complete.json").is_file():
            rows, operations = replay_cell(root, name, inputs[1][:8], mains)
            preflights[variant] = {
                "operations": operations,
                "technical_passed": technical_passed(operations),
                "primary_predictions": [{k: row[k] for k in ("canonical_pair_id", *PRIMARY_FIELDS)} for row in rows],
            }
    result = {
        "schema_version": "dedup-coverage-diagnostic-assessment-v1",
        "diagnostic_only": True,
        "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
        "reference_changed": False,
        "eligible_for_release": False,
        "preflights": preflights,
        "supported_for_separate_fresh_local_validation": False,
    }
    if not (root / "run_complete.json").is_file():
        require((root / "stopped.json").is_file(), "COVERAGE_STILL_RUNNING", "do not score an active or unknown run")
        result.update(
            status="STOPPED_BEFORE_COMPLETE_FORMAL_SCHEDULE", stop=json.loads((root / "stopped.json").read_text())
        )
        result["completed_formal_cells"] = {}
        for spec in manifest["schedule"]:
            name = f"repeat_{spec['repeat']}_{spec['variant']}"
            if (root / name / "complete.json").is_file():
                _, ops = replay_cell(root, name, inputs[spec["repeat"]], mains)
                result["completed_formal_cells"][name] = ops
        write_json_atomic(output, result)
        return result
    labels = _read_csv(Path(manifest["paths"]["labels"]))
    _, _, baseline, _ = load_run(Path(manifest["baseline_root"]))
    ids = _index(labels, "labels").keys()
    baseline = [r for r in baseline if r["canonical_pair_id"] in ids]
    baseline_metrics = evaluate_predictions(labels, baseline)
    negatives = json.loads(Path(manifest["paths"]["negative_guards"]).read_text())["critic_repaired_negative_guards"]
    additional = json.loads(Path(manifest["paths"]["additional_guards"]).read_text())
    target_guards = [
        {**label, "final": {k: label[f"human_{k}"] for k in PRIMARY_FIELDS}}
        for label in labels
        if label["review_id"] in TARGETS
    ]
    require(
        len(negatives) == 38 and len(additional) == 22 and len(target_guards) == 3,
        "COVERAGE_GUARDS",
        "all frozen guards are required",
    )
    cells, predictions = {}, {}
    for spec in manifest["schedule"]:
        name = f"repeat_{spec['repeat']}_{spec['variant']}"
        rows, ops = replay_cell(root, name, inputs[spec["repeat"]], mains)
        require(
            ops["valid"] == ops["requested"] == 258 and ops["errors"] == 0,
            "COVERAGE_INCOMPLETE_FORMAL",
            "full 258-pair cell required",
        )
        predictions[name] = rows
        metrics = evaluate_predictions(labels, rows)
        guards = guard_report(labels, rows, baseline, negatives)
        guards["additional_critic_repair_guards"] = evaluate_primary_guards(additional, rows)
        cells[name] = {
            "metrics": metrics,
            "operations": ops,
            "guards": guards,
            "targets": evaluate_primary_guards(target_guards, rows),
            "local_gates": local_gates(metrics, baseline_metrics, guards, ops, 258),
        }
        if spec["variant"] == "coverage":
            gates = cells[name]["local_gates"]
            gates["checks"]["all_judge_retried_pairs_at_most_one_percent"] = ops["judge_retried_pairs"] / 258 <= 0.01
            gates["passed"] = all(gates["checks"].values())
    checks = {
        f"repeat_{r}_coverage": diagnostic_checks(
            cells[f"repeat_{r}_coverage"], cells[f"repeat_{r}_control"], cells[f"repeat_{r}_coverage"]["targets"]
        )
        for r in (1, 2)
    }
    repeats = {
        v: primary_changes(labels, predictions[f"repeat_1_{v}"], predictions[f"repeat_2_{v}"]) for v in VARIANTS
    }
    stable = len(repeats["coverage"]) <= len(repeats["control"])
    result.update(
        status="FORMAL_SCHEDULE_COMPLETE",
        cells=cells,
        checks=checks,
        paired_changes={
            str(r): primary_changes(labels, predictions[f"repeat_{r}_control"], predictions[f"repeat_{r}_coverage"])
            for r in (1, 2)
        },
        within_arm_repeat_changes=repeats,
        repeat_primary_changes_not_above_control=stable,
        supported_for_separate_fresh_local_validation=stable and all(all(c.values()) for c in checks.values()),
    )
    write_json_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.18-local-context-diagnostic"))
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.boundary is not None, "COVERAGE_BOUNDARY_REQUIRED", "supply the verified native boundary report")
        print(json.dumps({"contract": prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}))
    elif args.action == "run":
        run(args.root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "COVERAGE_OUTPUT_REQUIRED",
            "supply a new assessment path",
        )
        result = summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
