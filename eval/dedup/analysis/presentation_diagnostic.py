# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Versioned evidence-presentation experiment over immutable critic/main contracts."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.critic_diagnostic import (
    ANALYSIS,
    RESOURCES,
    _jsonl,
    _write_jsonl,
    bind_fixed_main,
    primary_changes,
    run_cell,
    validate_freeze,
)
from eval.dedup.analysis.development_diagnostic import _index, evaluate_primary_guards, load_run
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, evaluate_predictions
from eval.dedup.analysis.local_iteration import guard_report, local_gates
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

VARIANTS = ("control", "ordered")
TARGETS = {"H0347", "H0453", "H0748"}
TOKENIZER = Path(
    "/raid/hfang/dedup_eval_cache/huggingface/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)


def runner_config(variant: str) -> Path:
    require(variant in VARIANTS, "PRESENTATION_VARIANT", "unknown presentation arm")
    return RESOURCES / (
        "hs_v06216_control_qwen_c64.yaml" if variant == "control" else "hs_v06217_ordered_qwen_c64.yaml"
    )


def validate_presentation_config() -> None:
    control = yaml.safe_load(runner_config("control").read_text())
    ordered = yaml.safe_load(runner_config("ordered").read_text())
    ordered["execution"]["stages"][0]["name"] = control["execution"]["stages"][0]["name"]
    ordered["execution"]["stages"][0]["judges"][0]["prompt_path"] = control["execution"]["stages"][0]["judges"][0][
        "prompt_path"
    ]
    require(ordered == control, "PRESENTATION_POLICY_CHANGED", "only pair presentation and stage name may differ")


def presentation_record(row: dict) -> dict:
    """Index the existing span coordinates; never infer content or alter the blind payload."""
    payload = row["payload"]
    packet = payload["semantic_diff_evidence"]
    complete = (
        packet["status"] == "COMPLETE"
        and payload["long_document_evidence"]["truncated"] is False
        and all(isinstance(payload[f"document_{side}"]["text"], str) for side in ("a", "b"))
    )
    lines = []
    if complete:
        for span in packet["spans"]:
            lines.append(f"[{span['span_id']} {span['kind']}]")
            sides = ("A", "B") if span["kind"] == "SHARED" else (span["side"],)
            for side in sides:
                prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
                start, end, text = (span[f"{prefix}{key}"] for key in ("start_char", "end_char", "text"))
                original = payload[f"document_{side.lower()}"]["text"]
                require(
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(original)
                    and original[start:end] == text,
                    "PRESENTATION_SPAN_MISMATCH",
                    "span index must align to the same original text",
                    span_id=span["span_id"],
                    side=side,
                )
                preview = text if len(text) <= 64 else text[:32] + " (...) " + text[-32:]
                lines.append(f"{side}[{start}:{end}]: {json.dumps(preview, ensure_ascii=False)}")
    return {**row, "presentation_full_available": complete, "presentation_index": "\n".join(lines)}


def message_renderer(variant: str) -> Callable[[dict], list[dict]]:
    """Use NDD's actual response recipe, including its generated schema instructions."""
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    from eval.llm_judge.run_llm_judge import build_config_builder

    path = runner_config(variant)
    config = yaml.safe_load(path.read_text())
    builder, _ = build_config_builder(
        path,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=config["execution"]["stages"][0]["judges"],
    )
    column = builder.get_column_configs()[0]
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))

    def render(row: dict) -> list[dict]:
        record = presentation_record(row)
        return [
            {"role": role, "content": renderer.render(prompt_template=template, record=record, prompt_type=kind)}
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render


def token_preflight(
    inputs: list[dict],
    tokenizer: Any,
    *,
    context_budget: int = 32768,
    output_budget: int = 4096,
    safety_margin: int = 2048,
) -> dict:
    """No clipping or pair exclusion may hide the cost of repeating visible evidence."""
    rows = []
    for variant in VARIANTS:
        render = message_renderer(variant)
        for row in inputs:
            messages = render(row)
            count = len(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            rows.append(
                {
                    "variant": variant,
                    "canonical_pair_id": row["canonical_pair_id"],
                    "message_sha256": sha256_json(messages),
                    "input_tokens": count,
                }
            )
    maximum = max(r["input_tokens"] for r in rows)
    require(
        maximum + output_budget + safety_margin <= context_budget,
        "PRESENTATION_CONTEXT_BUDGET",
        "presentation exceeds conservative local budget; do not silently truncate or drop pairs",
        maximum=maximum,
    )
    return {
        "context_budget": context_budget,
        "output_budget": output_budget,
        "safety_margin": safety_margin,
        "budget_source": "client safety budget from local runner declaration, not a claim about the Hub server limit",
        "max_input_tokens": {v: max(r["input_tokens"] for r in rows if r["variant"] == v) for v in VARIANTS},
        "rows": rows,
    }


def prepare(root: Path, parent: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.rejudge_comparison import _resource_hashes

    require(not root.exists(), "PRESENTATION_ROOT_EXISTS", "use a new diagnostic root")
    old = validate_freeze(parent)
    require(
        old["schema_version"] == "dedup-critic-paired-diagnostic-v1"
        and len(_jsonl(parent / "input_repeat_1.jsonl")) == 258,
        "PRESENTATION_PARENT",
        "requires the frozen .16 258-pair diagnostic inputs",
    )
    validate_presentation_config()
    source_manifest = json.loads((Path(old["source_root"]) / "run_manifest.json").read_text())
    tokenizer_files = {
        str(TOKENIZER / name): digest for name, digest in source_manifest["tokenizer"]["asset_checksums"].items()
    }
    require(
        all(sha256_file(p) == digest for p, digest in tokenizer_files.items()),
        "PRESENTATION_TOKENIZER_CHANGED",
        "local tokenizer differs from the source contract",
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    inputs = _jsonl(parent / "input_repeat_1.jsonl")
    token_audit = token_preflight(inputs, tokenizer)
    fixed_main = json.loads((parent / "fixed_main.json").read_text())
    paths = dict(old["paths"])
    paths["protocol"] = str(ANALYSIS / "v06217_design.md")
    paths["reference_review"] = str(ANALYSIS / "v06217_reference_review.md")
    frozen = dict(old["frozen_files"]) | tokenizer_files
    dependencies = [
        Path(__file__),
        parent / "manifest.json",
        parent / "input_freeze.json",
        Path(paths["protocol"]),
        Path(paths["reference_review"]),
        ANALYSIS.parent.parent / "llm_judge/run_llm_judge.py",
    ]
    frozen.update({str(p.resolve()): sha256_file(p) for p in dependencies})
    for variant in VARIANTS:
        config = runner_config(variant)
        frozen.update({str(config.parent / name): digest for name, digest in _resource_hashes(config).items()})
    settings = {**old["settings"], "ray_temp_dir": "/raid/hfang/ihb/r17diag"}
    manifest = {
        "schema_version": "dedup-presentation-diagnostic-v1",
        "diagnostic_only": True,
        "eligible_for_release": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_diagnostic": str(parent),
        "parent_contract_digest": old["diagnostic_contract_digest"],
        "source_root": old["source_root"],
        "baseline_root": old["baseline_root"],
        "settings": settings,
        "paths": paths,
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
    }
    manifest["diagnostic_contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "fixed_main.json", fixed_main)
    write_json_atomic(root / "token_preflight.json", token_audit)
    for repeat in (1, 2):
        _write_jsonl(
            root / f"input_repeat_{repeat}.jsonl",
            [presentation_record(row) for row in _jsonl(parent / f"input_repeat_{repeat}.jsonl")],
        )
    write_json_atomic(
        root / "input_freeze.json",
        {
            str(p): sha256_file(p)
            for p in [
                root / "fixed_main.json",
                root / "token_preflight.json",
                root / "input_repeat_1.jsonl",
                root / "input_repeat_2.jsonl",
            ]
        },
    )
    return manifest


def run(root: Path) -> None:
    from eval.dedup.judging.request_relay import RequestRelay
    from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

    manifest = validate_freeze(root)
    require(
        not (root / "started.json").exists(),
        "PRESENTATION_ALREADY_STARTED",
        "do not repeat partially observed experiments",
    )
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(bool(key), "PRESENTATION_CREDENTIAL_MISSING", "inference credential unavailable")
    inputs = {r: _jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    write_json_atomic(
        root / "started.json",
        {"at_utc": datetime.now(UTC).isoformat(), "contract": manifest["diagnostic_contract_digest"]},
    )
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

        def runtime(variant: str) -> ExternalJudgeRuntime:
            return ExternalJudgeRuntime(
                runner_config(variant),
                endpoint=relay.endpoint,
                provider_api_key="unused",  # pragma: allowlist secret
                served_model_overrides={"judge": settings["logical_model"]},
                ray_temp_dir=settings["ray_temp_dir"],
                num_cpus=None,
            )

        with runtime("control"):
            for spec in [{"repeat": 0, "variant": v} for v in VARIANTS] + manifest["schedule"]:
                validate_freeze(root)
                repeat, variant = spec["repeat"], spec["variant"]
                cell = f"repeat_{repeat}_{variant}" if repeat else f"preflight_{variant}"
                batch = inputs[repeat] if repeat else inputs[1][: manifest["preflight_pairs_per_arm"]]
                with runtime(variant) as borrowed:
                    complete = run_cell(root / cell, batch, mains, borrowed, relay, settings)
                print(
                    json.dumps({"cell": cell, **{k: v for k, v in complete.items() if k != "artifacts"}}), flush=True
                )
                require(
                    complete["valid"] == complete["requested"] and complete["errors"] == 0,
                    "PRESENTATION_CELL_FAILED",
                    "incomplete cell stops the diagnostic",
                )
    validate_freeze(root)
    write_json_atomic(
        root / "run_complete.json",
        {
            "diagnostic_only": True,
            "eligible_for_release": False,
            "formal_requests": 1032,
            "preflight_requests": 16,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def replay_cell(root: Path, name: str) -> tuple[list[dict], dict]:
    complete = json.loads((root / name / "complete.json").read_text())
    require(
        all(sha256_file(p) == digest for p, digest in complete["artifacts"].items()),
        "PRESENTATION_RESULTS_CHANGED",
        "cell artifacts changed",
    )
    rows = _jsonl(root / name / "predictions.jsonl")
    inputs = _index(_jsonl(root / "input_repeat_1.jsonl"), "inputs")
    mains = json.loads((root / "fixed_main.json").read_text())
    raw = {}
    for path in (root / name).glob("attempt_*/output/*.jsonl"):
        for r in _jsonl(path):
            critic = r.get("qwen_dedup_record_binding_critic")
            if isinstance(critic, dict):
                raw[(r["canonical_pair_id"], sha256_json(critic))] = r
    require(
        _index(rows, name).keys() == inputs.keys() and complete["valid"] == complete["requested"] == len(inputs),
        "PRESENTATION_MEMBERSHIP",
        "all frozen pairs must be accounted for",
    )
    for row in rows:
        pid = row["canonical_pair_id"]
        key = pid, row["critic_response_sha256"]
        require(key in raw, "PRESENTATION_RAW_MISSING", "published critic digest has no raw match")
        replay = bind_fixed_main([inputs[pid]], [raw[key]], mains)[0]
        require(
            all(row[k] == value for k, value in replay.items()),
            "PRESENTATION_REPLAY_CHANGED",
            "public/evidence replay changed",
        )
    return rows, {k: v for k, v in complete.items() if k != "artifacts"}


def diagnostic_checks(candidate: dict, control: dict, target_report: dict) -> dict:
    return {
        "local_gates": candidate["local_gates"]["passed"],
        "all_three_targets_correct": target_report["passed"],
        "weighted_primary_noninferior": candidate["metrics"]["weighted"]["primary_decision_exact"]
        >= control["metrics"]["weighted"]["primary_decision_exact"],
        "over_group_noninferior": candidate["metrics"]["over_group"] <= control["metrics"]["over_group"],
    }


def summarize(root: Path, output: Path) -> dict:
    manifest = validate_freeze(root)
    require(
        (root / "run_complete.json").is_file(),
        "PRESENTATION_INCOMPLETE",
        "complete the frozen schedule before scoring",
    )
    labels = _read_csv(Path(manifest["paths"]["labels"]))
    _, _, source, _ = load_run(Path(manifest["source_root"]))
    _, _, baseline, _ = load_run(Path(manifest["baseline_root"]))
    ids = _index(labels, "labels").keys()
    baseline = [r for r in baseline if r["canonical_pair_id"] in ids]
    negatives = json.loads(Path(manifest["paths"]["negative_guards"]).read_text())["critic_repaired_negative_guards"]
    additional = json.loads(Path(manifest["paths"]["additional_guards"]).read_text())
    target_guards = [
        {**label, "final": {k: label[f"human_{k}"] for k in PRIMARY_FIELDS}}
        for label in labels
        if label["review_id"] in TARGETS
    ]
    require(
        len(negatives) == 38 and len(additional) == 22 and len(target_guards) == 3,
        "PRESENTATION_GUARDS",
        "all frozen guards are required",
    )
    baseline_metrics = evaluate_predictions(labels, baseline)
    cells, predictions = {}, {}
    for spec in manifest["schedule"]:
        name = f"repeat_{spec['repeat']}_{spec['variant']}"
        rows, operations = replay_cell(root, name)
        predictions[name] = rows
        metrics = evaluate_predictions(labels, rows)
        guards = guard_report(labels, rows, baseline, negatives)
        guards["additional_critic_repair_guards"] = evaluate_primary_guards(additional, rows)
        cells[name] = {
            "metrics": metrics,
            "operations": operations,
            "guards": guards,
            "targets": evaluate_primary_guards(target_guards, rows),
            "local_gates": local_gates(metrics, baseline_metrics, guards, operations, 258),
            "vs_historical": primary_changes(labels, source, rows),
        }
    checks, paired = {}, {}
    for repeat in (1, 2):
        name, control = f"repeat_{repeat}_ordered", f"repeat_{repeat}_control"
        checks[name] = diagnostic_checks(cells[name], cells[control], cells[name]["targets"])
        paired[name] = primary_changes(labels, predictions[control], predictions[name])
    repeats = {
        v: primary_changes(labels, predictions[f"repeat_1_{v}"], predictions[f"repeat_2_{v}"]) for v in VARIANTS
    }
    stable = len(repeats["ordered"]) <= len(repeats["control"])
    result = {
        "schema_version": "dedup-presentation-assessment-v1",
        "diagnostic_only": True,
        "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
        "cells": cells,
        "paired_changes": paired,
        "within_arm_repeat_changes": repeats,
        "checks": checks,
        "repeat_primary_changes_not_above_control": stable,
        "supported_for_separate_fresh_local_validation": stable and all(all(c.values()) for c in checks.values()),
        "eligible_for_release": False,
        "reference_changed": False,
    }
    write_json_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.16-critic-diagnostic"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        print(prepare(root, args.parent.resolve())["diagnostic_contract_digest"])
    elif args.action == "run":
        run(root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "PRESENTATION_OUTPUT_EXISTS",
            "provide a new assessment path",
        )
        result = summarize(root, args.output)
        print(
            json.dumps(
                {
                    "supported_for_separate_fresh_local_validation": result[
                        "supported_for_separate_fresh_local_validation"
                    ],
                    "checks": result["checks"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
