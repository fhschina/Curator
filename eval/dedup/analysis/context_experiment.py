# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh V5 component diagnostics with immutable inputs and preserved historical runs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from eval.dedup.analysis import composite_format_experiment as previous
from eval.dedup.analysis.bottleneck_audit import transition_matrix
from eval.dedup.analysis.coverage_experiment import CoverageExperiment, common, native
from eval.dedup.judging.context_coverage import (
    CONTRACT,
    ROUTING,
    adapt_context,
    bind_context_response,
    context_schema,
    critic_route,
    finalize_context,
    owned_output,
)
from eval.dedup.judging.context_runtime import (
    CONFIG,
    VERSION,
    ContextRuntime,
    context_record,
    context_renderer,
    encode_main_review,
)
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.payload_transport import bind_transported_payload
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

PARENT = Path("/raid/hfang/ihb/runs/v0.6.2.30-format-diagnostic")


class ContextArm(CoverageExperiment):
    def __init__(self, stage: str, path: Path = CONFIG, *, tokenizer: Any = None):
        require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
        self.stage, self.config, self.spec, self.tokenizer = stage, {"coverage": path}, {"version": VERSION}, tokenizer

    @cached_property
    def renderers(self) -> dict:
        return {"coverage": context_renderer(self.config["coverage"], stage=self.stage)}

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        require(variant == "coverage", "CONTEXT_ARM", "single stage arm required")
        require(self.stage != "main" or not mains, "COMPOSITE_MAIN_REUSE", "fresh main only")
        common._index(inputs, "context inputs")
        called, owned = [], []
        for packet in inputs:
            pid = packet["canonical_pair_id"]
            require(
                set(packet)
                <= {"canonical_pair_id", "payload", "judge_payload_hash", "repair_feedback", "proposed_main_text"}
                and packet["judge_payload_hash"] == sha256_json(packet["payload"])
                and isinstance(packet.get("repair_feedback"), str),
                "FOLLOWUP_BOUNDARY_CONTEXT_INPUT",
                "blind original input and stage-local feedback only",
            )
            result = owned_output(packet["payload"])
            route = "OWNED_INPUT" if result is not None else "NEEDS_FRESH_MAIN"
            if self.stage == "critic" and result is None:
                require(pid in mains, "COMPOSITE_MAIN_MISSING", "fresh main certificate required")
                route, result = critic_route(mains[pid], packet["payload"])
            if result is None:
                if self.stage == "critic":
                    require(
                        packet.get("proposed_main_text") == encode_main_review(mains[pid]),
                        "FOLLOWUP_BOUNDARY_CONTEXT_MAIN",
                        "exact accepted main claim required",
                    )
                context_record(packet, self.stage)
                if self.tokenizer is not None:
                    count = len(
                        self.tokenizer.apply_chat_template(
                            self.renderers[variant](packet),
                            tokenize=True,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                    require(
                        count + 4096 + 2048 <= 32768,
                        "COMPOSITE_TOKEN_BUDGET",
                        "no truncation or dropped cases",
                        pair_id=pid,
                        input_tokens=count,
                    )
                called.append(packet)
            else:
                require(
                    "proposed_main_text" not in packet,
                    "FOLLOWUP_BOUNDARY_CONTEXT_MAIN",
                    "unrequested branch cannot carry a claim",
                )
                owned.append(
                    {
                        "canonical_pair_id": pid,
                        "diagnostic_only": True,
                        "diagnostic_version": VERSION,
                        "stage": self.stage,
                        "routing_contract": ROUTING,
                        "route": route,
                        "judge_payload_hash": packet["judge_payload_hash"],
                        "fresh_main_sha256": sha256_json(mains[pid])
                        if self.stage == "critic" and pid in mains
                        else None,
                        "critic_request_status": "NOT_REQUESTED_OWNED_BRANCH",
                        "attempts": 0,
                        "retried": False,
                        **result,
                    }
                )
        return called, owned

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "CONTEXT_ARM", "single stage arm required")
        return ContextRuntime(
            self.config[variant],
            stage=self.stage,
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

        packets, rows = common._index(inputs, "inputs"), common._index(outputs, "outputs")
        require(packets.keys() == rows.keys(), "EXPERIMENT_RESPONSE_MISSING", "one response per called pair")
        called, owned = self.layout(inputs, mains, variant)
        require(not owned and len(called) == len(inputs), "FOLLOWUP_BOUNDARY_ROUTING", "owned branch cannot be called")
        results = []
        for pid, row in rows.items():
            packet = packets[pid]
            _, payload_audit = bind_transported_payload(packet, row)
            require(
                row.get("repair_feedback") == packet["repair_feedback"]
                and row.get("proposed_main_text") == packet.get("proposed_main_text"),
                "FOLLOWUP_BOUNDARY_CONTEXT_MAIN",
                "actual feedback or main claim changed",
            )
            require(
                not ({"qwen_dedup_semantic_judge", common.RECORD_BINDING_CRITIC_COLUMN, "proposed_main"} & row.keys()),
                "FOLLOWUP_BOUNDARY_CONTEXT_LEAK",
                "no legacy prediction columns",
            )
            trace = row.get(COLUMN + "__trace")
            require(
                isinstance(trace, list) and len(trace) >= 3 and all(isinstance(m, dict) for m in trace),
                "FOLLOWUP_BOUNDARY_TRACE",
                "full native conversation required",
            )
            messages = [{"role": m.get("role"), "content": native.trace_text(m.get("content"))} for m in trace[:2]]
            require(
                messages == self.renderers[variant](packet), "FOLLOWUP_BOUNDARY_PROMPT", "exact new prompt required"
            )
            assistants = [m for m in trace if m.get("role") == "assistant"]
            require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "assistant response required")
            value = StructuredResponseRecipe(context_schema(), pruning=False).parse(
                native.trace_text(assistants[-1].get("content"))
            )
            value, response_audit = bind_context_response(value, row.get(COLUMN))
            public = (
                adapt_context(value, packet["payload"])
                if self.stage == "main"
                else finalize_context(mains[pid], value, packet["payload"])
            )
            results.append(
                {
                    "canonical_pair_id": pid,
                    "diagnostic_only": True,
                    "diagnostic_version": VERSION,
                    "stage": self.stage,
                    "routing_contract": ROUTING,
                    "coverage_contract": CONTRACT,
                    "judge_payload_hash": packet["judge_payload_hash"],
                    "coverage_response": value,
                    "fresh_main_sha256": sha256_json(mains[pid]) if self.stage == "critic" else None,
                    "critic_request_status": "REQUESTED",
                    "payload_transport": payload_audit,
                    "response_transport": response_audit,
                    "native_corrections": len(assistants) - 1,
                    "native_request_messages_sha256": sha256_json(messages),
                    "native_trace_sha256": sha256_json(trace),
                    "raw_output_sha256": sha256_json(row),
                    **public,
                }
            )
        return results


def critic_inputs(inputs: list[dict], main_rows: list[dict]) -> tuple[list[dict], dict]:
    packets, predictions = common._index(inputs, "inputs"), common._index(main_rows, "main results")
    require(packets.keys() == predictions.keys(), "COMPOSITE_MAIN_COMPLETION", "complete fresh main required")
    result, mains = [], {}
    for pid, packet in packets.items():
        row = predictions[pid]
        require(
            row.get("stage") == "main"
            and row.get("diagnostic_version") == VERSION
            and row.get("judge_payload_hash") == packet["judge_payload_hash"],
            "COMPOSITE_MAIN_PROVENANCE",
            "same-version fresh main and original payload required",
        )
        prepared = {k: v for k, v in packet.items() if k != "repair_feedback"} | {"repair_feedback": ""}
        if owned_output(packet["payload"]) is None:
            require("coverage_response" in row, "COMPOSITE_MAIN_PROVENANCE", "fresh V5 certificate required")
            mains[pid] = row["coverage_response"]
            public = adapt_context(mains[pid], packet["payload"])
            require(all(row[k] == v for k, v in public.items()), "COMPOSITE_MAIN_PROVENANCE", "public/raw mismatch")
            if public["same_duplicate_group"] == "YES":
                prepared["proposed_main_text"] = encode_main_review(mains[pid])
        result.append(prepared)
    return result, mains


def run_components(
    root: Path, inputs: list[dict], runtime_factory: Any, relay: Any, settings: dict, *, tokenizer: Any = None
) -> dict:
    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "fresh cell required")
    arms = {s: ContextArm(s, tokenizer=tokenizer) for s in ("main", "critic")}
    with runtime_factory(arms["main"]) as runtime:
        main_ops = arms["main"].run_cell(root / "main", inputs, {}, runtime, relay, settings, "coverage")
    write_json_atomic(root / "main_operations.json", main_ops)
    require(
        main_ops["errors"] == 0 and main_ops["valid"] == len(inputs), "COMPOSITE_MAIN_COMPLETION", "incomplete main"
    )
    main, _ = arms["main"].replay_cell(root, "main", "coverage", inputs, {})
    packets, mains = critic_inputs(inputs, main)
    with runtime_factory(arms["critic"]) as runtime:
        critic_ops = arms["critic"].run_cell(root / "critic", packets, mains, runtime, relay, settings, "coverage")
    arms["critic"].replay_cell(root, "critic", "coverage", packets, mains)
    ops = {
        "main": main_ops,
        "critic": critic_ops,
        "complete": critic_ops["errors"] == 0 and critic_ops["valid"] == len(inputs),
    }
    write_json_atomic(root / "operations.json", ops)
    return ops


def compare(root: Path, inputs: list[dict], labels: list[dict]) -> dict:
    main, _ = ContextArm("main").replay_cell(root, "main", "coverage", inputs, {})
    packets, mains = critic_inputs(inputs, main)
    final, _ = ContextArm("critic").replay_cell(root, "critic", "coverage", packets, mains)
    ids = {r["canonical_pair_id"] for r in labels}
    result = transition_matrix(
        labels, [r for r in main if r["canonical_pair_id"] in ids], [r for r in final if r["canonical_pair_id"] in ids]
    )
    result["interpretation"] = (
        "Same fresh V5 main with and without the V5 critic; no arbitration or historical main reuse."
    )
    return result


def prepare(root: Path, boundary: Path) -> dict:
    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "fresh root required")
    parent = previous.validate(PARENT)
    evidence = json.loads(boundary.read_text())
    require(
        evidence["passed"]
        and evidence["external_model_calls"] == 0
        and evidence["full_boundary_tests_passed"] == 4
        and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
        "CONTEXT_BOUNDARY",
        "current four native main/critic first/retry tests required",
    )
    analysis = Path(__file__).parent
    files = [
        Path(__file__),
        CONFIG,
        CONFIG.parent / "hs_v06231_context_rules.jinja",
        analysis / "v06231_design.md",
        boundary,
        *[analysis.parent / "judging" / f"{n}.py" for n in ("context_coverage", "context_runtime")],
        *[
            analysis.parents[2] / "tests/eval/dedup" / f"test_{n}.py"
            for n in ("context_coverage", "context_runtime", "context_experiment")
        ],
    ]
    require(
        {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} <= evidence["artifacts"].keys(),
        "CONTEXT_BOUNDARY",
        "all new implementation assets must be tested",
    )
    labels = json.loads((PARENT / "labels_private.json").read_text())
    translation_id = next(r["canonical_pair_id"] for r in labels["human_development"] if r["review_id"] == "H0653")
    preflight_ids = parent["preflight_ids"] + [
        "actual_membership_conflict:ab",
        "actual_membership_conflict:ba",
        translation_id,
    ]
    require(
        len(preflight_ids) == len(set(preflight_ids)) == 11,
        "CONTEXT_PREFLIGHT",
        "retain original eight plus three repair probes",
    )
    manifest = {
        **{k: v for k, v in parent.items() if k not in {"contract_digest", "input_files", "created_at_utc"}},
        "version": VERSION,
        "schema_version": "dedup-context-diagnostic-v1",
        "response_contract": CONTRACT,
        "routing_contract": ROUTING,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_contract_digest": parent["contract_digest"],
        "preflight_ids": preflight_ids,
        "base_request_upper_bound": 222,
        "settings": parent["settings"]
        | {"prompt_version": VERSION, "runner_config": str(CONFIG), "ray_temp_dir": "/raid/hfang/ihb/r31diag"},
        "frozen_files": parent["frozen_files"]
        | parent["input_files"]
        | {str(PARENT / "manifest.json"): sha256_file(PARENT / "manifest.json")}
        | {str(p.resolve()): sha256_file(p) for p in files},
        "comparison": "Fresh V5 main versus same-main plus critic; new contextual contradiction contract and whole-pair translation instructions.",
        "notes": "Original eight preflight plus closed-list mirrors and H0653 (eleven); same formal 24 real/26 synthetic, labels, weights, order, model, context/retry gates. No semantic ranking against incomplete .30.",
    }
    for name in ("labels_private.json", "historical_12_private.json"):
        write_json_atomic(root / name, json.loads((PARENT / name).read_text()))
    for repeat in (1, 2):
        common._write_jsonl(
            root / f"input_repeat_{repeat}.jsonl", common._jsonl(PARENT / f"input_repeat_{repeat}.jsonl")
        )
    manifest["input_files"] = {str(p): sha256_file(p) for p in root.glob("*.json*")}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def validate(root: Path) -> dict:
    from eval.dedup.rejudge_comparison import _source_digest

    m = json.loads((root / "manifest.json").read_text())
    require(
        m["version"] == VERSION
        and m["response_contract"] == CONTRACT
        and m["contract_digest"] == sha256_json({k: v for k, v in m.items() if k != "contract_digest"}),
        "CONTEXT_MANIFEST",
        "immutable version and V5 contract required",
    )
    require(
        _source_digest() == m["source_implementation_sha256"]
        and all(sha256_file(p) == h for p, h in (m["frozen_files"] | m["input_files"]).items()),
        "CONTEXT_FREEZE",
        "frozen implementation or inputs changed",
    )
    return m


def run(root: Path, *, env_file: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.cli import _load_repository_env
    from eval.dedup.judging.request_relay import RequestRelay

    m = validate(root)
    require(not (root / "started.json").exists(), "COMPOSITE_STARTED", "never restart an observed run")
    if env_file is not None:
        require(env_file.is_file(), "COMPOSITE_ENV_FILE", "explicit credential file missing")
        _load_repository_env(env_file)
    settings = m["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(
        bool(key) and bool(os.environ.get("RAY_ADDRESS")),
        "COMPOSITE_RUNTIME_UNAVAILABLE",
        "owned Ray and configured credential required",
    )
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
    labels = json.loads((root / "labels_private.json").read_text())
    ContextArm("main", tokenizer=tokenizer).layout(common._jsonl(root / "input_repeat_1.jsonl"), {}, "coverage")
    write_json_atomic(
        root / "started.json",
        {
            "at_utc": datetime.now(UTC).isoformat(),
            "contract_digest": m["contract_digest"],
            "credential_source_path": str(env_file) if env_file else None,
            "credential_value_stored": False,
        },
    )
    report = {
        "version": VERSION,
        "contract_digest": m["contract_digest"],
        "eligible_for_release": False,
        "reference_changed": False,
        "cells": {},
    }
    try:
        with RequestRelay(
            logical_model=settings["logical_model"],
            upstream_base_url=settings["hub_base_url"],
            upstream_model=settings["hub_model"],
            upstream_api_key=key,
            timeout_seconds=settings["timeout_seconds"],
            expected_generation_parameters={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ) as relay:
            for name in m["schedule"]:
                validate(root)
                inputs = common._jsonl(
                    root / ("input_repeat_2.jsonl" if name == "repeat_2" else "input_repeat_1.jsonl")
                )
                if name == "preflight":
                    by_id = common._index(inputs, "preflight")
                    inputs = [by_id[pid] for pid in m["preflight_ids"]]
                ops = run_components(
                    root / name,
                    inputs,
                    lambda arm: arm.runtime("coverage", relay.endpoint, settings),
                    relay,
                    settings,
                    tokenizer=tokenizer,
                )
                require(ops["complete"], "COMPOSITE_COMPLETION", "incomplete stage stops schedule")
                if name == "preflight":
                    selected = [
                        r for cohort in labels.values() for r in cohort if r["canonical_pair_id"] in m["preflight_ids"]
                    ]
                    require(len(selected) == 11, "CONTEXT_PREFLIGHT", "all eleven preflight references required")
                    comparison = compare(root / name, inputs, selected)
                    passed = previous.preflight_passes(ops, comparison)
                    report["cells"][name] = {"operations": ops, "comparison": comparison, "passed": passed}
                    write_json_atomic(root / "preflight_assessment.json", report["cells"][name])
                    require(
                        passed,
                        "COMPOSITE_PREFLIGHT_FAILED",
                        "all eleven main/final correct and zero corrections required",
                    )
                else:
                    report["cells"][name] = {
                        "operations": ops,
                        **{cohort: compare(root / name, inputs, group) for cohort, group in labels.items()},
                    }
                write_json_atomic(root / f"schedule_after_{name}.json", report)
                print(
                    json.dumps(
                        {
                            "cell": name,
                            "main_calls": ops["main"]["called_pairs"],
                            "critic_calls": ops["critic"]["called_pairs"],
                            "complete": ops["complete"],
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        write_json_atomic(
            root / "stopped.json",
            {
                "at_utc": datetime.now(UTC).isoformat(),
                "issue": common._safe_retry_feedback(exc),
                "eligible_for_release": False,
            },
        )
        write_json_atomic(root / "assessment_partial.json", report)
        raise
    validate(root)
    write_json_atomic(root / "assessment.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.boundary is not None, "CONTEXT_BOUNDARY", "native boundary report required")
        print(json.dumps(prepare(args.root, args.boundary)))
    else:
        run(args.root, env_file=args.env_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
