# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh-main/critic component diagnostics using the existing immutable cell engine."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from eval.dedup.analysis.bottleneck_audit import transition_matrix
from eval.dedup.analysis.coverage_experiment import CoverageExperiment, common, native
from eval.dedup.judging.composite_coverage import (
    CONTRACT,
    ROUTING,
    adapt_composite,
    bind_composite_response,
    composite_schema,
    critic_route,
    finalize_composite,
    owned_output,
)
from eval.dedup.judging.composite_runtime import (
    CONFIG,
    VERSION,
    CompositeRuntime,
    composite_record,
    composite_renderer,
    encode_main_review,
)
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.payload_transport import bind_transported_payload
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class CompositeArm(CoverageExperiment):
    """Use the generic engine's coverage-column accounting for each separate stage."""

    def __init__(self, stage: str, path: Path = CONFIG, *, tokenizer: Any = None):
        require(stage in {"main", "critic"}, "COMPOSITE_STAGE", "unknown stage")
        self.stage = stage
        self.config = {"coverage": path}
        self.spec = {"version": VERSION}
        self.tokenizer = tokenizer

    @cached_property
    def renderers(self) -> dict:
        return {"coverage": composite_renderer(self.config["coverage"], stage=self.stage)}

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        require(variant == "coverage", "COMPOSITE_ARM", "use the shared coverage-column engine")
        require(
            self.stage != "main" or not mains,
            "COMPOSITE_MAIN_REUSE",
            "main stage requires fresh inference, not historical responses",
        )
        common._index(inputs, "composite inputs")
        called, owned = [], []
        for packet in inputs:
            pid = packet["canonical_pair_id"]
            require(
                set(packet)
                <= {"canonical_pair_id", "payload", "judge_payload_hash", "repair_feedback", "proposed_main_text"}
                and packet["judge_payload_hash"] == sha256_json(packet["payload"])
                and isinstance(packet.get("repair_feedback"), str),
                "FOLLOWUP_BOUNDARY_COMPOSITE_INPUT",
                "only blind input, exact hash and stage-local feedback allowed",
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
                        "FOLLOWUP_BOUNDARY_COMPOSITE_MAIN",
                        "critic must observe this pair's exact accepted main response",
                    )
                composite_record(packet, self.stage)
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
                        "do not truncate or drop oversized pairs",
                        pair_id=pid,
                        input_tokens=count,
                    )
                called.append(packet)
            else:
                require(
                    "proposed_main_text" not in packet,
                    "FOLLOWUP_BOUNDARY_COMPOSITE_MAIN",
                    "unrequested branches cannot carry a main claim",
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
        require(variant == "coverage", "COMPOSITE_ARM", "single stage arm required")
        return CompositeRuntime(
            self.config[variant],
            stage=self.stage,
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

        packets, rows = common._index(inputs, "composite inputs"), common._index(outputs, "composite outputs")
        require(packets.keys() == rows.keys(), "EXPERIMENT_RESPONSE_MISSING", "one response for each requested pair")
        called, owned = self.layout(inputs, mains, variant)
        require(
            not owned and len(called) == len(inputs), "FOLLOWUP_BOUNDARY_ROUTING", "owned branches cannot be submitted"
        )
        result = []
        for pid, row in rows.items():
            packet = packets[pid]
            _, payload_audit = bind_transported_payload(packet, row)
            require(
                row.get("repair_feedback") == packet["repair_feedback"]
                and row.get("proposed_main_text") == packet.get("proposed_main_text"),
                "FOLLOWUP_BOUNDARY_COMPOSITE_MAIN",
                "actual main claim or retry feedback changed",
            )
            require(
                not ({"qwen_dedup_semantic_judge", common.RECORD_BINDING_CRITIC_COLUMN, "proposed_main"} & row.keys()),
                "FOLLOWUP_BOUNDARY_COMPOSITE_LEAK",
                "legacy results cannot enter the new stages",
            )
            trace = row.get(COLUMN + "__trace")
            require(
                isinstance(trace, list) and len(trace) >= 3 and all(isinstance(m, dict) for m in trace),
                "FOLLOWUP_BOUNDARY_TRACE",
                "full native conversation required",
            )
            messages = [{"role": m.get("role"), "content": native.trace_text(m.get("content"))} for m in trace[:2]]
            require(
                messages == self.renderers[variant](packet),
                "FOLLOWUP_BOUNDARY_PROMPT",
                "actual initial messages differ from the candidate renderer",
            )
            assistants = [m for m in trace if m.get("role") == "assistant"]
            require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "assistant response missing")
            value = StructuredResponseRecipe(composite_schema(), pruning=False).parse(
                native.trace_text(assistants[-1].get("content"))
            )
            value, response_audit = bind_composite_response(value, row.get(COLUMN))
            public = (
                adapt_composite(value, packet["payload"])
                if self.stage == "main"
                else finalize_composite(mains[pid], value, packet["payload"])
            )
            result.append(
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
        return result


def critic_inputs(inputs: list[dict], main_rows: list[dict]) -> tuple[list[dict], dict]:
    """Require a complete fresh main pass; clear its retry feedback before critic work."""
    packets, predictions = common._index(inputs, "inputs"), common._index(main_rows, "main results")
    require(
        packets.keys() == predictions.keys(),
        "COMPOSITE_MAIN_COMPLETION",
        "do not run critic on an incomplete main pass",
    )
    result, mains = [], {}
    for pid, packet in packets.items():
        row = predictions[pid]
        require(
            row.get("stage") == "main"
            and row.get("diagnostic_version") == VERSION
            and row.get("judge_payload_hash") == packet["judge_payload_hash"],
            "COMPOSITE_MAIN_PROVENANCE",
            "fresh stage and exact payload required",
        )
        prepared = {k: v for k, v in packet.items() if k != "repair_feedback"} | {"repair_feedback": ""}
        if owned_output(packet["payload"]) is None:
            require("coverage_response" in row, "COMPOSITE_MAIN_PROVENANCE", "missing fresh main certificate")
            mains[pid] = row["coverage_response"]
            public = adapt_composite(mains[pid], packet["payload"])
            require(
                all(row[k] == v for k, v in public.items()),
                "COMPOSITE_MAIN_PROVENANCE",
                "main public output disagrees with certificate",
            )
            if public["same_duplicate_group"] == "YES":
                prepared["proposed_main_text"] = encode_main_review(mains[pid])
        result.append(prepared)
    return result, mains


def run_components(
    root: Path, inputs: list[dict], runtime_factory: Any, relay: Any, settings: dict, *, tokenizer: Any = None
) -> dict:
    """Fresh root only. Saved main outputs also supply the no-critic counterfactual."""
    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "never reuse historical cache or an observed run root")
    arms = {s: CompositeArm(s, tokenizer=tokenizer) for s in ("main", "critic")}
    with runtime_factory(arms["main"]) as runtime:
        main_ops = arms["main"].run_cell(root / "main", inputs, {}, runtime, relay, settings, "coverage")
    write_json_atomic(root / "main_operations.json", main_ops)
    require(
        main_ops["errors"] == 0 and main_ops["valid"] == len(inputs),
        "COMPOSITE_MAIN_COMPLETION",
        "stop before critic if main is incomplete",
    )
    main_rows, _ = arms["main"].replay_cell(root, "main", "coverage", inputs, {})
    packets, mains = critic_inputs(inputs, main_rows)
    with runtime_factory(arms["critic"]) as runtime:
        critic_ops = arms["critic"].run_cell(root / "critic", packets, mains, runtime, relay, settings, "coverage")
    arms["critic"].replay_cell(root, "critic", "coverage", packets, mains)
    operations = {
        "main": main_ops,
        "critic": critic_ops,
        "complete": critic_ops["errors"] == 0 and critic_ops["valid"] == len(inputs),
    }
    write_json_atomic(root / "operations.json", operations)
    return operations


def summarize_components(root: Path, inputs: list[dict], labels: list[dict]) -> dict:
    main_rows, main_ops = CompositeArm("main").replay_cell(root, "main", "coverage", inputs, {})
    packets, mains = critic_inputs(inputs, main_rows)
    final_rows, critic_ops = CompositeArm("critic").replay_cell(root, "critic", "coverage", packets, mains)
    require(
        main_ops["errors"] == critic_ops["errors"] == 0, "COMPOSITE_COMPLETION", "no scores for an incomplete schedule"
    )
    comparison = transition_matrix(labels, main_rows, final_rows)
    return {
        "diagnostic_version": VERSION,
        "reference_changed": False,
        "not_release_evidence": True,
        "main_operations": main_ops,
        "critic_operations": critic_ops,
        "same_main_with_and_without_critic": comparison,
    }


def prepare_diagnostic(root: Path, *, boundary: Path) -> dict:
    """Freeze 24 development cases and 26 synthetic cases separately, before model calls."""
    from eval.dedup.analysis.bottleneck_audit import (
        FULL_ROOT,
        LOCAL_ROOT,
        REFERENCE,
        REFERENCE_SHA256,
        component_replay,
    )
    from eval.dedup.analysis.composite_containment_review import SPEC, regression_packets

    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "fresh diagnostic root required")
    parent = common.validate_freeze(LOCAL_ROOT)
    evidence = json.loads(boundary.read_text())
    require(
        evidence["passed"]
        and evidence["external_model_calls"] == 0
        and evidence["full_boundary_tests_passed"] == 4
        and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
        "COMPOSITE_BOUNDARY",
        "current main/critic native first/retry validation required",
    )
    require(
        sha256_file(REFERENCE) == REFERENCE_SHA256, "COMPOSITE_REFERENCE", "historical reference must remain unchanged"
    )
    main, final, payloads, provenance = component_replay(FULL_ROOT, "v6", main_policy="v6-route")
    analysis = Path(__file__).parent
    panel_path = analysis / "bottleneck_panel_v1.json"
    panel = json.loads(panel_path.read_text())
    spec = json.loads(SPEC.read_text())
    ids = {r["review_id"] for r in panel["cases"]} | {"H0521", "H0822"}
    require(
        len(ids) == 24, "COMPOSITE_PANEL", "22 frozen guards/clear cases plus two approved-policy development cases"
    )
    labels = [r for r in common._read_csv(REFERENCE) if r["review_id"] in ids]
    require(len(labels) == 24, "COMPOSITE_PANEL", "missing development labels")
    real_ids = {r["canonical_pair_id"] for r in labels}
    real = [p for p in payloads if p["canonical_pair_id"] in real_ids]
    synthetic, expected = regression_packets(spec["examples"])
    synthetic_labels = [
        {
            "canonical_pair_id": r["canonical_pair_id"],
            "review_id": r["canonical_pair_id"],
            "stratum_population_n": 1,
            "stratum_sample_n": 1,
            "human_reason_code": "synthetic_policy",
            **{f"human_{k}": v for k, v in r["expected"].items()},
        }
        for r in expected
    ]
    inputs = [
        {
            "canonical_pair_id": p["canonical_pair_id"],
            "payload": p["payload"],
            "judge_payload_hash": sha256_json(p["payload"]),
            "repair_feedback": "",
        }
        for p in real + synthetic
    ]
    require(
        len(inputs) == 50 and len({p["canonical_pair_id"] for p in inputs}) == 50,
        "COMPOSITE_PANEL",
        "exact joint membership required",
    )
    files = [
        Path(__file__),
        analysis / "bottleneck_audit.py",
        analysis / "composite_containment_review.py",
        analysis / "v06229_design.md",
        SPEC,
        panel_path,
        CONFIG,
        boundary,
        *CONFIG.parent.glob("hs_v06229_*.jinja"),
        *[analysis.parent / "judging" / f"{name}.py" for name in ("composite_coverage", "composite_runtime")],
        *[
            analysis.parents[2] / "tests/eval/dedup" / f"test_{name}.py"
            for name in ("composite_coverage", "composite_runtime", "composite_experiment")
        ],
    ]
    required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}}
    require(
        required <= evidence["artifacts"].keys(),
        "COMPOSITE_BOUNDARY",
        "all implementation assets must be covered by the current boundary report",
    )
    settings = parent["settings"] | {
        "prompt_version": VERSION,
        "runner_config": str(CONFIG),
        "ray_temp_dir": "/raid/hfang/ihb/r29diag",
    }
    require(
        settings["max_output_tokens"] == 4096
        and settings["max_parallel_requests"] == 16
        and settings["temperature"] == 0
        and settings["top_p"] == 1
        and settings["max_retries"] == 2,
        "COMPOSITE_SETTINGS",
        "same model and bounded generation settings required",
    )
    manifest = {
        "schema_version": "dedup-composite-diagnostic-v1",
        "version": VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "response_contract": CONTRACT,
        "routing_contract": ROUTING,
        "reference_changed": False,
        "eligible_for_release": False,
        "selection_is_prediction_aware_development": True,
        "comparison": "Fresh V4 main versus the SAME main followed by V4 critic; old .12 is contextual, not an isolated policy-only ablation.",
        "settings": settings,
        "historical_12_provenance": provenance,
        "frozen_files": parent["frozen_files"]
        | {str(p.resolve()): sha256_file(p) for p in files}
        | {str(REFERENCE): REFERENCE_SHA256},
        "source_implementation_sha256": parent["source_implementation_sha256"],
        "schedule": ["preflight", "repeat_1", "repeat_2"],
        "preflight_ids": [
            f"{name}:{orientation}"
            for name, orientation in (
                ("independent_addition", "ab"),
                ("independent_addition", "ba"),
                ("chrome_only", "ab"),
                ("missing_record_field", "ab"),
                ("cookie_empty_anchor", "ab"),
                ("actual_sku_conflict", "ab"),
                ("actual_state_conflict", "ab"),
                ("faithful_translation", "ab"),
            )
        ],
        "base_request_upper_bound": 216,
        "notes": "No old main/cache reuse. Synthetic and human-development scores stay separate. Preflight requires zero corrections and all eight primary decisions correct; it is not an unseen evaluation set.",
    }
    write_json_atomic(
        root / "labels_private.json", {"human_development": labels, "synthetic_policy": synthetic_labels}
    )
    write_json_atomic(
        root / "historical_12_private.json",
        {
            "main": [r for r in main if r["canonical_pair_id"] in real_ids],
            "final": [r for r in final if r["canonical_pair_id"] in real_ids],
        },
    )
    for repeat in (1, 2):
        common._write_jsonl(
            root / f"input_repeat_{repeat}.jsonl",
            sorted(inputs, key=lambda r: sha256_json([VERSION, repeat, r["canonical_pair_id"]])),
        )
    manifest["input_files"] = {str(p): sha256_file(p) for p in root.glob("*.json*")}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def validate_diagnostic(root: Path) -> dict:
    from eval.dedup.rejudge_comparison import _source_digest

    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest["version"] == VERSION
        and manifest["contract_digest"] == sha256_json({k: v for k, v in manifest.items() if k != "contract_digest"}),
        "COMPOSITE_MANIFEST",
        "manifest changed",
    )
    require(
        _source_digest() == manifest["source_implementation_sha256"],
        "COMPOSITE_SOURCE",
        "shared runtime source changed",
    )
    require(
        all(sha256_file(p) == h for p, h in (manifest["frozen_files"] | manifest["input_files"]).items()),
        "COMPOSITE_FREEZE",
        "code, prompt, labels, settings or input changed",
    )
    return manifest


def _subset_comparison(root: Path, inputs: list[dict], labels: list[dict]) -> dict:
    mains, _ = CompositeArm("main").replay_cell(root, "main", "coverage", inputs, {})
    critic_packets, certificates = critic_inputs(inputs, mains)
    finals, _ = CompositeArm("critic").replay_cell(root, "critic", "coverage", critic_packets, certificates)
    ids = {r["canonical_pair_id"] for r in labels}
    return transition_matrix(
        labels,
        [r for r in mains if r["canonical_pair_id"] in ids],
        [r for r in finals if r["canonical_pair_id"] in ids],
    )


def run_diagnostic(root: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.judging.request_relay import RequestRelay

    manifest = validate_diagnostic(root)
    require(not (root / "started.json").exists(), "COMPOSITE_STARTED", "never restart an observed diagnostic")
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(
        bool(key) and bool(os.environ.get("RAY_ADDRESS")),
        "COMPOSITE_RUNTIME_UNAVAILABLE",
        "existing owned Ray cluster and configured inference credential required",
    )
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
    labels = json.loads((root / "labels_private.json").read_text())
    # Validate all fresh main payloads before the first external call; the exact
    # main claim is measured again before any critic request by CompositeArm.layout.
    inputs = common._jsonl(root / "input_repeat_1.jsonl")
    CompositeArm("main", tokenizer=tokenizer).layout(inputs, {}, "coverage")
    write_json_atomic(
        root / "started.json",
        {"at_utc": datetime.now(UTC).isoformat(), "contract_digest": manifest["contract_digest"]},
    )
    report = {
        "version": VERSION,
        "contract_digest": manifest["contract_digest"],
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
            for name in manifest["schedule"]:
                validate_diagnostic(root)
                batch = common._jsonl(
                    root / ("input_repeat_2.jsonl" if name == "repeat_2" else "input_repeat_1.jsonl")
                )
                if name == "preflight":
                    by_id = common._index(batch, "preflight")
                    batch = [by_id[pid] for pid in manifest["preflight_ids"]]
                ops = run_components(
                    root / name,
                    batch,
                    lambda arm: arm.runtime("coverage", relay.endpoint, settings),
                    relay,
                    settings,
                    tokenizer=tokenizer,
                )
                require(ops["complete"], "COMPOSITE_COMPLETION", "incomplete stage stops the diagnostic")
                if name == "preflight":
                    selected = [
                        r for r in labels["synthetic_policy"] if r["canonical_pair_id"] in manifest["preflight_ids"]
                    ]
                    comparison = _subset_comparison(root / name, batch, selected)
                    passed = all(
                        ops[s]["judge_retried_pairs"] == 0
                        and ops[s]["http_statuses"] == {"200": ops[s]["called_pairs"]}
                        for s in ("main", "critic")
                    ) and set(comparison["transitions"]) == {"CORRECT -> CORRECT"}
                    report["cells"][name] = {"operations": ops, "comparison": comparison, "passed": passed}
                    write_json_atomic(root / "preflight_assessment.json", report["cells"][name])
                    require(
                        passed,
                        "COMPOSITE_PREFLIGHT_FAILED",
                        "zero corrections and all eight main/final primary decisions correct required; do not submit the formal panel",
                    )
                else:
                    report["cells"][name] = {
                        "operations": ops,
                        **{
                            cohort: _subset_comparison(root / name, batch, cohort_labels)
                            for cohort, cohort_labels in labels.items()
                        },
                    }
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
    validate_diagnostic(root)
    write_json_atomic(root / "assessment.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--boundary", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.boundary is not None, "COMPOSITE_BOUNDARY", "boundary report required")
        print(json.dumps(prepare_diagnostic(args.root, boundary=args.boundary)))
    else:
        run_diagnostic(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
