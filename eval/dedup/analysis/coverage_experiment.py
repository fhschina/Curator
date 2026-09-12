# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configurable paired coverage experiments with explicit call ownership and raw replay."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis import coverage_diagnostic as common
from eval.dedup.analysis import coverage_followup as native
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT, route_coverage
from eval.dedup.judging.payload_transport import bind_transported_payload, encode_repair_feedback
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

REPO = common.ANALYSIS.parents[2]
VARIANTS = ("control", "coverage")
NOT_REQUESTED = "NOT_REQUESTED_OWNED_BRANCH"


def paired_checks(candidate: dict, control: dict) -> dict:
    return {
        **common.diagnostic_checks(candidate, control, candidate["targets"]),
        "control_called_retry_rate_at_most_one_percent": control["operations"]["called_pair_retry_rate"] is not None
        and control["operations"]["called_pair_retry_rate"] <= 0.01,
    }


class CoverageExperiment:
    """Version resources are explicit inputs; frozen historical modules are never patched."""

    def __init__(self, spec_path: Path):
        self.spec_path = spec_path.resolve()
        self.spec = json.loads(self.spec_path.read_text())
        require(
            self.spec["coverage_routing"] == ROUTING_CONTRACT and self.spec["control_routing"] == "ALL_PAIRS",
            "EXPERIMENT_ROUTING_CONTRACT",
            "unsupported routing policy",
        )
        self.config = {v: (REPO / self.spec[v + "_config"]).resolve() for v in VARIANTS}
        require(
            yaml.safe_load(self.config["control"].read_text()) == yaml.safe_load(native.CONTROL_CONFIG.read_text()),
            "EXPERIMENT_CONTROL_CHANGED",
            "control must preserve the verified native control configuration",
        )
        native.validate_control()

    @cached_property
    def renderers(self) -> dict:
        return {
            "control": native.renderer("control"),
            "coverage": common.coverage_renderer(self.config["coverage"]),
        }

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        require(variant in VARIANTS, "EXPERIMENT_VARIANT", "unknown arm")
        common._index(inputs, "routing inputs")
        called, owned = [], []
        for packet in inputs:
            pid = packet["canonical_pair_id"]
            require(pid in mains, "EXPERIMENT_MAIN_MISSING", "fixed main missing")
            require(
                packet["judge_payload_hash"] == sha256_json(packet["payload"]),
                "FOLLOWUP_BOUNDARY_PAYLOAD",
                "original input digest mismatch",
            )
            route = route_coverage(mains[pid], packet["payload"]) if variant == "coverage" else None
            if route is None or route.public_output is None:
                called.append(packet)
            else:
                owned.append(
                    {
                        "canonical_pair_id": pid,
                        "diagnostic_only": True,
                        "diagnostic_version": self.spec["version"],
                        "fixed_main_sha256": sha256_json(mains[pid]),
                        "judge_payload_hash": packet["judge_payload_hash"],
                        "critic_request_status": NOT_REQUESTED,
                        "routing_contract": ROUTING_CONTRACT,
                        "route": route.route,
                        "attempts": 0,
                        "retried": False,
                        **route.public_output,
                    }
                )
        return called, owned

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

        packets, raw = common._index(inputs, "inputs"), common._index(outputs, "outputs")
        require(packets.keys() == raw.keys(), "EXPERIMENT_RESPONSE_MISSING", "every called input needs one response")
        called, owned = self.layout(inputs, mains, variant)
        require(not owned and len(called) == len(inputs), "FOLLOWUP_BOUNDARY_ROUTING", "owned branch was submitted")
        column = common.COLUMN if variant == "coverage" else common.RECORD_BINDING_CRITIC_COLUMN
        predictions = []
        for pid, row in raw.items():
            packet = packets[pid]
            bound, audit = bind_transported_payload(packet, row)
            require(
                isinstance(packet["repair_feedback"], str) and row.get("repair_feedback") == packet["repair_feedback"],
                "FOLLOWUP_BOUNDARY_FEEDBACK",
                "actual retry feedback differs",
            )
            trace = row.get(column + "__trace")
            require(
                isinstance(trace, list) and len(trace) >= 3 and all(isinstance(m, dict) for m in trace),
                "FOLLOWUP_BOUNDARY_TRACE",
                "full native conversation required",
            )
            messages = [{"role": m.get("role"), "content": native.trace_text(m.get("content"))} for m in trace[:2]]
            require(
                messages == self.renderers[variant](packet),
                "FOLLOWUP_BOUNDARY_PROMPT",
                "actual initial request differs from version renderer",
            )
            assistants = [m for m in trace if m.get("role") == "assistant"]
            require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "assistant response missing")
            if variant == "coverage":
                prediction = common.bind_coverage([packet], [bound], mains)[0]
            else:
                require(common.COLUMN not in row, "FOLLOWUP_BOUNDARY_COLUMN", "control cannot observe coverage")
                parsed = StructuredResponseRecipe(native.control_schema(), pruning=False).parse(
                    native.trace_text(assistants[-1].get("content"))
                )
                require(parsed == row.get(column), "EXPERIMENT_RAW_PRUNED", "raw control column differs")
                prediction = common.bind_fixed_main([packet], [bound], mains)[0]
            predictions.append(
                {
                    **prediction,
                    "diagnostic_version": self.spec["version"],
                    "critic_request_status": "REQUESTED",
                    "payload_transport": audit,
                    "native_request_messages_sha256": sha256_json(messages),
                    "native_trace_sha256": sha256_json(trace),
                    "raw_output_sha256": sha256_json(row),
                    "native_corrections": len(assistants) - 1,
                }
            )
        return predictions

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

        if variant == "coverage":
            return common.CoverageWitnessRuntime(
                self.config[variant],
                endpoint=endpoint,
                model_name=settings["logical_model"],
                ray_temp_dir=settings["ray_temp_dir"],
            )
        return ExternalJudgeRuntime(
            self.config[variant],
            endpoint=endpoint,
            provider_api_key="unused",
            served_model_overrides={"judge": settings["logical_model"]},
            ray_temp_dir=settings["ray_temp_dir"],
            num_cpus=None,
        )

    def run_cell(
        self, root: Path, inputs: list[dict], mains: dict, runtime: Any, relay: Any, settings: dict, variant: str
    ) -> dict:
        from eval.dedup.judging.request_relay import RelayContext

        require(not root.exists(), "EXPERIMENT_CELL_EXISTS", "never reuse an observed cell")
        called, owned = self.layout(inputs, mains, variant)
        common._write_jsonl(root / "owned_predictions.jsonl", owned)
        pending, accepted, retried, corrections, fatal = called, [], set(), [], False
        column = common.COLUMN if variant == "coverage" else common.RECORD_BINDING_CRITIC_COLUMN
        for attempt in range(1, settings["max_retries"] + 2):
            if not pending:
                break
            folder = root / f"attempt_{attempt:02d}"
            common._write_jsonl(folder / "input.jsonl", pending)
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
            outputs = common._index(common._read_output_rows(folder / "output"), "native outputs")
            require(
                outputs.keys() <= common._index(pending, "pending").keys(),
                "FOLLOWUP_BOUNDARY_MEMBERSHIP",
                "foreign response",
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
                        {"canonical_pair_id": pid, "outer_attempt": attempt, "native_corrections": max(0, count - 1)}
                    )
                try:
                    prediction = self.bind([packet], [row] if row is not None else [], mains, variant)[0]
                    accepted.append({**prediction, "attempts": attempt, "retried": attempt > 1})
                except Exception as exc:  # noqa: BLE001 - retain model and deterministic failures without substituting answers
                    feedback, is_fatal = common._safe_retry_feedback(exc), native.fatal_boundary(exc)
                    fatal = fatal or is_fatal
                    errors.append({"canonical_pair_id": pid, "issue": feedback, "fatal_boundary": is_fatal})
                    next_pending.append({**packet, "repair_feedback": encode_repair_feedback(feedback)})
            write_json_atomic(folder / "validation_errors.json", errors)
            pending = next_pending
            if not pending or fatal:
                break
            if attempt <= settings["max_retries"]:
                retried.update(p["canonical_pair_id"] for p in pending)
        common._write_jsonl(root / "predictions.jsonl", owned + accepted)
        write_json_atomic(root / "native_corrections.json", corrections)
        write_json_atomic(root / "terminal_pair_ids.json", [p["canonical_pair_id"] for p in pending])
        events = [e for p in root.glob("attempt_*/events.jsonl") for e in common._jsonl(p)]
        native_pairs = {r["canonical_pair_id"] for r in corrections if r["native_corrections"]}
        complete = {
            "requested": len(inputs),
            "valid": len(owned) + len(accepted),
            "errors": len(pending),
            "called_pairs": len(called),
            "not_requested_owned_pairs": len(owned),
            "called_valid": len(accepted),
            "called_final_contract_completion": len(accepted) / len(called) if called else None,
            "retried": len(retried),
            "native_corrections": sum(r["native_corrections"] for r in corrections),
            "native_corrected_pairs": len(native_pairs),
            "judge_retried_pairs": len(retried | native_pairs),
            "called_pair_retry_rate": len(retried | native_pairs) / len(called) if called else None,
            "fatal_boundary": fatal,
            "http_statuses": dict(Counter(str(e["http_status"]) for e in events)),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": {str(p): sha256_file(p) for p in sorted(root.rglob("*.json*")) if p.is_file()},
        }
        write_json_atomic(root / "complete.json", complete)
        return complete

    def replay_cell(
        self, root: Path, name: str, variant: str, inputs: list[dict], mains: dict
    ) -> tuple[list[dict], dict]:
        folder = root / name
        complete = json.loads((folder / "complete.json").read_text())
        require(
            all(sha256_file(p) == h for p, h in complete["artifacts"].items()),
            "EXPERIMENT_RESULTS_CHANGED",
            "cell artifact changed",
        )
        called, owned = self.layout(inputs, mains, variant)
        require(
            common._jsonl(folder / "owned_predictions.jsonl") == owned,
            "EXPERIMENT_OWNED_CHANGED",
            "owned public results changed",
        )
        rows = common._jsonl(folder / "predictions.jsonl")
        packets, published = common._index(called, "called"), common._index(rows, "published")
        owned_index = common._index(owned, "owned")
        terminal = json.loads((folder / "terminal_pair_ids.json").read_text())
        require(
            complete["requested"] == len(inputs)
            and complete["valid"] == len(rows)
            and complete["called_pairs"] == len(called)
            and complete["not_requested_owned_pairs"] == len(owned)
            and complete["called_valid"] == len(rows) - len(owned)
            and len(terminal) == len(set(terminal)) == complete["errors"]
            and not (set(terminal) & published.keys())
            and set(terminal) | published.keys() == common._index(inputs, "inputs").keys(),
            "EXPERIMENT_REPLAY_MEMBERSHIP",
            "all called, owned and terminal pairs must be accounted for",
        )
        raw = {}
        for attempt in sorted(folder.glob("attempt_*")):
            attempted = common._index(common._jsonl(attempt / "input.jsonl"), "attempt inputs")
            require(
                attempted.keys() <= packets.keys(), "EXPERIMENT_REPLAY_CALL_SET", "owned/foreign pair was requested"
            )
            for pid, packet in attempted.items():
                require(
                    {k: v for k, v in packet.items() if k != "repair_feedback"}
                    == {k: v for k, v in packets[pid].items() if k != "repair_feedback"},
                    "EXPERIMENT_REPLAY_INPUT",
                    "attempt changed original input",
                )
            for row in common._read_output_rows(attempt / "output"):
                pid = row["canonical_pair_id"]
                require(pid in attempted, "EXPERIMENT_REPLAY_RAW", "foreign raw response")
                raw[(pid, int(attempt.name.split("_")[1]), sha256_json(row))] = (attempted[pid], row)
        for prediction in rows:
            pid = prediction["canonical_pair_id"]
            if pid in owned_index:
                require(
                    prediction == owned_index[pid],
                    "EXPERIMENT_OWNED_CHANGED",
                    "owned row cannot claim a response or proof",
                )
                continue
            key = (pid, prediction["attempts"], prediction["raw_output_sha256"])
            require(key in raw, "EXPERIMENT_RAW_MISSING", "exact accepted attempt/digest missing")
            packet, response = raw[key]
            replay = self.bind([packet], [response], mains, variant)[0]
            require(
                all(prediction[k] == v for k, v in replay.items()), "EXPERIMENT_REPLAY_CHANGED", "bound output changed"
            )
        return rows, {k: v for k, v in complete.items() if k != "artifacts"}

    def call_sets(self, inputs: dict[int, list[dict]], mains: dict) -> dict:
        result = {}
        for repeat, rows in inputs.items():
            for variant in VARIANTS:
                called, owned = self.layout(rows, mains, variant)
                result[f"repeat_{repeat}_{variant}"] = {
                    "all_pair_ids": [p["canonical_pair_id"] for p in rows],
                    "called_pair_ids": [p["canonical_pair_id"] for p in called],
                    "owned_predictions": owned,
                }
        n = self.spec["technical_pairs_per_arm"]
        result["preflight_pair_ids"] = {v: result[f"repeat_1_{v}"]["called_pair_ids"][:n] for v in VARIANTS}
        require(
            all(len(ids) == n for ids in result["preflight_pair_ids"].values()),
            "EXPERIMENT_PREFLIGHT_SIZE",
            "insufficient called inputs",
        )
        return result

    def token_preflight(self, inputs: list[dict], mains: dict, tokenizer: Any) -> dict:
        rows = []
        for variant in VARIANTS:
            called, _ = self.layout(inputs, mains, variant)
            for packet in called:
                messages = self.renderers[variant](packet)
                count = len(
                    tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(
                    count + 4096 + 2048 <= 32768,
                    "EXPERIMENT_TOKEN_BUDGET",
                    "do not truncate or exclude oversized pairs",
                )
                rows.append(
                    {
                        "variant": variant,
                        "canonical_pair_id": packet["canonical_pair_id"],
                        "input_tokens": count,
                        "message_sha256": sha256_json(messages),
                    }
                )
        return {
            "context_budget": 32768,
            "output_budget": 4096,
            "safety_margin": 2048,
            "budget_source": "client safety budget, not service context limit",
            "rows": rows,
            "max_input_tokens": {v: max(r["input_tokens"] for r in rows if r["variant"] == v) for v in VARIANTS},
        }

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "EXPERIMENT_ROOT_EXISTS", "use a fresh run root")
        old = common.validate_freeze(parent)
        report = json.loads(boundary.read_text())
        test = REPO / "tests/eval/dedup/test_coverage_experiment.py"
        coverage = yaml.safe_load(self.config["coverage"].read_text())
        required = {
            str(p.resolve())
            for p in (
                Path(__file__),
                test,
                self.spec_path,
                *self.config.values(),
                self.config["coverage"].parent / coverage["system_prompt_path"],
                self.config["coverage"].parent / coverage["prompt_path"],
                common.ANALYSIS.parent / "judging/coverage_routing.py",
            )
        }
        require(
            report["passed"]
            and report["external_model_calls"] == 0
            and report["full_boundary_tests_passed"] == 4
            and required <= report["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in report["artifacts"].items()),
            "EXPERIMENT_BOUNDARY_UNVERIFIED",
            "both arms require current full-pipeline first/retry evidence",
        )
        labels = common._read_csv(Path(old["paths"]["labels"]))
        common.validate_projection(labels, common._read_csv(Path(old["paths"]["reference"])))
        inputs = {
            r: [
                {**p, "repair_feedback": ""}
                for p in common.blind_rows(common._jsonl(parent / f"input_repeat_{r}.jsonl"), repeat=r)
            ]
            for r in (1, 2)
        }
        mains = json.loads((parent / "fixed_main.json").read_text())
        require(
            all(
                len(rows) == self.spec["public_pairs_per_cell"] == 258
                and common._index(rows, "inputs").keys() == common._index(labels, "labels").keys() == mains.keys()
                for rows in inputs.values()
            ),
            "EXPERIMENT_MEMBERSHIP",
            "all frozen pairs and main responses required",
        )
        calls = self.call_sets(inputs, mains)
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        files = [
            Path(__file__),
            self.spec_path,
            test,
            boundary,
            REPO / self.spec["design"],
            REPO / self.spec["protocol"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            *self.config.values(),
            self.config["coverage"].parent / coverage["system_prompt_path"],
            self.config["coverage"].parent / coverage["prompt_path"],
            common.ANALYSIS.parent / "judging/coverage_routing.py",
            REPO / "tests/eval/dedup/test_coverage_routing.py",
        ]
        manifest = {
            **{k: v for k, v in old.items() if k not in {"diagnostic_contract_digest", "token_preflight_sha256"}},
            "schema_version": "dedup-routed-coverage-experiment-v1",
            "experiment_spec": str(self.spec_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "parent_diagnostic": str(parent),
            "parent_contract_digest": old["diagnostic_contract_digest"],
            "settings": {**old["settings"], "ray_temp_dir": self.spec["ray_temp_dir"]},
            "paths": {**old["paths"], "protocol": str(REPO / self.spec["protocol"])},
            "frozen_files": old["frozen_files"] | {str(p.resolve()): sha256_file(p) for p in files},
            "variant_contracts": {
                "control": {
                    "runner_config": str(self.config["control"]),
                    "adapter_policy": "v8",
                    "routing": "ALL_PAIRS",
                },
                "coverage": {
                    "runner_config": str(self.config["coverage"]),
                    "adapter_policy": "dedup-retained-coverage-v1",
                    "routing": ROUTING_CONTRACT,
                },
            },
            "call_sets_sha256": sha256_json(calls),
            "token_preflight_sha256": sha256_json(tokens),
            "formal_requests": sum(len(calls[f"repeat_{r}_{v}"]["called_pair_ids"]) for r in (1, 2) for v in VARIANTS),
            "formal_public_pairs": 1032,
        }
        manifest["diagnostic_contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "manifest.json", manifest)
        for name, value in (
            ("fixed_main", mains),
            ("call_sets", calls),
            ("token_preflight", tokens),
            ("coverage_schema", common.coverage_schema()),
        ):
            write_json_atomic(root / f"{name}.json", value)
        for r, rows in inputs.items():
            common._write_jsonl(root / f"input_repeat_{r}.jsonl", rows)
        write_json_atomic(
            root / "input_freeze.json",
            {str(p): sha256_file(p) for p in root.glob("*.json*") if p.name != "manifest.json"},
        )
        return manifest

    def frozen_inputs(self, root: Path) -> tuple[dict, dict, dict, dict]:
        manifest = common.validate_freeze(root)
        require(
            manifest.get("experiment_spec") == str(self.spec_path),
            "EXPERIMENT_SPEC_MISMATCH",
            "run belongs to another spec",
        )
        inputs = {r: common._jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
        mains = json.loads((root / "fixed_main.json").read_text())
        calls = self.call_sets(inputs, mains)
        require(
            calls == json.loads((root / "call_sets.json").read_text())
            and sha256_json(calls) == manifest["call_sets_sha256"],
            "EXPERIMENT_CALL_SET_CHANGED",
            "pre-request call set changed",
        )
        return manifest, inputs, mains, calls

    def run(self, root: Path) -> None:
        from eval.dedup.judging.request_relay import RequestRelay

        manifest, inputs, mains, calls = self.frozen_inputs(root)
        require(
            not (root / "started.json").exists(), "EXPERIMENT_ALREADY_STARTED", "never restart an observed experiment"
        )
        settings = manifest["settings"]
        key = os.environ.get(settings["api_key_env"], "").strip()
        require(bool(key), "EXPERIMENT_CREDENTIAL_MISSING", "inference credential unavailable")
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
                self.runtime("control", relay.endpoint, settings),
            ):
                preflights = {}
                for spec in [{"repeat": 0, "variant": v} for v in VARIANTS] + manifest["schedule"]:
                    self.frozen_inputs(root)
                    repeat, variant = spec["repeat"], spec["variant"]
                    if repeat:
                        require(
                            len(preflights) == 2 and all(common.technical_passed(c) for c in preflights.values()),
                            "EXPERIMENT_PREFLIGHT_FAILED",
                            "failed technical preflight stops formal submission",
                        )
                    name = f"repeat_{repeat}_{variant}" if repeat else f"preflight_{variant}"
                    index = common._index(inputs[1], "preflight inputs")
                    batch = inputs[repeat] if repeat else [index[p] for p in calls["preflight_pair_ids"][variant]]
                    with self.runtime(variant, relay.endpoint, settings) as runtime:
                        complete = self.run_cell(root / name, batch, mains, runtime, relay, settings, variant)
                    print(
                        json.dumps({"cell": name, **{k: v for k, v in complete.items() if k != "artifacts"}}),
                        flush=True,
                    )
                    require(
                        not complete["fatal_boundary"],
                        "EXPERIMENT_BOUNDARY_FAILED",
                        "deterministic boundary failure stops the experiment",
                    )
                    if not repeat:
                        preflights[variant] = complete
                    else:
                        require(
                            complete["valid"] == len(batch) and complete["errors"] == 0,
                            "EXPERIMENT_CELL_FAILED",
                            "incomplete formal cell stops the schedule",
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
            raise
        self.frozen_inputs(root)
        write_json_atomic(
            root / "run_complete.json",
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "formal_public_pairs": 1032,
                "formal_base_model_requests": sum(
                    len(calls[f"repeat_{r}_{v}"]["called_pair_ids"]) for r in (1, 2) for v in VARIANTS
                ),
                "preflight_base_model_requests": sum(map(len, calls["preflight_pair_ids"].values())),
                "eligible_for_release": False,
            },
        )

    def summarize(self, root: Path, output: Path) -> dict:
        manifest, inputs, mains, calls = self.frozen_inputs(root)
        result = {
            "schema_version": "dedup-routed-coverage-assessment-v1",
            "version": self.spec["version"],
            "diagnostic_only": True,
            "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
            "eligible_for_release": False,
            "reference_changed": False,
            "supported_for_separate_fresh_local_validation": False,
            "preflights": {},
        }
        index = common._index(inputs[1], "preflight inputs")
        for variant in VARIANTS:
            name = f"preflight_{variant}"
            if (root / name / "complete.json").is_file():
                _, ops = self.replay_cell(
                    root, name, variant, [index[p] for p in calls["preflight_pair_ids"][variant]], mains
                )
                result["preflights"][variant] = {"operations": ops, "technical_passed": common.technical_passed(ops)}
        if not (root / "run_complete.json").is_file():
            require(
                (root / "stopped.json").is_file(), "EXPERIMENT_STILL_RUNNING", "do not score an active or unknown run"
            )
            result.update(
                status="STOPPED_BEFORE_COMPLETE_FORMAL_SCHEDULE",
                stop=json.loads((root / "stopped.json").read_text()),
                completed_formal_cells={},
            )
            for spec in manifest["schedule"]:
                name = f"repeat_{spec['repeat']}_{spec['variant']}"
                if (root / name / "complete.json").is_file():
                    _, ops = self.replay_cell(root, name, spec["variant"], inputs[spec["repeat"]], mains)
                    result["completed_formal_cells"][name] = ops
            write_json_atomic(output, result)
            return result
        labels = common._read_csv(Path(manifest["paths"]["labels"]))
        _, _, baseline, _ = common.load_run(Path(manifest["baseline_root"]))
        baseline = [r for r in baseline if r["canonical_pair_id"] in index]
        baseline_metrics = common.evaluate_predictions(labels, baseline)
        negatives = json.loads(Path(manifest["paths"]["negative_guards"]).read_text())[
            "critic_repaired_negative_guards"
        ]
        additional = json.loads(Path(manifest["paths"]["additional_guards"]).read_text())
        targets = [
            {**r, "final": {k: r[f"human_{k}"] for k in common.PRIMARY_FIELDS}}
            for r in labels
            if r["review_id"] in common.TARGETS
        ]
        require(
            len(negatives) == 38 and len(additional) == 22 and len(targets) == 3,
            "EXPERIMENT_GUARDS",
            "all fixed guards required",
        )
        cells, predictions = {}, {}
        for spec in manifest["schedule"]:
            name = f"repeat_{spec['repeat']}_{spec['variant']}"
            rows, ops = self.replay_cell(root, name, spec["variant"], inputs[spec["repeat"]], mains)
            require(
                ops["requested"] == ops["valid"] == 258 and ops["errors"] == 0,
                "EXPERIMENT_INCOMPLETE",
                "all pairs required",
            )
            predictions[name] = rows
            metrics = common.evaluate_predictions(labels, rows)
            guards = common.guard_report(labels, rows, baseline, negatives)
            guards["additional_critic_repair_guards"] = common.evaluate_primary_guards(additional, rows)
            gates = common.local_gates(metrics, baseline_metrics, guards, ops, 258)
            gates["checks"]["all_called_judge_retried_pairs_at_most_one_percent"] = (
                ops["called_pair_retry_rate"] is not None and ops["called_pair_retry_rate"] <= 0.01
            )
            gates["passed"] = all(gates["checks"].values())
            cells[name] = {
                "metrics": metrics,
                "operations": ops,
                "guards": guards,
                "targets": common.evaluate_primary_guards(targets, rows),
                "local_gates": gates,
            }
        checks = {
            f"repeat_{r}_coverage": paired_checks(cells[f"repeat_{r}_coverage"], cells[f"repeat_{r}_control"])
            for r in (1, 2)
        }
        repeats = {
            v: common.primary_changes(labels, predictions[f"repeat_1_{v}"], predictions[f"repeat_2_{v}"])
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
                str(r): common.primary_changes(
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
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = CoverageExperiment(args.spec)
    if args.action == "prepare":
        require(
            args.parent is not None and args.boundary is not None,
            "EXPERIMENT_PREPARE_INPUT",
            "supply parent and current boundary evidence",
        )
        print(
            json.dumps(
                {"contract": experiment.prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}
            )
        )
    elif args.action == "run":
        experiment.run(args.root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "EXPERIMENT_OUTPUT_REQUIRED",
            "supply a fresh report path",
        )
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
