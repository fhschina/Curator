# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Immutable format-only follow-up with fresh main/critic outputs and unchanged guards."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from eval.dedup.analysis import composite_experiment as base
from eval.dedup.analysis.bottleneck_audit import transition_matrix
from eval.dedup.analysis.coverage_experiment import common
from eval.dedup.judging.composite_coverage import adapt_composite, owned_output
from eval.dedup.judging.composite_format_runtime import (
    CONFIG,
    VERSION,
    FormatRuntime,
    format_blocks,
    format_renderer,
)
from eval.dedup.judging.composite_runtime import composite_renderer, encode_main_review
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

PARENT = Path("/raid/hfang/ihb/runs/v0.6.2.29-composite-diagnostic")


class FormatArm(base.CompositeArm):
    def __init__(self, stage: str, path: Path = CONFIG, *, tokenizer: Any = None):
        super().__init__(stage, path, tokenizer=tokenizer)
        self.spec = {"version": VERSION}

    @cached_property
    def renderers(self) -> dict:
        return {"coverage": format_renderer(self.config["coverage"], stage=self.stage)}

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        called, owned = super().layout(inputs, mains, variant)
        return called, [{**r, "diagnostic_version": VERSION} for r in owned]

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        # The inherited binder checks the actual new prompts before these rows are
        # versioned; historical .29 traces cannot be relabeled as .30 responses.
        return [{**r, "diagnostic_version": VERSION} for r in super().bind(inputs, outputs, mains, variant)]

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "COMPOSITE_FORMAT_ARM", "single stage arm required")
        return FormatRuntime(
            self.config[variant],
            stage=self.stage,
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )


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
            "this version's fresh main and original payload required",
        )
        prepared = {k: v for k, v in packet.items() if k != "repair_feedback"} | {"repair_feedback": ""}
        if owned_output(packet["payload"]) is None:
            require("coverage_response" in row, "COMPOSITE_MAIN_PROVENANCE", "fresh certificate required")
            mains[pid] = row["coverage_response"]
            public = adapt_composite(mains[pid], packet["payload"])
            require(all(row[k] == v for k, v in public.items()), "COMPOSITE_MAIN_PROVENANCE", "public/raw mismatch")
            if public["same_duplicate_group"] == "YES":
                prepared["proposed_main_text"] = encode_main_review(mains[pid])
        result.append(prepared)
    return result, mains


def run_components(
    root: Path, inputs: list[dict], runtime_factory: Any, relay: Any, settings: dict, *, tokenizer: Any = None
) -> dict:
    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "fresh cell required; no old main or cache reuse")
    arms = {s: FormatArm(s, tokenizer=tokenizer) for s in ("main", "critic")}
    with runtime_factory(arms["main"]) as runtime:
        main_ops = arms["main"].run_cell(root / "main", inputs, {}, runtime, relay, settings, "coverage")
    write_json_atomic(root / "main_operations.json", main_ops)
    require(
        main_ops["errors"] == 0 and main_ops["valid"] == len(inputs), "COMPOSITE_MAIN_COMPLETION", "incomplete main"
    )
    main_rows, _ = arms["main"].replay_cell(root, "main", "coverage", inputs, {})
    packets, mains = critic_inputs(inputs, main_rows)
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
    main, _ = FormatArm("main").replay_cell(root, "main", "coverage", inputs, {})
    packets, certificates = critic_inputs(inputs, main)
    final, _ = FormatArm("critic").replay_cell(root, "critic", "coverage", packets, certificates)
    ids = {r["canonical_pair_id"] for r in labels}
    return transition_matrix(
        labels, [r for r in main if r["canonical_pair_id"] in ids], [r for r in final if r["canonical_pair_id"] in ids]
    )


def prompt_contrast(inputs: list[dict]) -> list[dict]:
    """Verify original payload, semantic instructions and schema are byte-identical."""
    old, new = composite_renderer(stage="main"), format_renderer(stage="main")
    system_block, pair_block = format_blocks()
    rows = []
    for packet in inputs:
        before, after = old(packet), new(packet)
        normalized = [
            {**after[0], "content": after[0]["content"].removesuffix(system_block).replace(VERSION, base.VERSION)},
            {**after[1], "content": after[1]["content"].replace(pair_block, "", 1)},
        ]
        require(normalized == before, "COMPOSITE_FORMAT_CONTRAST", "only role version and format blocks may differ")
        rows.append(
            {
                "canonical_pair_id": packet["canonical_pair_id"],
                "base_messages_sha256": sha256_json(before),
                "candidate_messages_sha256": sha256_json(after),
                "normalized_identical": True,
            }
        )
    return rows


def prepare(root: Path, boundary: Path) -> dict:
    require(not root.exists(), "COMPOSITE_ROOT_EXISTS", "fresh diagnostic root required")
    parent = base.validate_diagnostic(PARENT)
    evidence = json.loads(boundary.read_text())
    require(
        evidence["passed"]
        and evidence["external_model_calls"] == 0
        and evidence["full_boundary_tests_passed"] == 4
        and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
        "COMPOSITE_FORMAT_BOUNDARY",
        "current main/critic first/retry native boundary required",
    )
    analysis = Path(__file__).parent
    files = [
        Path(__file__),
        analysis.parent / "judging/composite_format_runtime.py",
        CONFIG,
        CONFIG.parent / "hs_v06230_branch_format.jinja",
        analysis / "v06230_design.md",
        boundary,
        *[
            analysis.parents[2] / "tests/eval/dedup" / f"test_{n}.py"
            for n in ("composite_format_runtime", "composite_format_experiment")
        ],
    ]
    require(
        {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} <= evidence["artifacts"].keys(),
        "COMPOSITE_FORMAT_BOUNDARY",
        "all new implementation assets must be tested",
    )
    inputs = common._jsonl(PARENT / "input_repeat_1.jsonl")
    contrast = prompt_contrast(inputs)
    manifest = {
        **{k: v for k, v in parent.items() if k not in {"contract_digest", "input_files", "created_at_utc"}},
        "version": VERSION,
        "schema_version": "dedup-composite-format-diagnostic-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_contract_digest": parent["contract_digest"],
        "settings": parent["settings"]
        | {"prompt_version": VERSION, "runner_config": str(CONFIG), "ray_temp_dir": "/raid/hfang/ihb/r30diag"},
        "frozen_files": parent["frozen_files"]
        | parent["input_files"]
        | {str(PARENT / "manifest.json"): sha256_file(PARENT / "manifest.json")}
        | {str(p.resolve()): sha256_file(p) for p in files},
        "comparison": "Fresh main versus the SAME main plus critic. Only encoding instructions differ from .29; no complete .29 semantic comparator exists.",
        "notes": "Identical .29 labels, weights, payloads, repeat order, schema, semantic policy, model and gates. Never restart observed runs; zero-correction preflight before both repeats.",
    }
    for name in ("labels_private.json", "historical_12_private.json"):
        write_json_atomic(root / name, json.loads((PARENT / name).read_text()))
    for repeat in (1, 2):
        common._write_jsonl(
            root / f"input_repeat_{repeat}.jsonl", common._jsonl(PARENT / f"input_repeat_{repeat}.jsonl")
        )
    write_json_atomic(root / "prompt_contrast.json", contrast)
    manifest["input_files"] = {str(p): sha256_file(p) for p in root.glob("*.json*")}
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def validate(root: Path) -> dict:
    from eval.dedup.rejudge_comparison import _source_digest

    m = json.loads((root / "manifest.json").read_text())
    require(
        m["version"] == VERSION
        and m["contract_digest"] == sha256_json({k: v for k, v in m.items() if k != "contract_digest"}),
        "COMPOSITE_FORMAT_MANIFEST",
        "immutable version and manifest required",
    )
    require(
        _source_digest() == m["source_implementation_sha256"]
        and all(sha256_file(p) == h for p, h in (m["frozen_files"] | m["input_files"]).items()),
        "COMPOSITE_FORMAT_FREEZE",
        "frozen implementation, inputs or labels changed",
    )
    return m


def preflight_passes(ops: dict, comparison: dict) -> bool:
    return (
        ops["complete"]
        and all(
            ops[s]["judge_retried_pairs"] == 0 and ops[s]["http_statuses"] == {"200": ops[s]["called_pairs"]}
            for s in ("main", "critic")
        )
        and set(comparison["transitions"]) == {"CORRECT -> CORRECT"}
    )


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
        "owned Ray cluster and configured credential required",
    )
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
    labels = json.loads((root / "labels_private.json").read_text())
    FormatArm("main", tokenizer=tokenizer).layout(common._jsonl(root / "input_repeat_1.jsonl"), {}, "coverage")
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
                    selected = [r for r in labels["synthetic_policy"] if r["canonical_pair_id"] in m["preflight_ids"]]
                    comparison = compare(root / name, inputs, selected)
                    passed = preflight_passes(ops, comparison)
                    report["cells"][name] = {"operations": ops, "comparison": comparison, "passed": passed}
                    write_json_atomic(root / "preflight_assessment.json", report["cells"][name])
                    require(
                        passed,
                        "COMPOSITE_PREFLIGHT_FAILED",
                        "all eight main/final correct and zero corrections required",
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
        require(args.boundary is not None, "COMPOSITE_FORMAT_BOUNDARY", "boundary report required")
        print(json.dumps(prepare(args.root, args.boundary)))
    else:
        run(args.root, env_file=args.env_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
