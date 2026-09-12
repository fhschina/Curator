# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Transport-only .32 diagnostic on immutable local panels and the original development set."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.dedup.analysis import directional_experiment as semantic
from eval.dedup.analysis.bottleneck_audit import (
    FULL_ROOT,
    REFERENCE,
    REFERENCE_SHA256,
    transition_matrix,
)
from eval.dedup.analysis.coverage_experiment import common
from eval.dedup.analysis.development_diagnostic import classify_primary_error
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.directional_runtime import VERSION
from eval.dedup.judging.paced_relay import TRANSPORT_CONTRACT, PacedRelay, TransportProfile
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

PARENT = Path("/raid/hfang/ihb/runs/v0.6.2.32-directional-diagnostic")
PAYLOADS = FULL_ROOT / "data/judge_payloads.jsonl"
PAYLOADS_SHA256 = "1e7acb8dbc7c9e667dc93945066afcfdd389bf6264e702ca72ee15a2fe2ff736"
EXECUTION_CONTRACT = "dedup-paced-development-v1"
CLIENT_CONTEXT_BUDGET = 65536


class PacedArm(semantic.DirectionalArm):
    def __init__(self, stage: str, *, tokenizer: Any = None):
        # The old 32k value is a client guard, not the shared service's resolved context limit.
        super().__init__(stage, tokenizer=None)
        self.budget_tokenizer = tokenizer

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        called, owned = super().layout(inputs, mains, variant)
        if self.budget_tokenizer is not None:
            for packet in called:
                count = token_count(self, packet, self.budget_tokenizer)
                require(
                    count + 4096 + 2048 <= CLIENT_CONTEXT_BUDGET,
                    "PACED_TOKEN_BUDGET",
                    "full original input, output reserve and margin must fit; never truncate",
                    pair_id=packet["canonical_pair_id"],
                    input_tokens=count,
                )
        return called, owned


def token_count(arm: PacedArm, packet: dict, tokenizer: Any) -> int:
    return len(
        tokenizer.apply_chat_template(
            arm.renderers["coverage"](packet),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def transport_receipts(root: Path) -> dict:
    events = [e for p in sorted(root.rglob("events.jsonl")) for e in common._jsonl(p)]
    external = [e for e in events if e.get("external_request") is True]
    last_attempt = {}
    for event in external:
        key = event["logical_request_sequence"]
        if key not in last_attempt or event["upstream_attempt"] > last_attempt[key]["upstream_attempt"]:
            last_attempt[key] = event
    require(
        len({e["sequence"] for e in external}) == len(external),
        "PACED_RECEIPT_DUPLICATE",
        "each upstream attempt needs its own receipt",
    )
    return {
        "external_attempts": len(external),
        "upstream_http_statuses": dict(Counter(str(e["upstream_http_status"]) for e in external)),
        "network_failures": sum(e["upstream_http_status"] is None for e in external),
        "transport_retry_attempts": sum(e["upstream_attempt"] > 1 for e in external),
        "local_rejections": len(events) - len(external),
        "unrecovered_logical_requests": sum(e["upstream_http_status"] != 200 for e in last_attempt.values()),
        "total_admission_wait_seconds": sum(e["admission_wait_seconds"] for e in external),
    }


def accounted_predictions(inputs: list[dict], predictions: list[dict], terminal_ids: list[str]) -> list[dict]:
    packets, accepted = common._index(inputs, "inputs"), common._index(predictions, "accepted predictions")
    terminal = set(terminal_ids)
    require(
        len(terminal) == len(terminal_ids)
        and not terminal & accepted.keys()
        and terminal | accepted.keys() == packets.keys(),
        "PACED_DENOMINATOR",
        "every input must have either an accepted result or a recorded terminal failure",
    )
    # These are scoring-only missing-output sentinels, never schema-valid Judge results.
    return [
        accepted.get(pid)
        or {
            "canonical_pair_id": pid,
            "metric_only_missing_output": True,
            **dict.fromkeys((*PRIMARY_FIELDS, "relation_type", "material_difference"), "UNRESOLVED"),
        }
        for pid in packets
    ]


def run_components(
    root: Path, inputs: list[dict], runtime_factory: Any, relay: Any, settings: dict, *, tokenizer: Any = None
) -> dict:
    require(not root.exists(), "PACED_CELL_EXISTS", "fresh cell required")
    stages, predictions = {}, {}
    packets, mains = inputs, {}
    for stage in ("main", "critic"):
        arm = PacedArm(stage, tokenizer=tokenizer)
        with runtime_factory(arm) as runtime:
            ops = arm.run_cell(root / stage, packets, mains, runtime, relay, settings, "coverage")
        rows, _ = arm.replay_cell(root, stage, "coverage", packets, mains)
        stages[stage], predictions[stage] = ops, rows
        receipts = transport_receipts(root / stage)
        write_json_atomic(root / f"{stage}_transport.json", receipts)
        require(
            not ops["fatal_boundary"]
            and receipts["local_rejections"] == 0
            and receipts["unrecovered_logical_requests"] == 0
            and all(k in {"200", "429", "502", "503", "504", "None"} for k in receipts["upstream_http_statuses"]),
            "PACED_FATAL_EXECUTION",
            "boundary or unrecovered transport failure stops the diagnostic",
        )
        if stage == "main":
            valid_ids = {r["canonical_pair_id"] for r in rows}
            packets, mains = semantic.critic_inputs([p for p in inputs if p["canonical_pair_id"] in valid_ids], rows)
    main_terminal = json.loads((root / "main/terminal_pair_ids.json").read_text())
    critic_terminal = json.loads((root / "critic/terminal_pair_ids.json").read_text())
    scoring = {
        "main": accounted_predictions(inputs, predictions["main"], main_terminal),
        "final": accounted_predictions(inputs, predictions["critic"], main_terminal + critic_terminal),
    }
    for name, rows in scoring.items():
        common._write_jsonl(root / f"{name}_scoring_only.jsonl", rows)
    result = {
        **stages,
        "requested": len(inputs),
        "final_valid": len(predictions["critic"]),
        "final_terminal_errors": len(main_terminal) + len(critic_terminal),
        "complete": not main_terminal and not critic_terminal,
        "transport": transport_receipts(root),
    }
    write_json_atomic(root / "operations.json", result)
    return result


def compare(root: Path, labels: list[dict]) -> dict:
    ids = {r["canonical_pair_id"] for r in labels}
    main, final = (
        [r for r in common._jsonl(root / f"{s}_scoring_only.jsonl") if r["canonical_pair_id"] in ids]
        for s in ("main", "final")
    )
    result = transition_matrix(labels, main, final)
    result["interpretation"] = (
        "Same fresh .32 main, with and without critic; terminal failures remain in denominators."
    )
    predicted = common._index(final, "final scoring")
    errors = [
        {
            "review_id": label["review_id"],
            "canonical_pair_id": label["canonical_pair_id"],
            "weight": _weight(label),
            "reason": label.get("human_reason_code"),
            "error": classify_primary_error(label, predicted[label["canonical_pair_id"]]),
            "missing_output": predicted[label["canonical_pair_id"]].get("metric_only_missing_output", False),
        }
        for label in labels
    ]
    result["error_distribution"] = dict(Counter(r["error"] for r in errors))
    result["errors"] = sorted(
        [r for r in errors if r["error"] != "CORRECT"], key=lambda r: (-r["weight"], r["review_id"])
    )
    return result


def prepare(root: Path, boundary: Path, *, tokenizer: Any, full: bool = True) -> dict:
    require(not root.exists(), "PACED_ROOT_EXISTS", "never reuse a previous execution root")
    parent = semantic.validate(PARENT)
    require(
        sha256_file(PAYLOADS) == PAYLOADS_SHA256 and sha256_file(REFERENCE) == REFERENCE_SHA256,
        "PACED_REFERENCE_CHANGED",
        "original 1000 payloads and reference weights required",
    )
    inputs = [p | {"repair_feedback": ""} for p in common._jsonl(PAYLOADS)]
    labels = _read_csv(REFERENCE)
    require(
        len(inputs) == len(labels) == 1000
        and common._index(inputs, "inputs").keys() == common._index(labels, "reference").keys(),
        "PACED_FULL_MEMBERSHIP",
        "exact original development membership required",
    )
    arm = PacedArm("main", tokenizer=tokenizer)
    called, owned = arm.layout(inputs, {}, "coverage")
    counts = [
        {"canonical_pair_id": p["canonical_pair_id"], "input_tokens": token_count(arm, p, tokenizer)} for p in called
    ]
    counts.sort(key=lambda r: (-r["input_tokens"], r["canonical_pair_id"]))
    evidence = json.loads(boundary.read_text())
    require(
        evidence["passed"]
        and evidence["external_model_calls"] == 0
        and evidence["full_boundary_tests_passed"] == 4
        and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
        "PACED_BOUNDARY",
        "four tested native main/critic first/transport-retry boundaries required",
    )
    files = [
        Path(__file__),
        Path(__file__).with_name("v06232_paced_protocol.md"),
        boundary,
        Path(__file__).parents[1] / "judging/paced_relay.py",
        Path(__file__).parents[3] / "tests/eval/dedup/test_paced_relay.py",
        Path(__file__).parents[3] / "tests/eval/dedup/test_paced_development.py",
    ]
    require(
        {str(p.resolve()) for p in files if p.suffix == ".py"} <= evidence["artifacts"].keys(),
        "PACED_BOUNDARY",
        "all new implementation assets must be tested",
    )
    common._write_jsonl(root / "input_full.jsonl", inputs)
    write_json_atomic(root / "labels_full_private.json", labels)
    write_json_atomic(root / "labels_local_private.json", json.loads((PARENT / "labels_private.json").read_text()))
    for repeat in (1, 2):
        common._write_jsonl(
            root / f"input_repeat_{repeat}.jsonl", common._jsonl(PARENT / f"input_repeat_{repeat}.jsonl")
        )
    write_json_atomic(
        root / "input_token_audit.json",
        {
            "called": len(called),
            "owned": len(owned),
            "counts": counts,
            "above_old_client_guard": sum(r["input_tokens"] + 6144 > 32768 for r in counts),
            "client_context_budget": CLIENT_CONTEXT_BUDGET,
            "resolved_upstream_context_limit": None,
        },
    )
    manifest = {
        "execution_contract": EXECUTION_CONTRACT,
        "version": VERSION,
        "transport_contract": TRANSPORT_CONTRACT,
        "transport_profile": asdict(TransportProfile()),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_contract_digest": parent["contract_digest"],
        "preflight_ids": parent["preflight_ids"],
        "long_input_id": counts[0]["canonical_pair_id"],
        "schedule": ["preflight", "long_input", "repeat_1", "repeat_2"] + (["full"] if full else []),
        "settings": parent["settings"] | {"ray_temp_dir": "/raid/hfang/ihb/r32paced"},
        "frozen_files": parent["frozen_files"]
        | parent["input_files"]
        | {str(p.resolve()): sha256_file(p) for p in [*files, PARENT / "manifest.json", PAYLOADS, REFERENCE]},
        "input_files": {str(p): sha256_file(p) for p in root.glob("*.json*")},
        "reference_status": "Development reference includes prediction-aware AI adjudication; not a blind human holdout.",
        "eligible_for_release": False,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def validate(root: Path) -> dict:
    semantic.validate(PARENT)
    m = json.loads((root / "manifest.json").read_text())
    require(
        m["execution_contract"] == EXECUTION_CONTRACT
        and m["version"] == VERSION
        and m["contract_digest"] == sha256_json({k: v for k, v in m.items() if k != "contract_digest"})
        and all(sha256_file(p) == h for p, h in (m["frozen_files"] | m["input_files"]).items()),
        "PACED_FREEZE",
        "frozen execution, semantic source, payload or reference changed",
    )
    return m


def run(root: Path, *, env_file: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.cli import _load_repository_env

    m = validate(root)
    require(not (root / "started.json").exists(), "PACED_STARTED", "never restart an observed execution")
    _load_repository_env(env_file)
    settings = m["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(
        bool(key) and bool(os.environ.get("RAY_ADDRESS")),
        "PACED_RUNTIME",
        "existing credential and owned Ray required",
    )
    tokenizer = AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
    local_labels = json.loads((root / "labels_local_private.json").read_text())
    full_labels = json.loads((root / "labels_full_private.json").read_text())
    write_json_atomic(
        root / "started.json",
        {
            "at_utc": datetime.now(UTC).isoformat(),
            "contract_digest": m["contract_digest"],
            "credential_source_path": str(env_file),
            "credential_value_stored": False,
        },
    )
    report = {
        "version": VERSION,
        "execution_contract": EXECUTION_CONTRACT,
        "contract_digest": m["contract_digest"],
        "eligible_for_release": False,
        "reference_changed": False,
        "cells": {},
    }
    try:
        with PacedRelay(
            profile=TransportProfile(**m["transport_profile"]),
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
                full_input = name in {"long_input", "full"}
                filename = "input_full.jsonl" if full_input else f"input_repeat_{2 if name == 'repeat_2' else 1}.jsonl"
                inputs = common._jsonl(root / filename)
                groups = {"development": full_labels} if full_input else local_labels
                if name in {"preflight", "long_input"}:
                    ids = m["preflight_ids"] if name == "preflight" else [m["long_input_id"]]
                    indexed = common._index(inputs, "inputs")
                    inputs = [indexed[pid] for pid in ids]
                    groups = {k: [r for r in group if r["canonical_pair_id"] in ids] for k, group in groups.items()}
                ops = run_components(
                    root / name,
                    inputs,
                    lambda arm: arm.runtime("coverage", relay.endpoint, settings),
                    relay,
                    settings,
                    tokenizer=tokenizer,
                )
                report["cells"][name] = {
                    "operations": ops,
                    **{k: compare(root / name, rows) for k, rows in groups.items() if rows},
                }
                write_json_atomic(root / f"schedule_after_{name}.json", report)
                print(
                    json.dumps(
                        {
                            "cell": name,
                            "valid": ops["final_valid"],
                            "requested": len(inputs),
                            "transport": ops["transport"],
                        }
                    ),
                    flush=True,
                )
                if name in {"preflight", "long_input"}:
                    require(
                        ops["complete"],
                        "PACED_TECHNICAL_PREFLIGHT",
                        "all preflight outputs must bind to original input",
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
    report["transport"] = transport_receipts(root)
    write_json_atomic(root / "assessment.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        from transformers import AutoTokenizer

        require(args.boundary is not None, "PACED_BOUNDARY", "boundary report required")
        m = prepare(
            args.root,
            args.boundary,
            tokenizer=AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True),
            full=not args.local_only,
        )
        print(json.dumps({"root": str(args.root), "contract_digest": m["contract_digest"], "schedule": m["schedule"]}))
    else:
        require(args.env_file is not None, "PACED_CREDENTIAL", "explicit existing env file required")
        run(args.root, env_file=args.env_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
