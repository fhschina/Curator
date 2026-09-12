# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired named-target scope trials over identical, frozen full-run coverage."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from eval.dedup.analysis import critic_subject_experiment as previous
from eval.dedup.judging import critic_subject_kinds as candidate
from eval.dedup.judging import critic_subject_scope as control
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

selected = previous.selected
reference = selected.base.previous.reference
SYSTEM = previous.SYSTEM.with_name("v06212_subject_kinds_v3_system.txt")
PANEL = (
    "H0007",
    "H0010",
    "H0017",
    "H0022",
    "H0165",
    "H0195",
    "H0289",
    "H0312",
    "H0333",
    "H0347",
    "H0606",
    "H0641",
    "H0720",
    "H0752",
    "H0876",
    "H0878",
    "H0893",
    "H0669",
    "H0820",
    "H0809",
    "H0186",
    "H0597",
    "H0977",
    "H0115",
    "H0056",
    "H0221",
    "H0413",
    "H0636",
    "H0700",
    "H0889",
    "H0920",
    "H0594",
)
REPAIRS = ("H0606", "H0809")
ARMS = {"subject_control": control, "candidate": candidate}


def reviewed_sources(parent: Path, next_stage: str) -> dict:
    sources = reference.verify_freeze(parent / "manifest.json")
    assessment = json.loads((parent / "assessment.json").read_text())
    review = json.loads((parent / "review_complete.json").read_text())
    errors = {p["review_id"] for c in assessment["cells"].values() for p in c["pairs"] if p["requires_cause_review"]}
    require(
        review.get("assessment_sha256") == sha256_file(parent / "assessment.json")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == next_stage,
        "SUBJECT_KINDS_REVIEW",
        "all actual parent errors must be reviewed before another iteration",
    )
    for path, digest in review.get("review_sources", {}).items():
        require(sha256_file(Path(path)) == digest, "SUBJECT_KINDS_REVIEW_SOURCE", "unchanged source review required")
        sources[path] = digest
    for name in ("assessment.json", "review_complete.json", "complete.json"):
        sources[str(parent / name)] = sha256_file(parent / name)
    return sources


def prepare(root: Path, parent: Path, *, full: bool = False) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "SUBJECT_KINDS_ROOT", "new run root required")
    sources = reviewed_sources(parent, "NAMED_TARGET_FULL1000" if full else "NAMED_TARGET_PROOF_PILOT")
    if full:
        prior = json.loads((parent / "assessment.json").read_text())
        require(prior["named_target_pilot_gate"], "SUBJECT_KINDS_GATE", "both paired pilot repeats must pass")
        origin = Path(json.loads((parent / "manifest.json").read_text())["coverage_origin"])
    else:
        origin = parent
    sources.update(reference.verify_freeze(origin / "manifest.json"))
    projection = json.loads((origin / "manifest.json").read_text())["transport_projection"]
    old = json.loads((origin / "specialist/manifest.json").read_text())
    rows = json.loads((origin / "specialist/panel_private.json").read_text())
    coverage = json.loads((origin / "coverage/assessment.json").read_text())
    require(
        coverage["cells"]["1/candidate"]["engineering_failures"] == 0,
        "SUBJECT_KINDS_BASE",
        "no missing coverage may be hidden by a specialist bypass",
    )
    rows = [r for r in rows if full or r["review_id"] in PANEL]
    require(len(rows) == (1000 if full else 32), "SUBJECT_KINDS_POPULATION", "complete declared population")
    for row in rows:
        row["original_main_public"] = deepcopy(row["main_public"])
        row["main_public"] = deepcopy(row["coverage_base"]["1"])
    specs = {name: {"adapter": module.__name__} for name, module in ARMS.items()}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests = []
    repeats = 1 if full else 2
    for rep in range(1, repeats + 1):
        for row in rows:
            if candidate.route(row["main_public"], row["payload"]) != "REVIEW_BILATERAL_SUBJECTS":
                continue
            for name, module in ARMS.items():
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **selected.base.previous.GENERATION,
                    "messages": module.messages(
                        row["payload"], (SYSTEM if name == "candidate" else previous.SYSTEM).read_text()
                    ),
                    "response_format": previous.transport.response_format(
                        module.response_schema(row["payload"]), projection
                    ),
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "SUBJECT_KINDS_CONTEXT", "no source truncation")
                requests.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": rep,
                        "arm": name,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                        "deadline_utc_epoch": selected.base.DEADLINE,
                    }
                )
    for path in (
        Path(__file__).resolve(),
        Path(candidate.__file__).resolve(),
        SYSTEM,
        Path(__file__).with_name("critic_subject_kinds_v3.md"),
        origin / "specialist/panel_private.json",
        origin / "coverage/assessment.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests)
    manifest = {
        "contract_version": candidate.CONTRACT,
        "stage": "named_target_full1000" if full else "named_target_pilot32",
        "population": len(rows),
        "repeat_count": repeats,
        "arms": list(ARMS),
        "specs": specs,
        "coverage_origin": str(origin),
        "parent": str(parent),
        "model": old["model"],
        "endpoint": old["endpoint"],
        "main_online_calls": 0,
        "coverage_online_calls": 0,
        "logical_requests": len(requests),
        "reference_changed": False,
        "deadline_utc_epoch": selected.base.DEADLINE,
        "sources": sources,
        "artifacts": {str(root / n): sha256_file(root / n) for n in ("panel_private.json", "requests_frozen.json")},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def assess(root: Path) -> dict:
    reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    expected = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in requests}
    receipts = [json.loads(f.read_text()) for f in (root / "responses").glob("*.json")]
    found = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in receipts}
    require(
        len(found) == len(receipts) and found.keys() <= expected.keys(),
        "SUBJECT_KINDS_RECEIPTS",
        "known unique responses",
    )
    result = {
        "population": len(rows),
        "stage": manifest["stage"],
        "reference_status": reference.DRAFT_STATUS,
        "main_online_calls": 0,
        "reference_changed": False,
        "release_eligible": False,
        "cells": {},
    }
    for rep in range(1, manifest["repeat_count"] + 1):
        for arm in (*ARMS, "saved_coverage"):
            predictions, pairs = [], []
            for row in rows:
                public, status, rule = row["main_public"], "DETERMINISTIC_BYPASS", "UNCHANGED_COVERAGE"
                if arm != "saved_coverage" and candidate.route(public, row["payload"]) == "REVIEW_BILATERAL_SUBJECTS":
                    key = (rep, arm, row["canonical_pair_id"])
                    receipt = found.get(key)
                    if receipt is not None:
                        require(
                            receipt["request_sha256"] == expected[key]["request_sha256"],
                            "SUBJECT_KINDS_BINDING",
                            "exact frozen request required",
                        )
                    if receipt is None or receipt["status"] != "VALID":
                        public, status = previous.subject.veto.unresolved_judge_output_v3(), "ENGINEERING_FAILURE"
                    else:
                        choice = receipt["raw_response"]["choices"][0]
                        require(
                            choice["finish_reason"] == "stop"
                            and choice["message"]["content"] == receipt["assistant_content"],
                            "SUBJECT_KINDS_RAW",
                            "complete original response required",
                        )
                        public, rule, _ = selected.parse_output(
                            arm, receipt["assistant_content"], row, manifest["specs"]
                        )
                        require(public == receipt["public"], "SUBJECT_KINDS_REPLAY", "exact adapter replay")
                        status = "VALID"
                error = reference.classify_primary_error(row["draft_label"], public)
                pairs.append(
                    {
                        "review_id": row["review_id"],
                        "canonical_pair_id": row["canonical_pair_id"],
                        "primary": reference.primary(public),
                        "draft_error": error,
                        "status": status,
                        "rule": rule,
                        "requires_cause_review": error != "CORRECT" or status == "ENGINEERING_FAILURE",
                    }
                )
                predictions.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        **public,
                        **({"metric_only_missing_output": True} if status == "ENGINEERING_FAILURE" else {}),
                    }
                )
            result["cells"][f"{rep}/{arm}"] = {
                "pairs": pairs,
                "engineering_failures": sum(p["status"] == "ENGINEERING_FAILURE" for p in pairs),
                "scores": {
                    n: reference.score([r[f] for r in rows], predictions)
                    for n, f in (("historical", "historical_label"), ("partial_draft", "draft_label"))
                },
            }
    guards = {}
    for rep in range(1, manifest["repeat_count"] + 1):
        guards[str(rep)] = []
        for row, observed in zip(rows, result["cells"][f"{rep}/candidate"]["pairs"], strict=True):
            if row["review_id"] not in PANEL:
                continue
            target = (
                {"same_duplicate_group": "NO", "a_can_replace_b": "NO", "b_can_replace_a": "NO"}
                if row["review_id"] in REPAIRS
                else reference.primary(row["main_public"])
            )
            if observed["primary"] != target or observed["status"] == "ENGINEERING_FAILURE":
                guards[str(rep)].append(row["review_id"])
    result["clear_scope_guard_failures"] = guards
    result["candidate_repeat_disagreements"] = (
        [
            a["review_id"]
            for a, b in zip(
                result["cells"]["1/candidate"]["pairs"], result["cells"]["2/candidate"]["pairs"], strict=True
            )
            if a["primary"] != b["primary"] or a["status"] != b["status"]
        ]
        if manifest["repeat_count"] == 2
        else []
    )
    result["named_target_pilot_gate"] = (
        len(rows) == 32
        and manifest["repeat_count"] == 2
        and not result["candidate_repeat_disagreements"]
        and not any(guards.values())
    )
    metrics = result["cells"]["1/candidate"]["scores"]["partial_draft"]["weighted"]
    result["full_development_precision_recall_75"] = (
        len(rows) == 1000
        and all(metrics[k] is not None and metrics[k] >= 0.75 for k in ("duplicate_precision", "duplicate_recall"))
        and result["cells"]["1/candidate"]["engineering_failures"] == 0
    )
    result["response_artifacts"] = {str(f): sha256_file(f) for f in (root / "responses").glob("*.json")}
    write_json_atomic(root / "assessment.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    reference.verify_freeze(root / "manifest.json")
    require(
        not (root / "started.json").exists()
        and time.time() < selected.base.DEADLINE - selected.base.STOP_ADMISSION_MARGIN,
        "SUBJECT_KINDS_START",
        "new run before deadline required",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {r["canonical_pair_id"]: r for r in json.loads((root / "panel_private.json").read_text())}
    requests = json.loads((root / "requests_frozen.json").read_text())
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SUBJECT_KINDS_CREDENTIAL", "existing credential source required")
    write_json_atomic(
        root / "started.json", {"at_utc": datetime.now(UTC).isoformat(), "deadline_utc_epoch": selected.base.DEADLINE}
    )
    profile = selected.base.previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=2 * len(requests),
    )
    with previous.paced_relay_v2.PacedRelayV2(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=selected.base.previous.GENERATION,
    ) as relay:
        relay.set_context(
            selected.base.previous.RelayContext(manifest["contract_version"], 0, root / "transport_events.jsonl")
        )
        for rep in range(1, manifest["repeat_count"] + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        selected.collect,
                        req,
                        rows[req["canonical_pair_id"]],
                        endpoint=relay.endpoint,
                        output=root / "responses" / f"{rep}-{req['arm']}-{req['canonical_pair_id']}.json",
                        specs=manifest["specs"],
                    )
                    for req in requests
                    if req["repeat"] == rep
                ]
                for future in as_completed(futures):
                    value = future.result()
                    print(json.dumps({k: value[k] for k in ("repeat", "arm", "status")}), flush=True)
    result = assess(root)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "external_attempts": sum(e.get("external_request") is True for e in events),
            "local_rejections": sum(e.get("external_request") is False for e in events),
            "http_statuses": dict(
                Counter(str(e.get("http_status")) for e in events if e.get("external_request") is True)
            ),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.parent is not None, "SUBJECT_KINDS_PARENT", "reviewed parent required")
        result = prepare(args.root, args.parent, full=args.full)
    else:
        require(args.env_file is not None, "SUBJECT_KINDS_ENV", "existing credential source required")
        result = run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cells", "sources", "artifacts", "response_artifacts")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
