# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Conditional proof-verifier trials with unchanged upstream subject proposals."""

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
from eval.dedup.analysis import critic_subject_kinds_experiment as kinds_trial
from eval.dedup.judging import critic_subject_proof_verifier as candidate
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

selected = previous.selected
reference = selected.base.previous.reference
SYSTEM = previous.SYSTEM.with_name("v06212_subject_proof_verifier_v4_system.txt")


def prepare(root: Path, parent: Path, *, full: bool = False) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "SUBJECT_PROOF_ROOT", "new run root required")
    sources = kinds_trial.reviewed_sources(
        parent, "PROOF_BOUND_VERIFIER_FULL1000" if full else "PROOF_BOUND_VERIFIER_PILOT"
    )
    if full:
        require(
            json.loads((parent / "assessment.json").read_text())["proof_verifier_pilot_gate"],
            "SUBJECT_PROOF_GATE",
            "all pilot protections and repeats must pass",
        )
    origin = Path(json.loads((parent / "manifest.json").read_text())["coverage_origin"])
    sources.update(reference.verify_freeze(origin / "manifest.json"))
    old = json.loads((origin / "manifest.json").read_text())
    rows = json.loads((origin / "specialist/panel_private.json").read_text())
    require(
        json.loads((origin / "coverage/assessment.json").read_text())["cells"]["1/candidate"]["engineering_failures"]
        == 0,
        "SUBJECT_PROOF_BASE",
        "missing upstream coverage must not be hidden",
    )
    receipts = {}
    for path in (origin / "specialist/responses").glob("*.json"):
        value = json.loads(path.read_text())
        require(value["status"] == "VALID", "SUBJECT_PROOF_UPSTREAM", "all original specialist replies must be valid")
        receipts[value["canonical_pair_id"]] = value
        sources[str(path)] = sha256_file(path)
    rows = [r for r in rows if full or r["review_id"] in kinds_trial.PANEL]
    require(len(rows) == (1000 if full else 32), "SUBJECT_PROOF_MEMBERSHIP", "complete original declared rows")
    for row in rows:
        row["original_main_public"] = deepcopy(row["main_public"])
        row["main_public"] = deepcopy(row["coverage_base"]["1"])
        base, payload = row["main_public"], row["payload"]
        if previous.subject.route(base, payload) == "REVIEW_BILATERAL_SUBJECTS":
            receipt = receipts[row["canonical_pair_id"]]
            choice = receipt["raw_response"]["choices"][0]
            require(
                choice["finish_reason"] == "stop" and choice["message"]["content"] == receipt["assistant_content"],
                "SUBJECT_PROOF_RAW",
                "complete original subject response required",
            )
            proposal = selected.base.previous.strict_json(receipt["assistant_content"])
            require(
                previous.subject.apply_review(base, payload, proposal)[0] == receipt["public"],
                "SUBJECT_PROOF_UPSTREAM_REPLAY",
                "original proposal must replay exactly",
            )
            payload["subject_proposal"] = proposal
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests = []
    for rep in (1, 2):
        for row in rows:
            if candidate.route(row["main_public"], row["payload"]) != "VERIFY_FIXED_SUBJECT_VETO":
                continue
            body = {
                "model": "Qwen/Qwen3.8-27B-FP8",
                **selected.base.previous.GENERATION,
                "messages": candidate.messages(row["payload"], SYSTEM.read_text()),
                "response_format": previous.transport.response_format(
                    candidate.response_schema(), old["transport_projection"]
                ),
            }
            tokens = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(tokens + 6144 <= 32768, "SUBJECT_PROOF_CONTEXT", "no original text truncation")
            requests.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    "repeat": rep,
                    "arm": "candidate",
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
        Path(__file__).with_name("critic_subject_proof_verifier_v4.md"),
        origin / "specialist/panel_private.json",
        origin / "specialist/assessment.json",
        origin / "coverage/assessment.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests)
    result = {
        "contract_version": candidate.CONTRACT,
        "stage": "proof_verifier_full1000" if full else "proof_verifier_pilot32",
        "population": len(rows),
        "repeat_count": 2,
        "logical_requests": len(requests),
        "specs": {"candidate": {"adapter": candidate.__name__}},
        "coverage_origin": str(origin),
        "parent": str(parent),
        "model": old["model"],
        "endpoint": old["endpoint"],
        "main_online_calls": 0,
        "coverage_online_calls": 0,
        "subject_online_calls": 0,
        "reference_changed": False,
        "deadline_utc_epoch": selected.base.DEADLINE,
        "sources": sources,
        "artifacts": {str(root / n): sha256_file(root / n) for n in ("panel_private.json", "requests_frozen.json")},
    }
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def assess(root: Path) -> dict:
    reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    requests = {
        (r["repeat"], r["canonical_pair_id"]): r for r in json.loads((root / "requests_frozen.json").read_text())
    }
    receipts = [json.loads(f.read_text()) for f in (root / "responses").glob("*.json")]
    found = {(r["repeat"], r["canonical_pair_id"]): r for r in receipts}
    require(
        len(found) == len(receipts) and found.keys() <= requests.keys(),
        "SUBJECT_PROOF_RECEIPTS",
        "unique known replies",
    )
    result = {
        "population": len(rows),
        "stage": manifest["stage"],
        "reference_status": reference.DRAFT_STATUS,
        "reference_changed": False,
        "main_online_calls": 0,
        "release_eligible": False,
        "cells": {},
    }
    for rep in (1, 2):
        for arm in ("candidate", "saved_scope", "saved_coverage"):
            pairs, predictions = [], []
            for row in rows:
                base, payload = row["main_public"], row["payload"]
                public, status, rule = base, "DETERMINISTIC_BYPASS", "UNCHANGED_COVERAGE"
                if arm == "saved_scope":
                    public, rule = candidate.proposed_result(base, payload)
                    status = "FIXED_SAVED_SUBJECT_SCOPE"
                elif arm == "candidate" and candidate.route(base, payload) == "VERIFY_FIXED_SUBJECT_VETO":
                    key = (rep, row["canonical_pair_id"])
                    receipt = found.get(key)
                    if receipt is not None:
                        require(
                            receipt["arm"] == "candidate"
                            and receipt["request_sha256"] == requests[key]["request_sha256"],
                            "SUBJECT_PROOF_REQUEST",
                            "exact bound original request",
                        )
                    if receipt is None or receipt["status"] != "VALID":
                        public, status, rule = (
                            previous.subject.veto.unresolved_judge_output_v3(),
                            "ENGINEERING_FAILURE",
                            "MISSING_OR_INVALID_VERIFICATION",
                        )
                    else:
                        choice = receipt["raw_response"]["choices"][0]
                        require(
                            choice["finish_reason"] == "stop"
                            and choice["message"]["content"] == receipt["assistant_content"],
                            "SUBJECT_PROOF_RAW",
                            "exact complete original response",
                        )
                        public, rule, _ = selected.parse_output(
                            "candidate", receipt["assistant_content"], row, manifest["specs"]
                        )
                        require(public == receipt["public"], "SUBJECT_PROOF_REPLAY", "exact verifier replay")
                        status = "VALID"
                error = reference.classify_primary_error(row["draft_label"], public)
                pairs.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "review_id": row["review_id"],
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
    result["clear_scope_guard_failures"] = {}
    for rep in (1, 2):
        failed = []
        for row, observed in zip(rows, result["cells"][f"{rep}/candidate"]["pairs"], strict=True):
            if row["review_id"] not in kinds_trial.PANEL:
                continue
            expected = (
                {"same_duplicate_group": "NO", "a_can_replace_b": "NO", "b_can_replace_a": "NO"}
                if row["review_id"] in kinds_trial.REPAIRS
                else reference.primary(row["main_public"])
            )
            if observed["primary"] != expected or observed["status"] == "ENGINEERING_FAILURE":
                failed.append(row["review_id"])
        result["clear_scope_guard_failures"][str(rep)] = failed
    result["candidate_repeat_disagreements"] = [
        a["review_id"]
        for a, b in zip(result["cells"]["1/candidate"]["pairs"], result["cells"]["2/candidate"]["pairs"], strict=True)
        if a["primary"] != b["primary"] or a["status"] != b["status"]
    ]
    result["proof_verifier_pilot_gate"] = (
        len(rows) == 32
        and not any(result["clear_scope_guard_failures"].values())
        and not result["candidate_repeat_disagreements"]
    )
    result["full_development_precision_recall_75"] = len(rows) == 1000 and all(
        result["cells"][f"{rep}/candidate"]["engineering_failures"] == 0
        and all(
            result["cells"][f"{rep}/candidate"]["scores"]["partial_draft"]["weighted"][metric] is not None
            and result["cells"][f"{rep}/candidate"]["scores"]["partial_draft"]["weighted"][metric] >= 0.75
            for metric in ("duplicate_precision", "duplicate_recall")
        )
        for rep in (1, 2)
    )
    result["response_artifacts"] = {str(f): sha256_file(f) for f in (root / "responses").glob("*.json")}
    write_json_atomic(root / "assessment.json", result)
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
        require(args.parent is not None, "SUBJECT_PROOF_PARENT", "reviewed parent required")
        result = prepare(args.root, args.parent, full=args.full)
    else:
        require(args.env_file is not None, "SUBJECT_PROOF_ENV", "existing credential source required")
        result = run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cells", "sources", "artifacts", "response_artifacts")}
        )
    )
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
