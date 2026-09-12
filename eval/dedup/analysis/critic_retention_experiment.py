# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Deadline-bounded, frozen critic comparisons with source-oriented proofs."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml
from jinja2 import Environment, StrictUndefined

from eval.dedup.analysis import critic_scope_experiment as previous
from eval.dedup.judging import critic_retention_v2 as critic
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

HERE = Path(__file__).resolve().parent
PRIOR = Path("/raid/hfang/ihb/runs/v06212-critic-scope-v1-fixed-execution")
DEADLINE = 1789128052  # New user-requested window ends 2026-09-11 12:00:52 UTC.
STOP_ADMISSION_MARGIN = 210
DISPUTES = {
    "H0093": "Complete named data-processing consent plus product versus old interface-only negative.",
    "H0195": "Sparse product/archive headings: actual incompatible roles or harmless surrounding UI?",
    "H0352": "Credit-disclosure completeness versus short ancillary notice.",
    "H0748": "Stories/byline fragment: substantive independent content or non-main chrome?",
}
NEGATIVES = ("H0537", "H0514", "H0850", "H0316", "H0701", "H0760")
POSITIVES = ("H0108", "H0126", "H0419", "H0998")
EQUIVALENT = ("H0007", "H0893", "H0347", "H0417", "H0810")


def renderers() -> dict:
    render = previous.renderers()
    config = yaml.safe_load((previous.RESOURCES / "v06212_retention_v2.yaml").read_text())
    env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701 - plain text, not HTML
    old_pair = env.from_string((previous.RESOURCES / "v06212_critic_scope_v1_pair.jinja").read_text())
    old_config = yaml.safe_load(previous.CONFIG.read_text())
    pair = env.from_string((previous.RESOURCES / config["prompt_path"]).read_text())
    system = (previous.RESOURCES / config["system_prompt_path"]).read_text()

    def formatted(payload: dict, *, candidate: bool) -> list[dict]:
        previous.assert_blind_payload(payload)
        return [
            {"role": "system", "content": system if candidate else render["candidate"](payload)[0]["content"]},
            {
                "role": "user",
                "content": (pair if candidate else old_pair).render(
                    payload=payload,
                    rubric=json.dumps((config if candidate else old_config)["rubric"], ensure_ascii=False),
                    schema=json.dumps(critic.response_schema() if candidate else critic.format_only_schema()),
                ),
            },
        ]

    return {
        "control": render["control"],
        "format_only": lambda p: formatted(p, candidate=False),
        "candidate": lambda p: formatted(p, candidate=True),
    }


def parse_output(arm: str, text: str, row: dict) -> tuple[dict, str, dict]:
    if arm == "control":
        return previous.parse_output(arm, text, row)
    require(arm in ("format_only", "candidate"), "RETENTION_ARM", "unknown experiment arm")
    value = previous.strict_json(text)
    public, rule = critic.apply_review(row["main_public"], row["payload"], value, format_only=arm == "format_only")
    return public, rule, value


def offline_format_diagnosis() -> dict:
    rows = previous._index(json.loads((PRIOR / "panel_private.json").read_text()), "prior panel")
    cases = []
    for path in sorted((PRIOR / "responses").glob("*-candidate-*.json")):
        receipt = json.loads(path.read_text())
        row = rows[receipt["canonical_pair_id"]]
        record = {
            "review_id": row["review_id"],
            "repeat": receipt["repeat"],
            "source_response_sha256": sha256_file(path),
            "original_status": receipt["status"],
            "original_error_code": receipt.get("error_code"),
        }
        value = previous.strict_json(receipt["assistant_content"])
        record["original_version_value"] = value.pop("contract_version", None)
        try:
            critic.apply_review(row["main_public"], row["payload"], value, format_only=True)
            record["counterfactual_status"] = "VALID"
        except DedupEvaluationError as exc:
            record.update(counterfactual_status="INVALID", remaining_error_code=exc.issue.code)
        cases.append(record)
    return {
        "interpretation": "Metadata-only counterfactual, not fresh model outputs; no labels, scores or original statuses changed.",
        "new_model_calls": 0,
        "scored": False,
        "cases": cases,
    }


def prepare(root: Path, *, stage: str = "pilot", parent: Path | None = None) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root = root.resolve()
    require(not root.exists(), "RETENTION_ROOT_EXISTS", "never overwrite a frozen or observed experiment")
    require(stage in ("pilot", "panel", "full"), "RETENTION_STAGE", "unknown evaluation stage")
    sources = previous.reference.verify_freeze(PRIOR / "manifest.json")
    rows = json.loads((PRIOR / "panel_private.json").read_text())
    if stage != "pilot":
        require(parent is not None, "RETENTION_PARENT", "expansion needs a completed reviewed predecessor")
        previous.reference.verify_freeze(parent / "manifest.json")
        assessment = json.loads((parent / "assessment.json").read_text())
        completion = json.loads((parent / "complete.json").read_text())
        require(
            completion["assessment_sha256"] == sha256_file(parent / "assessment.json")
            and assessment["next_step"] == "REVIEW_BEFORE_EXPANSION",
            "RETENTION_EXPANSION_GATE",
            "a failed or incomplete predecessor cannot be expanded",
        )
        review = json.loads((parent / "review_complete.json").read_text())
        require(
            review.get("assessment_sha256") == sha256_file(parent / "assessment.json")
            and review.get("next_stage") == stage
            and review.get("all_observed_disagreements_reviewed") is True,
            "RETENTION_REVIEW_GATE",
            "review must bind the exact assessment and proposed expansion",
        )
        sources.update({str(p): sha256_file(p) for p in (parent / "assessment.json", parent / "review_complete.json")})
    if stage == "pilot":
        rows = [r for r in rows if r["in_pilot"]]
    if stage == "full":
        mains, finals, packets, _ = previous.historical.component_replay(
            previous.historical.FULL_ROOT, "v6", main_policy="v6-route"
        )
        raw, raw_sources = previous.matched_raw_outputs(previous.historical.FULL_ROOT, finals)
        sources.update(raw_sources)
        main = previous._index(mains, "mains")
        final = previous._index(finals, "finals")
        payloads = previous._index(packets, "payloads")
        old = previous._index(
            previous.historical._read_csv(previous.REFERENCE_ROOT / "reference_historical.csv"), "historical"
        )
        draft = previous.historical._read_csv(previous.REFERENCE_ROOT / "reference_policy_v2_partial_draft.csv")
        inventory = previous._index(rows, "prior inventory")
        rows = []
        for label in draft:
            pid = label["canonical_pair_id"]
            rows.append(
                {
                    **inventory.get(pid, {}),
                    "canonical_pair_id": pid,
                    "review_id": label["review_id"],
                    "payload": payloads[pid]["payload"],
                    "raw_main": raw[pid][0],
                    "main_public": {k: main[pid][k] for k in previous.JUDGE_FIELDS_V3},
                    "saved_final": {k: final[pid][k] for k in previous.JUDGE_FIELDS_V3},
                    "draft_label": label,
                    "historical_label": old[pid],
                }
            )
    rows.sort(key=lambda r: r["review_id"])
    require(len(rows) == {"pilot": 19, "panel": 96, "full": 1000}[stage], "RETENTION_POPULATION", "full denominator")
    arms = ("control", "format_only", "candidate") if stage == "pilot" else ("control", "candidate")
    repeats = 1 if stage == "full" else 2
    render = renderers()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests_frozen = []
    for repeat in range(1, repeats + 1):
        for row in rows:
            if critic.veto.route(row["main_public"], row["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
                continue
            for arm in arms if repeat % 2 else tuple(reversed(arms)):
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **previous.GENERATION,
                    "messages": render[arm](row["payload"]),
                    "response_format": {"type": "json_object"},
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "RETENTION_CONTEXT", "never truncate a frozen input to fit")
                requests_frozen.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": repeat,
                        "arm": arm,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                        "deadline_utc_epoch": DEADLINE,
                    }
                )
    for path in (
        Path(__file__),
        Path(critic.__file__),
        HERE / "critic_five_hour_v1.md",
        previous.RESOURCES / "v06212_retention_v2.yaml",
        previous.RESOURCES / "v06212_retention_v2_system.jinja",
        previous.RESOURCES / "v06212_retention_v2_pair.jinja",
    ):
        sources[str(path.resolve())] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests_frozen)
    write_json_atomic(root / "offline_format_diagnosis.json", offline_format_diagnosis())
    manifest = {
        "contract_version": critic.CONTRACT,
        "stage": stage,
        "deadline_utc_epoch": DEADLINE,
        "stop_admission_margin_seconds": STOP_ADMISSION_MARGIN,
        "model": "nvidia/qwen/qwen3.8-27b",
        "endpoint": "https://inference-api.nvidia.com/v1",
        "arms": list(arms),
        "repeat_count": repeats,
        "population": len(rows),
        "logical_requests": len(requests_frozen),
        "fixed_main": "v0.6.2.12",
        "main_online_calls": 0,
        "reference_changed": False,
        "pending_adjudication": DISPUTES,
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir() if p.is_file()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def collect(request: dict, row: dict, *, endpoint: str, output: Path) -> dict:
    require(sha256_json(request["body"]) == request["request_sha256"], "RETENTION_REQUEST", "frozen body changed")
    receipt = {k: request[k] for k in ("canonical_pair_id", "repeat", "arm", "request_sha256")}
    receipt.update(status="REQUEST_STARTED", contract_version=critic.CONTRACT)
    write_json_atomic(output.parent / "request_started" / output.name, receipt)
    try:
        if time.time() >= request["deadline_utc_epoch"] - STOP_ADMISSION_MARGIN:
            receipt.update(status="DEADLINE_NOT_SUBMITTED", error_code="FIVE_HOUR_DEADLINE")
        else:
            response = requests.post(endpoint + "/chat/completions", json=request["body"], timeout=200)
            receipt["http_status"] = response.status_code
            if response.status_code != 200:
                receipt.update(status="TRANSPORT_FAILURE", error_code=f"HTTP_{response.status_code}")
            else:
                raw = response.json()
                receipt.update(raw_response=raw, status="RECEIVED_UNVALIDATED")
                write_json_atomic(output.parent / "raw_received" / output.name, receipt)
                choice = raw["choices"][0]
                require(choice.get("finish_reason") == "stop", "RETENTION_FINISH", "truncated completion is invalid")
                receipt["assistant_content"] = choice["message"]["content"]
                public, rule, value = parse_output(request["arm"], receipt["assistant_content"], row)
                receipt.update(status="VALID", public=public, rule=rule, parsed_review=value)
    except DedupEvaluationError as exc:
        receipt.update(status="VALIDATION_FAILURE", error_code=exc.issue.code)
    except Exception as exc:  # noqa: BLE001 - never store credential-bearing exception messages
        receipt.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_code=type(exc).__name__)
    finally:
        write_json_atomic(output, receipt)
    return receipt


def assess(root: Path) -> dict:
    previous.reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    requests_frozen = json.loads((root / "requests_frozen.json").read_text())
    request_map = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in requests_frozen}
    receipts = [json.loads(p.read_text()) for p in sorted((root / "responses").glob("*.json"))]
    found = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in receipts}
    require(
        len(found) == len(receipts) and found.keys() <= request_map.keys(),
        "RETENTION_RECEIPTS",
        "unique bound responses",
    )
    for key, response in found.items():
        require(
            response["request_sha256"] == request_map[key]["request_sha256"], "RETENTION_BINDING", "request binding"
        )
        if response["status"] == "VALID":
            choice = response["raw_response"]["choices"][0]
            require(
                choice.get("finish_reason") == "stop"
                and choice["message"]["content"] == response["assistant_content"],
                "RETENTION_RAW_BINDING",
                "saved accepted text must equal raw assistant text",
            )
    result = {
        "contract_digest": manifest["contract_digest"],
        "stage": manifest["stage"],
        "population": len(rows),
        "release_eligible": False,
        "reference_status": previous.reference.DRAFT_STATUS,
        "cells": {},
        "main_online_calls": 0,
        "responses_saved": len(receipts),
        "logical_requests": len(request_map),
    }
    for repeat in range(1, manifest["repeat_count"] + 1):
        for arm in manifest["arms"]:
            predictions, diagnostics = [], []
            for row in rows:
                pid, rid = row["canonical_pair_id"], row["review_id"]
                rule = critic.veto.route(row["main_public"], row["payload"])
                receipt = found.get((repeat, arm, pid))
                if rule != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
                    public, status = row["main_public"], "DETERMINISTIC_BYPASS"
                elif receipt is None or receipt["status"] != "VALID":
                    public = critic.veto.unresolved_judge_output_v3()
                    status, rule = "ENGINEERING_FAILURE", "MISSING_OR_INVALID_OUTPUT"
                else:
                    public, rule, _ = parse_output(arm, receipt["assistant_content"], row)
                    require(public == receipt["public"], "RETENTION_REPLAY", "accepted result must replay exactly")
                    status = "VALID"
                predictions.append(
                    {
                        "canonical_pair_id": pid,
                        **public,
                        **({"metric_only_missing_output": True} if status == "ENGINEERING_FAILURE" else {}),
                    }
                )
                error = previous.reference.classify_primary_error(row["draft_label"], public)
                diagnostics.append(
                    {
                        "canonical_pair_id": pid,
                        "review_id": rid,
                        "primary": previous.reference.primary(public),
                        "draft_error": error,
                        "status": status,
                        "rule": rule,
                        "error_code": (receipt or {}).get("error_code"),
                        "review_state": "REFERENCE_DISPUTE"
                        if rid in DISPUTES
                        else "ENGINEERING_WITH_CAUSE_REVIEW_REQUIRED"
                        if status == "ENGINEERING_FAILURE"
                        else "SEMANTIC_REVIEW_REQUIRED"
                        if error != "CORRECT"
                        else "REFERENCE_AGREEMENT_NOT_NEW_GOLD",
                        "dispute": DISPUTES.get(rid),
                    }
                )
            result["cells"][f"{repeat}/{arm}"] = {
                "scores": {
                    name: previous.reference.score([r[field] for r in rows], predictions)
                    for name, field in (("historical", "historical_label"), ("partial_draft", "draft_label"))
                },
                "pairs": diagnostics,
                "engineering_failures": sum(d["status"] == "ENGINEERING_FAILURE" for d in diagnostics),
                "guard_failures": [
                    d["review_id"]
                    for d in diagnostics
                    if (d["review_id"] in NEGATIVES and d["primary"]["same_duplicate_group"] != "NO")
                    or (d["review_id"] in (*POSITIVES, *EQUIVALENT) and d["draft_error"] != "CORRECT")
                ],
            }
    candidate_cells = [result["cells"][f"{i}/candidate"] for i in range(1, manifest["repeat_count"] + 1)]
    differences = []
    if len(candidate_cells) == 2:
        differences = [
            a["review_id"]
            for a, b in zip(candidate_cells[0]["pairs"], candidate_cells[1]["pairs"], strict=True)
            if a["primary"] != b["primary"] or a["status"] != b["status"]
        ]
    result["candidate_repeat_disagreements"] = differences
    worse = any(
        cell["scores"]["partial_draft"]["weighted"]["primary_decision_exact"]
        < result["cells"][f"{i}/control"]["scores"]["partial_draft"]["weighted"]["primary_decision_exact"]
        for i, cell in enumerate(candidate_cells, 1)
    )
    result["next_step"] = (
        "STOP_REVIEW_AND_REPAIR"
        if differences or worse or any(c["engineering_failures"] or c["guard_failures"] for c in candidate_cells)
        else "REVIEW_BEFORE_EXPANSION"
    )
    full_score = candidate_cells[0]["scores"]["partial_draft"]["weighted"]
    result["full_development_75_gate"] = (
        manifest["stage"] == "full"
        and result["next_step"] == "REVIEW_BEFORE_EXPANSION"
        and full_score["duplicate_precision"] >= 0.75
        and full_score["duplicate_recall"] >= 0.75
        and full_score["primary_decision_exact"] >= 0.79
    )
    result["response_artifacts"] = {str(p): sha256_file(p) for p in (root / "responses").rglob("*.json")}
    write_json_atomic(root / "assessment.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    previous.reference.verify_freeze(root / "manifest.json")
    require(not (root / "started.json").exists(), "RETENTION_STARTED", "observed experiments cannot restart")
    require(time.time() < DEADLINE - STOP_ADMISSION_MARGIN, "RETENTION_DEADLINE", "no new calls near hard deadline")
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "RETENTION_CREDENTIAL", "existing credential unavailable")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = previous._index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen = json.loads((root / "requests_frozen.json").read_text())
    write_json_atomic(root / "started.json", {"at_utc": datetime.now(UTC).isoformat(), "deadline_utc_epoch": DEADLINE})
    profile = previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=2 * len(frozen),
    )
    with previous.PacedRelay(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=previous.GENERATION,
    ) as relay:
        relay.set_context(previous.RelayContext("critic-retention-v2", 0, root / "transport_events.jsonl"))
        for repeat in range(1, manifest["repeat_count"] + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        collect,
                        request,
                        rows[request["canonical_pair_id"]],
                        endpoint=relay.endpoint,
                        output=root / "responses" / f"{repeat}-{request['arm']}-{request['canonical_pair_id']}.json",
                    )
                    for request in frozen
                    if request["repeat"] == repeat
                ]
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
            previous.reference.verify_freeze(root / "manifest.json")
    result = assess(root)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "contract_digest": manifest["contract_digest"],
            "external_attempts": len(events),
            "http_statuses": dict(Counter(str(e["upstream_http_status"]) for e in events)),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "panel", "full"), default="pilot")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, stage=args.stage, parent=args.parent)
    elif args.command == "run":
        require(args.env_file is not None, "RETENTION_CREDENTIAL", "explicit existing env file required")
        result = run(args.root, args.env_file)
    else:
        result = assess(args.root)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("sources", "artifacts", "response_artifacts", "cells")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
