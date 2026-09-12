# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Saved-coverage ablations isolating an optional subject-binding specialist."""

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

from eval.dedup.analysis import critic_paired_trial as paired
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.judging import critic_subject_binding as subject
from eval.dedup.judging import paced_relay_v2
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

SESSION = paired.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z"
SYSTEM = selected.base.previous.RESOURCES / "v06212_subject_binding_v1_system.txt"
SPEC = {"adapter": "eval.dedup.judging.critic_subject_binding"}


def prepare(root: Path, predecessor: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "SUBJECT_TRIAL_ROOT", "new root required")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    assessment = json.loads((predecessor / "assessment.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    errors = {r["review_id"] for c in assessment["cells"].values() for r in c["pairs"] if r["requires_cause_review"]}
    require(
        review.get("all_observed_disagreements_reviewed") is True
        and review.get("assessment_sha256") == sha256_file(predecessor / "assessment.json")
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == "V4_FIXED_OUTPUT_SUBJECT_BINDING_SPECIALIST_ONLY",
        "SUBJECT_TRIAL_REVIEW",
        "completed source-order review must authorize this isolated subject task",
    )
    old = json.loads((predecessor / "manifest.json").read_text())
    rows = json.loads((predecessor / "panel_private.json").read_text())
    require(len(rows) == 96 and old["repeat_count"] == 2, "SUBJECT_TRIAL_PANEL", "same reviewed panel required")
    receipts = {}
    for path in (predecessor / "responses").glob("*.json"):
        value = json.loads(path.read_text())
        if value["arm"] == "encoding_control":
            require(value["status"] == "VALID", "SUBJECT_TRIAL_BASE", "no missing base results")
            receipts[(value["repeat"], value["canonical_pair_id"])] = value
            sources[str(path)] = sha256_file(path)
    for row in rows:
        row["coverage_base"] = {}
        for repeat in (1, 2):
            receipt = receipts.get((repeat, row["canonical_pair_id"]))
            if receipt is None:
                require(
                    selected.base.critic.veto.route(row["main_public"], row["payload"])
                    != "REVIEW_POSITIVE_DIRECTIONS_ONLY",
                    "SUBJECT_TRIAL_BASE",
                    "only deterministic main bypass may lack a coverage receipt",
                )
                public = row["main_public"]
            else:
                public, _, _ = selected.parse_output(
                    "encoding_control", receipt["assistant_content"], row, old["specs"]
                )
                require(public == receipt["public"], "SUBJECT_TRIAL_REPLAY", "base must replay exactly")
            row["coverage_base"][str(repeat)] = deepcopy(public)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests = []
    for repeat in (1, 2):
        for row in rows:
            if subject.route(row["coverage_base"][str(repeat)], row["payload"]) != "REVIEW_BILATERAL_SUBJECTS":
                continue
            body = {
                "model": "Qwen/Qwen3.8-27B-FP8",
                **selected.base.previous.GENERATION,
                "messages": subject.messages(row["payload"], SYSTEM.read_text()),
                "response_format": transport.response_format(
                    subject.response_schema(row["payload"]), old["transport_projection"]
                ),
            }
            tokens = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(tokens + 6144 <= 32768, "SUBJECT_TRIAL_CONTEXT", "original visible spans must fit unchanged")
            requests.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    "repeat": repeat,
                    "arm": "candidate",
                    "body": body,
                    "request_sha256": sha256_json(body),
                    "input_tokens": tokens,
                    "deadline_utc_epoch": selected.base.DEADLINE,
                }
            )
    for path in (
        Path(__file__).resolve(),
        Path(subject.__file__).resolve(),
        Path(paced_relay_v2.__file__).resolve(),
        SYSTEM,
        Path(__file__).with_name("critic_subject_binding_v1.md"),
        predecessor / "assessment.json",
        predecessor / "review_complete.json",
        predecessor / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests)
    result = {
        "contract_version": subject.CONTRACT + "-saved-base96",
        "stage": "subject_saved_base96",
        "population": len(rows),
        "repeat_count": 2,
        "model": old["model"],
        "endpoint": old["endpoint"],
        "specs": {"candidate": SPEC},
        "main_online_calls": 0,
        "reference_changed": False,
        "coverage_online_calls": 0,
        "logical_requests": len(requests),
        "predecessor": str(predecessor),
        "deadline_utc_epoch": selected.base.DEADLINE,
        "transport_contract": paced_relay_v2.TRANSPORT_CONTRACT,
        "sources": sources,
        "artifacts": {str(root / n): sha256_file(root / n) for n in ("panel_private.json", "requests_frozen.json")},
    }
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def assess(root: Path) -> dict:
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    requested = {(r["repeat"], r["canonical_pair_id"]): r for r in requests}
    receipts = [json.loads(p.read_text()) for p in (root / "responses").glob("*.json")]
    found = {(r["repeat"], r["canonical_pair_id"]): r for r in receipts}
    require(
        len(found) == len(receipts) and found.keys() <= requested.keys(),
        "SUBJECT_TRIAL_RECEIPTS",
        "unique known receipts",
    )
    for key, receipt in found.items():
        require(
            receipt["request_sha256"] == requested[key]["request_sha256"],
            "SUBJECT_TRIAL_BINDING",
            "frozen request binding",
        )
    result = {
        "population": len(rows),
        "stage": manifest["stage"],
        "main_online_calls": 0,
        "reference_changed": False,
        "reference_status": selected.base.previous.reference.DRAFT_STATUS,
        "release_eligible": False,
        "full_development_75_gate": False,
        "cells": {},
    }
    for repeat in range(1, manifest["repeat_count"] + 1):
        for arm in ("saved_coverage", "candidate"):
            predictions, diagnostics = [], []
            for row in rows:
                base = row["coverage_base"][str(repeat)]
                public, rule, status = base, subject.route(base, row["payload"]), "DETERMINISTIC_BYPASS"
                receipt = found.get((repeat, row["canonical_pair_id"]))
                if arm == "saved_coverage":
                    status, rule = "FIXED_SAVED_COVERAGE", "EXACT_SAME_BASE_BOTH_ARMS"
                elif rule == "REVIEW_BILATERAL_SUBJECTS":
                    if receipt is None or receipt["status"] != "VALID":
                        public, status = selected.base.critic.veto.unresolved_judge_output_v3(), "ENGINEERING_FAILURE"
                    else:
                        choice = receipt["raw_response"]["choices"][0]
                        require(
                            choice["finish_reason"] == "stop"
                            and choice["message"]["content"] == receipt["assistant_content"],
                            "SUBJECT_TRIAL_RAW",
                            "complete original response required",
                        )
                        public, rule = subject.apply_review(
                            base, row["payload"], selected.base.previous.strict_json(receipt["assistant_content"])
                        )
                        require(
                            public == receipt["public"], "SUBJECT_TRIAL_REPLAY", "subject output must replay exactly"
                        )
                        status = "VALID"
                error = selected.base.previous.reference.classify_primary_error(row["draft_label"], public)
                predictions.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        **public,
                        **({"metric_only_missing_output": True} if status == "ENGINEERING_FAILURE" else {}),
                    }
                )
                diagnostics.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "review_id": row["review_id"],
                        "primary": selected.base.previous.reference.primary(public),
                        "draft_error": error,
                        "status": status,
                        "rule": rule,
                        "requires_cause_review": error != "CORRECT" or status == "ENGINEERING_FAILURE",
                    }
                )
            result["cells"][f"{repeat}/{arm}"] = {
                "scores": {
                    name: selected.base.previous.reference.score([r[field] for r in rows], predictions)
                    for name, field in (("historical", "historical_label"), ("partial_draft", "draft_label"))
                },
                "pairs": diagnostics,
                "engineering_failures": sum(p["status"] == "ENGINEERING_FAILURE" for p in diagnostics),
            }
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
    result["new_errors_from_previously_correct"] = {
        str(rep): [
            a["review_id"]
            for a, b in zip(
                result["cells"][f"{rep}/saved_coverage"]["pairs"],
                result["cells"][f"{rep}/candidate"]["pairs"],
                strict=True,
            )
            if a["draft_error"] == "CORRECT" and b["draft_error"] != "CORRECT"
        ]
        for rep in range(1, manifest["repeat_count"] + 1)
    }
    result["response_artifacts"] = {str(p): sha256_file(p) for p in (root / "responses").rglob("*.json")}
    write_json_atomic(root / "assessment.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    require(
        not (root / "started.json").exists()
        and time.time() < selected.base.DEADLINE - selected.base.STOP_ADMISSION_MARGIN,
        "SUBJECT_TRIAL_START",
        "new run before deadline required",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    rows = selected.base.previous._index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen = json.loads((root / "requests_frozen.json").read_text())
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SUBJECT_TRIAL_CREDENTIAL", "existing credential required")
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
        max_external_attempts=2 * len(frozen),
    )
    with paced_relay_v2.PacedRelayV2(
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
        for repeat in range(1, manifest["repeat_count"] + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = []
                for request in frozen:
                    if request["repeat"] != repeat:
                        continue
                    row = deepcopy(rows[request["canonical_pair_id"]])
                    row["main_public"] = row["coverage_base"][str(repeat)]
                    futures.append(
                        pool.submit(
                            selected.collect,
                            request,
                            row,
                            endpoint=relay.endpoint,
                            output=root / "responses" / f"{repeat}-candidate-{row['canonical_pair_id']}.json",
                            specs=manifest["specs"],
                        )
                    )
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
    result = assess(root)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "contract_digest": manifest["contract_digest"],
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
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, default=SESSION / "retention-v6-sourceorder96")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.predecessor)
    elif args.command == "run":
        require(args.env_file is not None, "SUBJECT_ENV", "explicit existing credential source required")
        result = run(args.root, args.env_file)
    else:
        result = assess(args.root)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("sources", "artifacts", "cells", "response_artifacts")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
