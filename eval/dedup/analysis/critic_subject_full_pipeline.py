# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Predeclared full-development coverage plus narrow subject-action diagnostics."""

from __future__ import annotations

import argparse
import importlib
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_full_diagnostic as full
from eval.dedup.analysis import critic_resilient_execution as resilient
from eval.dedup.analysis import critic_subject_experiment as subject_trial
from eval.dedup.analysis import critic_subject_scope_experiment as scope_trial
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

selected = subject_trial.selected
scope = scope_trial.scope


def reviewed_parent(parent: Path) -> dict:
    sources = selected.base.previous.reference.verify_freeze(parent / "manifest.json")
    assessment = json.loads((parent / "assessment.json").read_text())
    review = json.loads((parent / "review_complete.json").read_text())
    errors = {p["review_id"] for c in assessment["cells"].values() for p in c["pairs"] if p["requires_cause_review"]}
    require(
        assessment["population"] == 96
        and review.get("assessment_sha256") == sha256_file(parent / "assessment.json")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == "FULL1000_FIXED_PIPELINE_DIAGNOSTIC_NOT_RELEASE",
        "SUBJECT_FULL_REVIEW",
        "complete reviewed local scope trial required",
    )
    require(
        assessment["candidate_repeat_disagreements"] == []
        and all(not v for v in assessment["new_errors_from_previously_correct"].values())
        and all(not v for v in assessment["clear_guard_failures"].values())
        and all(assessment["cells"][f"{rep}/candidate"]["engineering_failures"] == 0 for rep in (1, 2)),
        "SUBJECT_FULL_GUARDS",
        "do not expand with a repeat, protection or engineering failure",
    )
    for name in ("assessment.json", "review_complete.json", "complete.json"):
        sources[str(parent / name)] = sha256_file(parent / name)
    return sources


def freeze_manifest(root: Path, value: dict, names: tuple[str, ...]) -> dict:
    value["artifacts"] = {str(root / name): sha256_file(root / name) for name in names}
    value["contract_digest"] = sha256_json(value)
    write_json_atomic(root / "manifest.json", value)
    return value


def prepare(root: Path, parent: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "SUBJECT_FULL_ROOT", "new full pipeline root required")
    sources = reviewed_parent(parent)
    rows, more = full.original_rows()
    sources.update(more)
    anchor = subject_trial.SESSION / "retention-v6-sourceorder96"
    old = json.loads((anchor / "manifest.json").read_text())
    spec = old["specs"]["encoding_control"]
    require(
        spec["adapter"] == "eval.dedup.judging.critic_retention_v4",
        "SUBJECT_FULL_BASE",
        "exact reviewed v4 coverage contract required",
    )
    module = importlib.import_module(spec["adapter"])
    render = {"control": selected.base.previous.renderers()["control"], "candidate": selected.renderer(spec)}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests, potential = [], []
    for row in rows:
        if selected.base.critic.veto.route(row["main_public"], row["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            continue
        bodies = [
            (
                arm,
                {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **selected.base.previous.GENERATION,
                    "messages": renderer(row["payload"]),
                    "response_format": subject_trial.transport.response_format(
                        module.response_schema(), old["transport_projection"]
                    )
                    if arm == "candidate"
                    else {"type": "json_object"},
                },
            )
            for arm, renderer in render.items()
        ]
        if subject_trial.subject.route(row["main_public"], row["payload"]) == "REVIEW_BILATERAL_SUBJECTS":
            bodies.append(
                (
                    "specialist",
                    {
                        "model": "Qwen/Qwen3.8-27B-FP8",
                        **selected.base.previous.GENERATION,
                        "messages": subject_trial.subject.messages(row["payload"], subject_trial.SYSTEM.read_text()),
                        "response_format": subject_trial.transport.response_format(
                            subject_trial.subject.response_schema(row["payload"]), old["transport_projection"]
                        ),
                    },
                )
            )
        for arm, body in bodies:
            tokens = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(
                tokens + 6144 <= 32768, "SUBJECT_FULL_CONTEXT", "all original spans must fit without input truncation"
            )
            request = {
                "canonical_pair_id": row["canonical_pair_id"],
                "repeat": 1,
                "arm": "candidate" if arm == "specialist" else arm,
                "body": body,
                "request_sha256": sha256_json(body),
                "input_tokens": tokens,
                "deadline_utc_epoch": selected.base.DEADLINE,
            }
            (potential if arm == "specialist" else requests).append(request)
    require(
        len(requests) == 500, "SUBJECT_FULL_REQUESTS", "250 original positive non-exact cases, two fresh coverage arms"
    )
    previous = json.loads((anchor / "requests_frozen.json").read_text())
    by = {(r["arm"], r["canonical_pair_id"]): r for r in requests}
    require(
        all(
            by[("candidate" if r["arm"] == "encoding_control" else "control", r["canonical_pair_id"])][
                "request_sha256"
            ]
            == r["request_sha256"]
            for r in previous
            if r["repeat"] == 1 and r["arm"] in ("control", "encoding_control")
        ),
        "SUBJECT_FULL_IDENTITY",
        "exact local control bodies must be preserved; no local outputs reused",
    )
    for path in (
        Path(__file__).resolve(),
        Path(full.__file__).resolve(),
        Path(resilient.__file__).resolve(),
        Path(scope_trial.__file__).resolve(),
        Path(subject_trial.__file__).resolve(),
        Path(scope.__file__).resolve(),
        subject_trial.SYSTEM,
        Path(__file__).with_name("critic_subject_full_pipeline_v1.md"),
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "potential_subject_requests.json", potential)
    coverage = root / "coverage"
    write_json_atomic(coverage / "panel_private.json", rows)
    write_json_atomic(coverage / "requests_frozen.json", requests)
    cm = {
        "contract_version": "dedup-v06212-v4-coverage-full-component",
        "stage": "full_coverage_component",
        "population": 1000,
        "repeat_count": 1,
        "arms": ["control", "candidate"],
        "specs": {"candidate": spec},
        "model": old["model"],
        "endpoint": old["endpoint"],
        "transport_contract": subject_trial.paced_relay_v2.TRANSPORT_CONTRACT,
        "logical_requests": 500,
        "main_online_calls": 0,
        "reference_changed": False,
        "sources": sources,
    }
    freeze_manifest(coverage, cm, ("panel_private.json", "requests_frozen.json"))
    result = {
        "contract_version": "dedup-v06212-v4-subject-v2-full1000-pipeline",
        "population": 1000,
        "repeat_count": 1,
        "stage": "full_pipeline",
        "parent": str(parent),
        "model": old["model"],
        "endpoint": old["endpoint"],
        "transport_projection": old["transport_projection"],
        "fixed_main": "v0.6.2.12 original raw replay v6-route",
        "main_online_calls": 0,
        "coverage_logical_requests": 500,
        "max_specialist_logical_requests": len(potential),
        "declared_final_action_contract": scope.CONTRACT,
        "coverage_manifest_sha256": sha256_file(coverage / "manifest.json"),
        "reference_changed": False,
        "release_eligible": False,
        "deadline_utc_epoch": selected.base.DEADLINE,
        "sources": sources,
    }
    return freeze_manifest(root, result, ("panel_private.json", "potential_subject_requests.json"))


def prepare_specialist(root: Path) -> dict:
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    plan = json.loads((root / "manifest.json").read_text())
    coverage = root / "coverage"
    require(
        sha256_file(coverage / "manifest.json") == plan["coverage_manifest_sha256"],
        "SUBJECT_FULL_COMPONENT",
        "frozen coverage manifest required",
    )
    selected.base.previous.reference.verify_freeze(coverage / "manifest.json")
    cm = json.loads((coverage / "manifest.json").read_text())
    ca = json.loads((coverage / "assessment.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    by = {p["canonical_pair_id"]: p for p in ca["cells"]["1/candidate"]["pairs"]}
    receipts = {}
    sources = dict(plan["sources"])
    for path in (coverage / "responses").glob("*.json"):
        receipt = json.loads(path.read_text())
        if receipt["arm"] == "candidate":
            receipts[receipt["canonical_pair_id"]] = receipt
            sources[str(path)] = sha256_file(path)
    eligible = set()
    for row in rows:
        pid = row["canonical_pair_id"]
        failure = by[pid]["status"] == "ENGINEERING_FAILURE"
        if failure:
            public = selected.base.critic.veto.unresolved_judge_output_v3()
        elif pid in receipts:
            receipt = receipts[pid]
            public, _, _ = selected.parse_output("candidate", receipt["assistant_content"], row, cm["specs"])
            require(
                receipt["status"] == "VALID" and public == receipt["public"],
                "SUBJECT_FULL_COVERAGE_REPLAY",
                "valid exact coverage base required",
            )
        else:
            require(
                selected.base.critic.veto.route(row["main_public"], row["payload"])
                != "REVIEW_POSITIVE_DIRECTIONS_ONLY",
                "SUBJECT_FULL_MISSING_COVERAGE",
                "missing active coverage output cannot bypass",
            )
            public = row["main_public"]
        row.update(coverage_base={"1": deepcopy(public)}, coverage_component_engineering_failure=failure)
        if subject_trial.subject.route(public, row["payload"]) == "REVIEW_BILATERAL_SUBJECTS":
            eligible.add(pid)
    potential = json.loads((root / "potential_subject_requests.json").read_text())
    requests = [r for r in potential if r["canonical_pair_id"] in eligible]
    require(
        {r["canonical_pair_id"] for r in requests} == eligible,
        "SUBJECT_FULL_ROUTING",
        "all admitted calls must be pre-frozen and solely deterministically routed",
    )
    for path in (root / "manifest.json", coverage / "assessment.json", coverage / "complete.json"):
        sources[str(path)] = sha256_file(path)
    target = root / "specialist"
    write_json_atomic(target / "panel_private.json", rows)
    write_json_atomic(target / "requests_frozen.json", requests)
    result = {
        "contract_version": "dedup-subject-v1-full1000-component",
        "stage": "full_specialist_component",
        "population": 1000,
        "repeat_count": 1,
        "model": plan["model"],
        "endpoint": plan["endpoint"],
        "specs": {"candidate": subject_trial.SPEC},
        "main_online_calls": 0,
        "logical_requests": len(requests),
        "declared_final_action_contract": scope.CONTRACT,
        "sources": sources,
    }
    return freeze_manifest(target, result, ("panel_private.json", "requests_frozen.json"))


def final_assessment(root: Path) -> dict:
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    rows = json.loads((root / "panel_private.json").read_text())
    coverage = json.loads((root / "coverage/assessment.json").read_text())
    limited = json.loads((root / "limited_scope/assessment.json").read_text())
    result = {
        "population": 1000,
        "stage": "full_pipeline_final",
        "main_online_calls": 0,
        "reference_changed": False,
        "reference_status": selected.base.previous.reference.DRAFT_STATUS,
        "release_eligible": False,
        "cells": {},
    }
    for name, original_cell in (
        ("1/fresh_legacy", coverage["cells"]["1/control"]),
        ("1/coverage_v4", coverage["cells"]["1/candidate"]),
        ("1/candidate", limited["cells"]["1/candidate"]),
    ):
        cell = deepcopy(original_cell)
        if name == "1/candidate":
            coverage_failed = {
                p["canonical_pair_id"]
                for p in coverage["cells"]["1/candidate"]["pairs"]
                if p["status"] == "ENGINEERING_FAILURE"
            }
            for p in cell["pairs"]:
                if p["canonical_pair_id"] in coverage_failed:
                    p.update(
                        status="ENGINEERING_FAILURE",
                        rule="COVERAGE_COMPONENT_FAILURE_PROPAGATED",
                        requires_cause_review=True,
                    )
            cell["engineering_failures"] = sum(p["status"] == "ENGINEERING_FAILURE" for p in cell["pairs"])
        predictions = [
            {
                "canonical_pair_id": p["canonical_pair_id"],
                **p["primary"],
                **({"metric_only_missing_output": True} if p["status"] == "ENGINEERING_FAILURE" else {}),
            }
            for p in cell["pairs"]
        ]
        cell["scores"] = {
            n: selected.base.previous.reference.score([r[f] for r in rows], predictions)
            for n, f in (("historical", "historical_label"), ("partial_draft", "draft_label"))
        }
        result["cells"][name] = cell
    for name, field in (("1/saved_12_main", "main_public"), ("1/saved_12_final", "saved_final")):
        predictions = [{"canonical_pair_id": r["canonical_pair_id"], **r[field]} for r in rows]
        pairs = [
            {
                "canonical_pair_id": r["canonical_pair_id"],
                "review_id": r["review_id"],
                "primary": selected.base.previous.reference.primary(r[field]),
                "draft_error": selected.base.previous.reference.classify_primary_error(r["draft_label"], r[field]),
                "status": "FIXED_SAVED_OUTPUT",
                "rule": "HISTORICAL_COMPARATOR",
                "requires_cause_review": selected.base.previous.reference.classify_primary_error(
                    r["draft_label"], r[field]
                )
                != "CORRECT",
            }
            for r in rows
        ]
        result["cells"][name] = {
            "scores": {
                n: selected.base.previous.reference.score([r[f] for r in rows], predictions)
                for n, f in (("historical", "historical_label"), ("partial_draft", "draft_label"))
            },
            "pairs": pairs,
            "engineering_failures": 0,
        }
    score = result["cells"]["1/candidate"]["scores"]["partial_draft"]["weighted"]
    result["full_development_precision_recall_75"] = (
        score["duplicate_precision"] is not None
        and score["duplicate_recall"] is not None
        and score["duplicate_precision"] >= 0.75
        and score["duplicate_recall"] >= 0.75
        and result["cells"]["1/candidate"]["engineering_failures"] == 0
    )
    result["full_development_75_gate"] = False
    result["promotion_status"] = "DIAGNOSTIC_ONLY_PENDING_ALL_ERROR_REVIEW_AND_INDEPENDENT_VALIDATION"
    result["source_assessments"] = {
        str(root / p): sha256_file(root / p)
        for p in ("coverage/assessment.json", "specialist/assessment.json", "limited_scope/assessment.json")
    }
    write_json_atomic(root / "assessment.json", result)
    totals = [json.loads((root / p).read_text()) for p in ("coverage/complete.json", "specialist/complete.json")]
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "external_attempts": sum(x["external_attempts"] for x in totals),
            "local_rejections": sum(x["local_rejections"] for x in totals),
            "main_online_calls": 0,
        },
    )
    return result


def run(root: Path, env_file: Path) -> dict:
    resilient.run(root / "coverage", env_file)
    prepare_specialist(root)
    subject_trial.run(root / "specialist", env_file)
    scope_trial.replay(root / "limited_scope", root / "specialist")
    return final_assessment(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=subject_trial.SESSION / "subject-v2-scope96")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.parent)
    else:
        require(args.env_file is not None, "SUBJECT_FULL_ENV", "explicit existing credential source required")
        result = run(args.root, args.env_file)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts", "cells")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
