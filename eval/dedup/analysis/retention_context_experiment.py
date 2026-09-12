# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Clean paired rerun with immutable contextual-conflict and serial-startup revisions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import checkpoint_preflight as preflight
from eval.dedup.analysis import critic_scope_experiment as common
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.analysis import reference_rebenchmark as reference
from eval.dedup.analysis import retention_v4_experiment as previous
from eval.dedup.analysis.retention_v4_execution import warmup
from eval.dedup.analysis.retention_v4_experiment import check_context, collect, request_body
from eval.dedup.judging import critic_retention_v4 as baseline_critic
from eval.dedup.judging import critic_subject_binding as subject
from eval.dedup.judging import critic_subject_proof_verifier as baseline_verifier
from eval.dedup.judging import critic_subject_scope as baseline_scope
from eval.dedup.judging import retention_context_runtime as runtime
from eval.dedup.judging import retention_context_v1 as candidate
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.paced_relay_v2 import PacedRelayV2
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.judging.schema_v4 import JUDGE_SCHEMA_V4, read_versioned_output, unresolved_judge_output_v4
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "retention_context_experiment.md"
ARMS = previous.ARMS
GENERATION = previous.GENERATION


def prepare(root: Path, prior: Path) -> dict:
    root, prior = root.resolve(), prior.resolve()
    require(not root.exists(), "RETENTION_CONTEXT_ROOT", "new root and entirely fresh responses required")
    sources = reference.verify_freeze(prior / "manifest.json")
    old = preflight.read(prior / "manifest.json")
    complete = preflight.read(prior / "complete.json")
    require(
        old["version"] == "v0.6.2.33-exp2" and complete["assessment_sha256"] == sha256_file(prior / "assessment.json"),
        "RETENTION_CONTEXT_PRIOR",
        "bound completed predecessor required",
    )
    old_assessment = preflight.read(prior / "assessment.json")
    for path, digest in old_assessment["response_artifacts"].items():
        require(sha256_file(path) == digest, "RETENTION_CONTEXT_PRIOR_RECEIPT", "preserve every old receipt")
    rows = preflight.read(prior / "panel_private.json")
    requests = preflight.read(prior / "main_requests.json")
    require(len(rows) == 48 and old["repeat_count"] == 2, "RETENTION_CONTEXT_PANEL", "same 48-case paired pilot")
    by_id = {r["canonical_pair_id"]: r for r in rows}
    warmup()
    rebuilt = []
    for item in requests:
        require(sha256_json(item["body"]) == item["request_sha256"], "RETENTION_CONTEXT_BODY", "bound old request")
        row = by_id[item["canonical_pair_id"]]
        require(candidate.input_route(row["payload"]) == "MODEL_REVIEW", "RETENTION_CONTEXT_ROUTE", "unchanged route")
        current = deepcopy(item)
        if item["arm"] == "candidate":
            body = request_body(runtime.messages(row["payload"]), candidate.response_schema())
            current.update(body=body, request_sha256=sha256_json(body), input_tokens_with_schema=check_context(body))
        else:
            require(item["arm"] == "baseline", "RETENTION_CONTEXT_ARM", "two original arms only")
            check_context(current["body"])
        require(
            all(current["body"][k] == item["body"][k] for k in ("model", *GENERATION)),
            "RETENTION_CONTEXT_GENERATION",
            "same model and generation; no stronger-model or temperature change",
        )
        rebuilt.append(current)
    expected = {
        (arm, r["canonical_pair_id"])
        for r in rows
        if candidate.input_route(r["payload"]) == "MODEL_REVIEW"
        for arm in ARMS
    }
    require(
        len(rebuilt) == len(expected) and {(r["arm"], r["canonical_pair_id"]) for r in rebuilt} == expected,
        "RETENTION_CONTEXT_REQUEST_JOIN",
        "every active pair and both arms need an explicit frozen request",
    )
    sources.update(runtime.specification()["sources"])
    for path in (
        Path(__file__).resolve(),
        PROTOCOL,
        Path(__file__).with_name("retention_v4_execution.py"),
        HERE.parents[2] / "tests/eval/dedup/test_retention_context_v1.py",
        HERE.parents[2] / "tests/eval/dedup/test_retention_context_runtime.py",
        HERE.parents[2] / "tests/eval/dedup/test_retention_context_experiment.py",
        prior / "manifest.json",
        prior / "complete.json",
        prior / "assessment.json",
        prior / "panel_private.json",
        prior / "main_requests.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "main_requests.json", rebuilt)
    write_text_atomic(root / "protocol.md", PROTOCOL.read_text())
    require(
        sha256_file(root / "panel_private.json") == sha256_file(prior / "panel_private.json"),
        "RETENTION_CONTEXT_PANEL_BYTES",
        "original payloads, reference labels and gate expectations unchanged",
    )
    manifest = {
        **{
            k: old[k]
            for k in (
                "output_schema",
                "population",
                "repeat_count",
                "model",
                "endpoint",
                "generation",
                "max_logical_calls",
                "max_external_attempts",
                "main_requests_per_repeat",
            )
        },
        "version": runtime.VERSION,
        "model_response_contract": candidate.CONTRACT,
        "prior": str(prior),
        "status": "FROZEN_PAIRED_PILOT_NOT_RELEASE",
        "fresh_upstream": True,
        "old_cache_reused": False,
        "reference_changed": False,
        "baseline_main_requests_unchanged": True,
        "semantic_repair_retries": 0,
        "tokenizer_initialization": "SERIAL_BEFORE_WORKERS",
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def assessment(rows: list[dict], results: list[dict]) -> dict:
    return {**previous.assessment(rows, results), "version": runtime.VERSION}


def execute_case(
    root: Path,
    row: dict,
    arm: str,
    repeat: int,
    endpoint: str,
    main_request: dict | None,
    coverage_renderer: Callable[[dict], list[dict]],
) -> dict:
    payload = row["payload"]
    key = f"{repeat}-{arm}-{row['canonical_pair_id']}"
    stages = []

    def call(stage: str, body: dict) -> dict:
        check_context(body)
        receipt = collect(root, key + "-" + stage, body, endpoint)
        stages.append(
            {
                "stage": stage,
                "response_sha256": sha256_file(root / "responses" / (key + "-" + stage + ".json")),
                "status": receipt["status"],
            }
        )
        require(
            receipt["status"] == "RECEIVED",
            receipt.get("error_code", "RETENTION_PILOT_RESPONSE"),
            "fresh stage response unavailable",
        )
        return receipt["parsed"]

    result = {
        "canonical_pair_id": row["canonical_pair_id"],
        "review_id": row["review_id"],
        "arm": arm,
        "repeat": repeat,
        "stages": stages,
        "status": "VALID",
        "components": {},
    }
    try:
        route = candidate.input_route(payload)
        if route == "MODEL_REVIEW":
            require(
                main_request is not None and sha256_json(main_request["body"]) == main_request["request_sha256"],
                "RETENTION_PILOT_MAIN",
                "bound frozen main request required",
            )
            raw = call("main", main_request["body"])
            public = (
                common.critic.main_decision(raw, payload) if arm == "baseline" else candidate.adapt_main(raw, payload)
            )
        else:
            public = candidate.adapt_main(None, payload)
        result["components"]["main"] = deepcopy(public)
        review_route = (
            common.critic.route(public, payload) if arm == "baseline" else candidate.critic_route(public, payload)
        )
        if review_route == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            body = request_body(
                coverage_renderer(payload) if arm == "baseline" else runtime.messages(payload, stage="critic"),
                baseline_critic.response_schema() if arm == "baseline" else candidate.response_schema(),
            )
            raw = call("coverage", body)
            public, _ = (
                baseline_critic.apply_review(public, payload, raw)
                if arm == "baseline"
                else candidate.apply_critic(public, payload, raw)
            )
        result["components"]["coverage"] = deepcopy(public)
        route = (
            baseline_scope.route(public, payload) if arm == "baseline" else candidate.subject_route(public, payload)
        )
        if route == "REVIEW_BILATERAL_SUBJECTS":
            raw = call(
                "subject",
                request_body(
                    subject.messages(
                        payload, (runtime.CONFIG.parent / "v06212_subject_binding_v1_system.txt").read_text()
                    ),
                    subject.response_schema(),
                ),
            )
            if arm == "baseline":
                bound = {**payload, "subject_proposal": raw}
                eligible = baseline_verifier.route(public, bound) == "VERIFY_FIXED_SUBJECT_VETO"
            else:
                bound = {**payload, "subject_proposal": raw}
                eligible = candidate.subject_proof(public, payload, raw) is not None
            result["subject_proposal"] = raw
            if eligible:
                verification = call(
                    "verifier",
                    request_body(
                        baseline_verifier.messages(
                            bound, (runtime.CONFIG.parent / "v06212_subject_proof_verifier_v4_system.txt").read_text()
                        ),
                        baseline_verifier.response_schema(),
                    ),
                )
                public, _ = (
                    baseline_verifier.apply_review(public, bound, verification)
                    if arm == "baseline"
                    else candidate.apply_subject_verification(public, payload, raw, verification)
                )
        read_versioned_output(public, "dedup-judge-output-v3" if arm == "baseline" else JUDGE_SCHEMA_V4)
        candidate.validate_evidence_offsets(public, payload)
        result["public"] = public
    except DedupEvaluationError as exc:
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=exc.issue.code,
            public=unresolved_judge_output_v4("PILOT_ENGINEERING_FAILURE"),
        )
    except Exception as exc:  # noqa: BLE001 - persisted errors must not include secrets
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=type(exc).__name__,
            public=unresolved_judge_output_v4("PILOT_ENGINEERING_FAILURE"),
        )
    write_json_atomic(root / "results" / (key + ".json"), result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    reference.verify_freeze(root / "manifest.json")
    manifest = preflight.read(root / "manifest.json")
    require(
        not (root / "started.json").exists(), "RETENTION_PILOT_STARTED", "pilot cannot reuse any old stage responses"
    )
    require(manifest["version"] == runtime.VERSION, "RETENTION_CONTEXT_VERSION", "bound candidate version required")
    warmup()
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "RETENTION_PILOT_CREDENTIAL", "existing NVIDIA credential required")
    rows = preflight.read(root / "panel_private.json")
    requests_frozen = {(r["arm"], r["canonical_pair_id"]): r for r in preflight.read(root / "main_requests.json")}
    coverage_renderer = selected.renderer(
        {
            "adapter": "eval.dedup.judging.critic_retention_v4",
            "config": str(runtime.CONFIG.parent / "v06212_retention_v4.yaml"),
            "system": str(runtime.CONFIG.parent / "v06212_retention_v4_system.jinja"),
            "pair": str(runtime.CONFIG.parent / "v06212_retention_v4_pair.jinja"),
        }
    )
    write_json_atomic(
        root / "started.json",
        {
            "contract_digest": manifest["contract_digest"],
            "credential_source": str(env_file),
            "credential_persisted": False,
            "tokenizer_initialization": "SERIAL_BEFORE_WORKERS",
        },
    )
    results = []
    profile = TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=manifest["max_external_attempts"],
    )
    with PacedRelayV2(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=GENERATION,
    ) as relay:
        relay.set_context(RelayContext(runtime.VERSION, 0, root / "transport_events.jsonl"))
        for rep in (1, 2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        execute_case,
                        root,
                        row,
                        arm,
                        rep,
                        relay.endpoint,
                        requests_frozen.get((arm, row["canonical_pair_id"])),
                        coverage_renderer,
                    )
                    for row in rows
                    for arm in (ARMS if rep == 1 else tuple(reversed(ARMS)))
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(
                        json.dumps(
                            {
                                "completed": len(results),
                                "repeat": rep,
                                "arm": result["arm"],
                                "review_id": result["review_id"],
                                "status": result["status"],
                            }
                        ),
                        flush=True,
                    )
    report = assessment(rows, results)
    report["response_artifacts"] = {
        str(p): sha256_file(p)
        for folder in ("requests", "responses", "results")
        for p in (root / folder).glob("*.json")
    }
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    report["external_attempts"] = sum(bool(e.get("external_request")) for e in events)
    report["transport_retries"] = sum(e.get("upstream_attempt", 1) > 1 for e in events)
    report["http_statuses"] = dict(Counter(str(e.get("http_status")) for e in events if e.get("external_request")))
    write_json_atomic(root / "assessment.json", report)
    reference.verify_freeze(root / "manifest.json")
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "external_attempts": report["external_attempts"],
            "predeclared_candidate_checks_passed": report["predeclared_candidate_checks_passed"],
            "full1000_admitted": False,
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    require(
        args.prior is not None if args.command == "prepare" else args.env_file is not None,
        "RETENTION_CONTEXT_ARGUMENT",
        "explicit completed predecessor or existing credential source required",
    )
    result = prepare(args.root, args.prior) if args.command == "prepare" else run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"sources", "artifacts", "response_artifacts"}}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
