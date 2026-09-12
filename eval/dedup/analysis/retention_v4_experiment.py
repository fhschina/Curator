# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Isolated, bounded paired pilot for the new retention contract."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import cache
from pathlib import Path

import requests
import yaml

from eval.dedup.analysis import checkpoint_preflight as preflight
from eval.dedup.analysis import critic_scope_experiment as common
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.analysis import reference_rebenchmark as reference
from eval.dedup.judging import critic_retention_v4 as baseline_critic
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.judging import critic_subject_binding as subject
from eval.dedup.judging import critic_subject_proof_verifier as baseline_verifier
from eval.dedup.judging import critic_subject_scope as baseline_scope
from eval.dedup.judging import retention_v4 as candidate
from eval.dedup.judging import retention_v4_runtime as runtime
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
PROTECTIONS = HERE / "retention_v4_protections.json"
PROTOCOL = HERE / "retention_v4_experiment.md"
PREFLIGHT = Path("/raid/hfang/ihb/runs/v06233-exp1-preflight-v1")
REBENCHMARK = Path("/raid/hfang/ihb/runs/v06212-v06233-exp1-adjudicated-rebenchmark-v1")
TITLE_CASES = {"H0106", "H0140", "H0185", "H0279", "H0292", "H0414", "H0831"}
ARMS = ("baseline", "candidate")
GENERATION = common.GENERATION


def baseline_renderer() -> tuple[Callable[[dict], list[dict]], dict]:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    from eval.llm_judge.run_llm_judge import build_config_builder

    path = runtime.CONFIG.parent / "hs_v06212_qwen_c64.yaml"
    config = yaml.safe_load(path.read_text())
    builder, _ = build_config_builder(
        path,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=[config["execution"]["stages"][0]["judges"][0]],
        provider_api_key="unused",
    )
    column = builder.get_column_configs()[0]
    recipe = create_response_recipe(column)
    renderer = RecordBasedPromptRenderer(recipe)

    def render(payload: dict) -> list[dict]:
        return [
            {
                "role": role,
                "content": renderer.render(
                    prompt_template=template, record={"payload": payload, "repair_feedback": None}, prompt_type=kind
                ),
            }
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render, recipe.data_type.model_json_schema()


def request_body(messages: list[dict], schema: dict) -> dict:
    return {
        "model": "Qwen/Qwen3.8-27B-FP8",
        **deepcopy(GENERATION),
        "messages": messages,
        "response_format": transport.response_format(schema, "portable_structure"),
    }


@cache
def local_tokenizer():  # noqa: ANN201 - optional model-specific tokenizer type
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def check_context(body: dict) -> int:
    tokenizer = local_tokenizer()
    count = len(
        tokenizer.apply_chat_template(
            body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
    )
    count += len(tokenizer.encode(json.dumps(body["response_format"])))
    require(
        count + 6144 <= 32768,
        "RETENTION_PILOT_CONTEXT",
        "full unchanged stage evidence plus schema and output reserve must fit",
    )
    return count


def offline_replay(panel: list[dict]) -> list[dict]:
    result = []
    for row in panel:
        route = candidate.input_route(row["payload"])
        public = None if route == "MODEL_REVIEW" else candidate.adapt_main(None, row["payload"])
        result.append(
            {
                "canonical_pair_id": row["canonical_pair_id"],
                "review_id": row["review_id"],
                "status": "UNAVAILABLE_MISSING_RETENTION_PROOF"
                if public is None
                else "DETERMINISTIC_EXACT_OR_PRESERVED_INPUT_ABSTENTION",
                "candidate_primary": None if public is None else preflight.primary(public),
                "raw_main_sha256": sha256_json(row["raw_main"]),
                "raw_output_reinterpreted": False,
            }
        )
    return result


def prepare(root: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root = root.resolve()
    require(not root.exists(), "RETENTION_PILOT_ROOT", "new experiment root/cache required")
    sources = reference.verify_freeze(PREFLIGHT / "summary.json")
    sources.update(reference.verify_freeze(REBENCHMARK / "summary.json"))
    spec = runtime.specification()
    sources.update(spec["sources"])
    session = preflight.benchmark.SESSION
    old = preflight.read(session / "subject-proof-v4-full1000/manifest.json")
    panel = preflight.read(session / "v4-subject-v2-full1000/panel_private.json")
    selected_ids = {r["canonical_pair_id"] for r in preflight.read(PREFLIGHT / "mechanism_panel.json")["cases"]}
    references = {
        r["canonical_pair_id"]: r
        for r in preflight.benchmark.historical._read_csv(REBENCHMARK / "reference_revised_1000.csv")
    }
    rows = []
    for row in panel:
        if row["canonical_pair_id"] not in selected_ids:
            continue
        ref = references[row["canonical_pair_id"]]
        expected = reference.primary(ref, "human_") if row["review_id"] in TITLE_CASES else None
        rows.append(
            {
                "canonical_pair_id": row["canonical_pair_id"],
                "review_id": row["review_id"],
                "payload": row["payload"],
                "population": "ORIGINAL_DEVELOPMENT_DIAGNOSTIC",
                "reference": ref,
                "gate_expected": expected,
            }
        )
    protections = preflight.read(PROTECTIONS)
    for case in protections["cases"]:
        payload = preflight.synthetic_payload(case["a"], case["b"])
        payload["payload_schema_version"] = "judge-visible-payload-v3"
        a, b = case["expected"]
        rows.append(
            {
                "canonical_pair_id": "synthetic_" + case["id"],
                "review_id": case["id"],
                "payload": payload,
                "population": protections["provenance"],
                "mechanism": case["mechanism"],
                "reference": None,
                "gate_expected": {
                    "a_can_replace_b": a,
                    "b_can_replace_a": b,
                    "same_duplicate_group": "YES" if "YES" in (a, b) else "NO",
                },
            }
        )
    require(
        len(rows) == 48 and len(selected_ids) == 36,
        "RETENTION_PILOT_POPULATION",
        "frozen 36 diagnostics plus 12 protections required",
    )
    render, schema = baseline_renderer()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    main_requests = []
    for row in rows:
        if candidate.input_route(row["payload"]) != "MODEL_REVIEW":
            continue
        for arm in ARMS:
            body = request_body(
                render(row["payload"]) if arm == "baseline" else runtime.messages(row["payload"]),
                schema if arm == "baseline" else candidate.response_schema(),
            )
            count = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            schema_tokens = len(tokenizer.encode(json.dumps(body["response_format"])))
            require(
                count + schema_tokens + 6144 <= 32768,
                "RETENTION_PILOT_CONTEXT",
                "complete original evidence plus schema and output reserve must fit",
            )
            main_requests.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    "arm": arm,
                    "body": body,
                    "request_sha256": sha256_json(body),
                    "input_tokens_with_schema": count + schema_tokens,
                }
            )
    for path in (
        Path(__file__).resolve(),
        PROTECTIONS,
        PROTOCOL,
        HERE.parents[2] / "tests/eval/dedup/test_retention_v4_experiment.py",
        HERE.parents[2] / "tests/eval/dedup/test_schema_v4.py",
        HERE.parents[2] / "tests/eval/dedup/test_retention_v4.py",
        HERE.parents[2] / "tests/eval/dedup/test_retention_v4_runtime.py",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "main_requests.json", main_requests)
    replay = offline_replay(panel)
    write_json_atomic(root / "offline_replay_1000.json", replay)
    write_text_atomic(root / "protocol.md", PROTOCOL.read_text())
    manifest = {
        "version": runtime.VERSION,
        "output_schema": JUDGE_SCHEMA_V4,
        "status": "FROZEN_PAIRED_PILOT_NOT_RELEASE",
        "population": len(rows),
        "repeat_count": 2,
        "model": old["model"],
        "endpoint": old["endpoint"],
        "generation": GENERATION,
        "max_logical_calls": 768,
        "max_external_attempts": 1536,
        "fresh_upstream": True,
        "semantic_repair_retries": 0,
        "old_cache_reused": False,
        "reference_changed": False,
        "offline_replay_counts": dict(Counter(r["status"] for r in replay)),
        "main_requests_per_repeat": len(main_requests),
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def collect(root: Path, key: str, body: dict, endpoint: str) -> dict:
    require(
        not (root / "requests" / (key + ".json")).exists(),
        "RETENTION_PILOT_REUSE",
        "fresh request required, never overwrite or reuse a receipt",
    )
    request = {"request_sha256": sha256_json(body), "body": body}
    write_json_atomic(root / "requests" / (key + ".json"), request)
    result = {"request_sha256": request["request_sha256"], "status": "REQUEST_STARTED"}
    try:
        response = requests.post(endpoint + "/chat/completions", json=body, timeout=200)
        result["http_status"] = response.status_code
        require(response.status_code == 200, "RETENTION_PILOT_HTTP", "non-successful transport")
        raw = response.json()
        result["raw_response"] = raw
        choice = raw["choices"][0]
        require(choice["finish_reason"] == "stop", "RETENTION_PILOT_FINISH", "truncated completions are failures")
        result["parsed"] = common.strict_json(choice["message"]["content"])
        result["status"] = "RECEIVED"
    except DedupEvaluationError as exc:
        result.update(status="FAILURE", error_code=exc.issue.code)
    except Exception as exc:  # noqa: BLE001 - exception strings can expose credentials
        result.update(status="FAILURE", error_code=type(exc).__name__)
    write_json_atomic(root / "responses" / (key + ".json"), result)
    return result


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


def assessment(rows: list[dict], results: list[dict]) -> dict:
    expected_keys = {(rep, arm, row["canonical_pair_id"]) for rep in (1, 2) for arm in ARMS for row in rows}
    by_key = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in results}
    require(
        len(by_key) == len(results) and set(by_key) == expected_keys,
        "RETENTION_PILOT_JOIN",
        "every paired case/repeat must remain in accounting",
    )
    cells = {}
    for arm in ARMS:
        for rep in (1, 2):
            cell = [by_key[(rep, arm, row["canonical_pair_id"])] for row in rows]
            failed = [
                row["review_id"]
                for row in rows
                if row["gate_expected"] is not None
                and preflight.primary(by_key[(rep, arm, row["canonical_pair_id"])]["public"]) != row["gate_expected"]
            ]
            cells[f"{rep}/{arm}"] = {
                "population": len(rows),
                "engineering_failures": sum(r["status"] != "VALID" for r in cell),
                "gate_failure_ids": failed,
                "group_unresolved": sum(r["public"]["same_duplicate_group"] == "UNRESOLVED" for r in cell),
                "main_calls": sum(any(s["stage"] == "main" for s in r["stages"]) for r in cell),
                "total_logical_calls": sum(len(r["stages"]) for r in cell),
            }
    repeat_changes = {
        arm: [
            row["review_id"]
            for row in rows
            if preflight.primary(by_key[(1, arm, row["canonical_pair_id"])]["public"])
            != preflight.primary(by_key[(2, arm, row["canonical_pair_id"])]["public"])
        ]
        for arm in ARMS
    }
    groups = {}
    for rep in (1, 2):
        by_text = {}
        for row in rows:
            a, b = (row["payload"][f"document_{s}"]["text"] for s in ("a", "b"))
            if (
                not all(isinstance(t, str) and t for t in (a, b))
                or row["payload"]["long_document_evidence"]["truncated"]
            ):
                continue
            normalized = preflight.normalized_primary(
                by_key[(rep, "candidate", row["canonical_pair_id"])]["public"], a, b
            )
            by_text.setdefault(sha256_json(sorted((a, b))), []).append((row["review_id"], sha256_json(normalized)))
        groups[str(rep)] = [[rid for rid, _ in items] for items in by_text.values() if len({p for _, p in items}) > 1]
    checks = (
        not repeat_changes["candidate"]
        and not any(groups.values())
        and all(
            not cells[f"{r}/candidate"]["gate_failure_ids"] and not cells[f"{r}/candidate"]["engineering_failures"]
            for r in (1, 2)
        )
    )
    return {
        "version": runtime.VERSION,
        "population": len(rows),
        "cells": cells,
        "repeat_primary_changes": repeat_changes,
        "candidate_exact_input_disagreements": groups,
        "predeclared_candidate_checks_passed": checks,
        "full1000_admitted": False,
        "release_eligible": False,
        "interpretation": "Selected mechanism pilot, not population precision/recall. Pending inherited reference disputes are preserved, not relabeled. Cause review is required before expansion.",
    }


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    reference.verify_freeze(root / "manifest.json")
    manifest = preflight.read(root / "manifest.json")
    require(
        not (root / "started.json").exists(), "RETENTION_PILOT_STARTED", "pilot cannot reuse any old stage responses"
    )
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        require(
            args.env_file is not None, "RETENTION_PILOT_CREDENTIAL", "explicit existing credential source required"
        )
    result = prepare(args.root) if args.command == "prepare" else run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"sources", "artifacts", "response_artifacts"}}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
