# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bounded critic-only pilot with immutable v0.6.2.12 mains and unchanged references."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml
from jinja2 import Environment, StrictUndefined

from eval.dedup.analysis import bottleneck_audit as historical
from eval.dedup.analysis import reference_rebenchmark as reference
from eval.dedup.analysis.development_diagnostic import _index, matched_raw_outputs
from eval.dedup.analysis.judge_calibration import _weight
from eval.dedup.judging import critic_intervention as critic
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.paced_relay import PacedRelay, TransportProfile
from eval.dedup.judging.payload import assert_blind_payload
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.judging.schema_v3 import JUDGE_FIELDS_V3
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "resources/local_ndd"
REFERENCE_ROOT = Path("/raid/hfang/ihb/runs/reference-policy-v2-rebenchmark-v1")
CONFIG = RESOURCES / "v06212_critic_scope_v1.yaml"
PILOT = (
    "H0108",
    "H0126",
    "H0419",
    "H0998",
    "H0093",
    "H0537",
    "H0701",
    "H0760",
    "H0195",
    "H0514",
    "H0850",
    "H0007",
    "H0893",
    "H0347",
    "H0417",
    "H0810",
    "H0316",
    "H0352",
    "H0748",
)
GUARDS = ("H0093", "H0537", "H0701", "H0760", "H0195", "H0514", "H0850", "H0316")
GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}


def strict_json(text: str) -> dict:
    def pairs(items: list) -> dict:
        value = {}
        for key, item in items:
            require(key not in value, "CRITIC_SCOPE_DUPLICATE_JSON_KEY", "duplicate keys are not repaired")
            value[key] = item
        return value

    result = json.loads(text, object_pairs_hook=pairs)
    require(isinstance(result, dict), "CRITIC_SCOPE_JSON", "one JSON object required")
    return result


def renderers() -> dict:
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    from eval.llm_judge.run_llm_judge import build_config_builder

    config = yaml.safe_load(CONFIG.read_text())
    base_path = RESOURCES / config["base_runner_config"]
    base = yaml.safe_load(base_path.read_text())
    original = base["execution"]["stages"][0]["judges"][1]
    require(
        config["contract_version"] == critic.CONTRACT and config["main_policy"] == "v6-route",
        "CRITIC_SCOPE_CONFIG",
        "base main routing and candidate contract are fixed",
    )
    require(
        set(config["rubric"]) == set(critic.ACTIONS),
        "CRITIC_SCOPE_RUBRIC",
        "rubric must cover precisely the action contract",
    )
    builder, _ = build_config_builder(
        base_path, endpoint="http://127.0.0.1:1/v1", models=base["models"], judges=[original]
    )
    column = builder.get_column_configs()[0]
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))
    env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701 - plain text LLM requests
    system = (RESOURCES / config["system_prompt_path"]).read_text()
    template = env.from_string((RESOURCES / config["prompt_path"]).read_text())

    def control(payload: dict) -> list[dict]:
        assert_blind_payload(payload)
        return [
            {
                "role": role,
                "content": renderer.render(
                    prompt_template=t, record={"payload": payload, "repair_feedback": None}, prompt_type=k
                ),
            }
            for role, t, k in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    def candidate(payload: dict) -> list[dict]:
        assert_blind_payload(payload)
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": template.render(
                    payload=payload,
                    rubric=json.dumps(config["rubric"], ensure_ascii=False),
                    schema=json.dumps(critic.response_schema()),
                ),
            },
        ]

    return {"control": control, "candidate": candidate}


def prepare(root: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root = root.resolve()
    require(
        not root.exists(),
        "CRITIC_SCOPE_ROOT_EXISTS",
        "new experimental root required; never overwrite an observed run",
    )
    sources = reference.verify_freeze(REFERENCE_ROOT / "summary.json")
    mains, finals, packets, proof = historical.component_replay(historical.FULL_ROOT, "v6", main_policy="v6-route")
    raw, digests = matched_raw_outputs(historical.FULL_ROOT, finals)
    sources.update(digests)
    refs = historical._read_csv(REFERENCE_ROOT / "reference_policy_v2_partial_draft.csv")
    labels = _index(refs, "draft")
    old_labels = _index(historical._read_csv(historical.REFERENCE), "historical reference")
    main, final, payloads = _index(mains, "main"), _index(finals, "final"), _index(packets, "packets")
    ids = {r["review_id"]: r["canonical_pair_id"] for r in refs}
    with (REFERENCE_ROOT / "ledger_1000.csv").open(encoding="utf-8", newline="") as stream:
        ledger = _index(list(csv.DictReader(stream)), "ledger")
    previous = {
        r["review_id"]: r for r in json.loads((reference.policy.REVIEW_ROOT / "review_121_private.json").read_text())
    }
    policy_cases = json.loads((REFERENCE_ROOT / "applications_private.json").read_text())
    applications = {r["review_id"]: r for r in policy_cases["supported"] + policy_cases["deferred"]}
    changed = [pid for pid in main if reference.primary(main[pid]) != reference.primary(final[pid])]
    require(len(changed) == 39, "CRITIC_SCOPE_CHANGED_SET", "all 39 historical interventions required")
    inventory = []
    for pid in sorted(changed, key=lambda p: labels[p]["review_id"]):
        rid = labels[pid]["review_id"]
        inventory.append(
            {
                "review_id": rid,
                "canonical_pair_id": pid,
                "weight": _weight(labels[pid]),
                "draft_errors": json.loads(ledger[pid]["draft_errors"]),
                "historical_flags": json.loads(ledger[pid]["historical_review_flags"]),
                "prior_review121": previous.get(rid),
                "policy_application": applications.get(rid),
                "main": raw[pid][0],
                "critic": raw[pid][1],
                "payload": payloads[pid]["payload"],
                "provenance": "PREDICTION_AWARE_DEVELOPMENT_INVENTORY_NOT_NEW_GOLD",
            }
        )
    panel = json.loads((reference.policy.REVIEW_ROOT / "panel_v2/private_selection.json").read_text())
    selected = (
        set(changed)
        | {r["canonical_pair_id"] for r in panel}
        | {ids[r] for r in PILOT}
        | {r["canonical_pair_id"] for r in policy_cases["supported"]}
    )
    private = []
    for pid in sorted(selected, key=lambda p: labels[p]["review_id"]):
        payload = payloads[pid]["payload"]
        private.append(
            {
                "review_id": labels[pid]["review_id"],
                "canonical_pair_id": pid,
                "payload": payload,
                "payload_sha256": sha256_json(payload),
                "raw_main": raw[pid][0],
                "fixed_main_sha256": sha256_json(raw[pid][0]),
                "main_public": {k: v for k, v in main[pid].items() if k != "canonical_pair_id"},
                "saved_final": {k: final[pid][k] for k in JUDGE_FIELDS_V3},
                "draft_label": labels[pid],
                "historical_label": old_labels[pid],
                "in_original64": pid in {r["canonical_pair_id"] for r in panel},
                "in_changed39": pid in changed,
                "in_pilot": labels[pid]["review_id"] in PILOT,
                "reference_flags": json.loads(ledger[pid]["historical_review_flags"]),
                "prior_review_assessment": previous.get(labels[pid]["review_id"], {}).get("assessment"),
                "policy_application_status": applications.get(labels[pid]["review_id"], {}).get("status"),
            }
        )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    render = renderers()
    requests_to_freeze = []
    for repeat in (1, 2):
        for row in private:
            if (
                not row["in_pilot"]
                or critic.route(row["main_public"], row["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY"
            ):
                continue
            for arm in ("control", "candidate") if repeat == 1 else ("candidate", "control"):
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **GENERATION,
                    "messages": render[arm](row["payload"]),
                    "response_format": {"type": "json_object"},
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(
                    tokens + 4096 + 2048 <= 32768,
                    "CRITIC_SCOPE_CONTEXT",
                    "pilot exceeds frozen client context guard; do not truncate",
                )
                requests_to_freeze.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": repeat,
                        "arm": arm,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                    }
                )
    require(len(requests_to_freeze) <= 76, "CRITIC_SCOPE_BUDGET", "pilot must remain bounded")
    write_json_atomic(root / "panel_private.json", private)
    write_json_atomic(root / "interventions39_private.json", inventory)
    write_json_atomic(root / "requests_frozen.json", requests_to_freeze)
    blind, key = historical.blind_packets(packets, selected)
    write_text_atomic(
        root / "review/inputs_blind.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind)
    )
    write_json_atomic(root / "review/private_key.json", key)
    resources = [
        CONFIG,
        RESOURCES / "v06212_critic_scope_v1_system.jinja",
        RESOURCES / "v06212_critic_scope_v1_pair.jinja",
        Path(critic.__file__),
        Path(__file__),
        HERE / "critic_scope_v1_protocol.md",
    ]
    sources.update({str(p.resolve()): sha256_file(p) for p in resources})
    manifest = {
        "contract_version": critic.CONTRACT,
        "status": "DIAGNOSTIC_NOT_RELEASE",
        "main_replay": proof,
        "fixed_main_version": "v0.6.2.12",
        "main_online_calls": 0,
        "reference_changed": False,
        "independently_adjudicated_new_pairs": 0,
        "panel_pairs": len(private),
        "historical_interventions": len(inventory),
        "pilot_review_ids": list(PILOT),
        "pilot_negative_guard_ids": list(GUARDS),
        "repeat_count": 2,
        "logical_requests": len(requests_to_freeze),
        "generation": GENERATION,
        "model": "nvidia/qwen/qwen3.8-27b",
        "endpoint": "https://inference-api.nvidia.com/v1",
        "transport_profile": {
            "min_interval_seconds": 2.0,
            "max_in_flight": 2,
            "max_attempts": 2,
            "retry_base_seconds": 10.0,
            "retry_cap_seconds": 40.0,
            "request_deadline_seconds": 180.0,
            "max_external_attempts": 152,
        },
        "schema_correction_retries": 0,
        "full_panel_online_authorized_by_this_protocol": False,
        "full_development_online_authorized_by_this_protocol": False,
        "max_input_tokens": max(r["input_tokens"] for r in requests_to_freeze),
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.rglob("*") if p.is_file()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def parse_output(arm: str, text: str, row: dict) -> tuple[dict, str, dict]:
    value = strict_json(text)
    if arm == "candidate":
        public, rule = critic.apply_review(row["main_public"], row["payload"], value)
        return public, rule, value
    require(
        arm == "control" and set(value) == {"record_binding_verdict", "retained_conflict"},
        "CRITIC_SCOPE_CONTROL_SCHEMA",
        "original control has exactly its two score fields",
    )
    require(
        all(isinstance(v, dict) and set(v) == {"score", "reasoning"} for v in value.values()),
        "CRITIC_SCOPE_CONTROL_SCHEMA",
        "control score fields must not be silently pruned",
    )
    public = adapt_ndd_judge_output(
        row["raw_main"],
        "dedup-judge-output-v3",
        payload=row["payload"],
        record_binding_critic=value,
        record_binding_policy="v6",
    )
    return public, "FRESH_V06212_CRITIC_CONTROL", value


def assess(root: Path) -> dict:
    reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    private = json.loads((root / "panel_private.json").read_text())
    pilot = [r for r in private if r["in_pilot"]]
    receipts = [json.loads(p.read_text()) for p in sorted((root / "responses").glob("*.json"))]
    found = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in receipts}
    frozen_requests = json.loads((root / "requests_frozen.json").read_text())
    requested = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in frozen_requests}
    require(len(found) == len(receipts), "CRITIC_SCOPE_RESPONSE_DUPLICATE", "duplicate response receipts")
    require(found.keys() <= requested.keys(), "CRITIC_SCOPE_RESPONSE_MEMBERSHIP", "unexpected response receipt")
    for key, receipt in found.items():
        require(
            receipt["request_sha256"] == requested[key]["request_sha256"],
            "CRITIC_SCOPE_RESPONSE_BINDING",
            "receipt must bind to its frozen request",
        )
        if receipt["status"] == "VALID":
            choice = receipt["raw_response"]["choices"][0]
            require(
                choice.get("finish_reason") == "stop" and choice["message"]["content"] == receipt["assistant_content"],
                "CRITIC_SCOPE_RESPONSE_BINDING",
                "accepted text must equal the complete raw response",
            )
    result = {
        "contract_digest": manifest["contract_digest"],
        "status": "LOCAL_PILOT_ONLY",
        "eligible_for_release": False,
        "reference_status": reference.DRAFT_STATUS,
        "pilot_pairs": len(pilot),
        "main_online_calls": 0,
        "logical_responses_saved": len(receipts),
        "cells": {},
        "guard_ids": list(GUARDS),
    }
    result["saved_baselines"] = {
        name: reference.score(
            [r["draft_label"] for r in pilot],
            [
                {
                    "canonical_pair_id": r["canonical_pair_id"],
                    **(
                        r["main_public"]
                        if name == "main"
                        or (
                            name == "route_only"
                            and critic.route(r["main_public"], r["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY"
                        )
                        else r["saved_final"]
                    ),
                }
                for r in pilot
            ],
        )
        for name in ("main", "saved_final", "route_only")
    }
    for repeat in (1, 2):
        for arm in ("control", "candidate"):
            predictions, diagnostics = [], []
            for row in pilot:
                routing = critic.route(row["main_public"], row["payload"])
                if routing != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
                    public, rule = row["main_public"], routing
                    status = "DETERMINISTIC_BYPASS"
                else:
                    receipt = found.get((repeat, arm, row["canonical_pair_id"]))
                    if receipt is None or receipt["status"] != "VALID":
                        public, rule, status = (
                            critic.unresolved_judge_output_v3(),
                            "MISSING_OR_INVALID_OUTPUT",
                            "ENGINEERING_FAILURE",
                        )
                    else:
                        public, rule, _ = parse_output(arm, receipt["assistant_content"], row)
                        require(
                            public == receipt["public"], "CRITIC_SCOPE_REPLAY", "saved public output no longer replays"
                        )
                        status = "VALID"
                predictions.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        **public,
                        **({"metric_only_missing_output": True} if status == "ENGINEERING_FAILURE" else {}),
                    }
                )
                diagnostics.append(
                    {
                        "review_id": row["review_id"],
                        "canonical_pair_id": row["canonical_pair_id"],
                        "status": status,
                        "rule": rule,
                        "primary": reference.primary(public),
                        "draft_error": reference.classify_primary_error(row["draft_label"], public),
                        "reference_flags": row["reference_flags"],
                        "policy_status": row["policy_application_status"],
                    }
                )
            scores = {
                label: reference.score([r[key] for r in pilot], predictions)
                for label, key in (("historical", "historical_label"), ("partial_draft", "draft_label"))
            }
            result["cells"][f"{repeat}/{arm}"] = {
                "scores": scores,
                "pairs": diagnostics,
                "guard_failures": [
                    d["review_id"]
                    for d in diagnostics
                    if d["review_id"] in GUARDS and d["primary"]["same_duplicate_group"] != "NO"
                ],
                "positive_target_failures": [
                    d["review_id"]
                    for d in diagnostics
                    if d["review_id"] in {"H0108", "H0126", "H0419", "H0998"} and d["draft_error"] != "CORRECT"
                ],
                "equivalence_protection_failures": [
                    d["review_id"]
                    for d in diagnostics
                    if d["review_id"] in {"H0007", "H0893", "H0347", "H0417", "H0810"}
                    and d["draft_error"] != "CORRECT"
                ],
                "engineering_failures": sum(d["status"] == "ENGINEERING_FAILURE" for d in diagnostics),
            }
    candidate = [result["cells"][f"{i}/candidate"] for i in (1, 2)]
    result["candidate_repeat_primary_disagreements"] = [
        a["review_id"]
        for a, b in zip(candidate[0]["pairs"], candidate[1]["pairs"], strict=True)
        if a["primary"] != b["primary"]
    ]
    result["next_step"] = (
        "STOP_BEFORE_EXPANSION"
        if result["candidate_repeat_primary_disagreements"]
        or any(
            c["guard_failures"]
            or c["engineering_failures"]
            or c["positive_target_failures"]
            or c["equivalence_protection_failures"]
            for c in candidate
        )
        else "REVIEW_PAIRED_SEMANTIC_RESULTS_BEFORE_ANY_EXPANSION"
    )
    result["observed_failures"] = [
        {
            "repeat": r["repeat"],
            "arm": r["arm"],
            "canonical_pair_id": r["canonical_pair_id"],
            "status": r["status"],
            "error_code": r.get("error_code"),
        }
        for r in receipts
        if r["status"] != "VALID"
    ]
    result["response_artifacts"] = {str(p): sha256_file(p) for p in sorted((root / "responses").glob("*.json"))}
    write_json_atomic(root / "assessment.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    require(not (root / "started.json").exists(), "CRITIC_SCOPE_STARTED", "an observed pilot cannot restart")
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "CRITIC_SCOPE_CREDENTIAL", "existing NVIDIA credential is unavailable")
    private = _index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen_requests = json.loads((root / "requests_frozen.json").read_text())
    write_json_atomic(
        root / "started.json",
        {
            "at_utc": datetime.now(UTC).isoformat(),
            "contract_digest": manifest["contract_digest"],
            "credential_value_stored": False,
        },
    )
    with PacedRelay(
        profile=TransportProfile(**manifest["transport_profile"]),
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=GENERATION,
    ) as relay:
        relay.set_context(RelayContext("critic-scope-pilot19", 0, root / "transport_events.jsonl"))

        def call(request: dict) -> dict:
            body = request["body"]
            require(
                sha256_json(body) == request["request_sha256"], "CRITIC_SCOPE_REQUEST", "request changed after freeze"
            )
            row = private[request["canonical_pair_id"]]
            receipt = {k: request[k] for k in ("canonical_pair_id", "repeat", "arm", "request_sha256")}
            try:
                response = requests.post(relay.endpoint + "/chat/completions", json=body, timeout=200)
                receipt["http_status"] = response.status_code
                if response.status_code != 200:
                    receipt.update(status="TRANSPORT_FAILURE", error_code=f"HTTP_{response.status_code}")
                else:
                    raw = response.json()
                    receipt["raw_response"] = raw
                    require(
                        raw["choices"][0].get("finish_reason") == "stop",
                        "CRITIC_SCOPE_FINISH",
                        "completion must not be truncated",
                    )
                    content = raw["choices"][0]["message"]["content"]
                    receipt["assistant_content"] = content
                    public, rule, value = parse_output(request["arm"], content, row)
                    receipt.update(status="VALID", public=public, rule=rule, parsed_review=value)
            except Exception as exc:  # noqa: BLE001 - persist safe failure metadata, never credential-bearing exception text
                receipt.update(
                    status="VALIDATION_OR_TRANSPORT_FAILURE",
                    error_code=exc.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__,
                )
            filename = f"{request['repeat']}-{request['arm']}-{request['canonical_pair_id']}.json"
            write_json_atomic(root / "responses" / filename, receipt)
            return receipt

        for repeat in (1, 2):
            batch = [r for r in frozen_requests if r["repeat"] == repeat]
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(call, r) for r in batch]
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
            require(
                all(sha256_file(p) == h for p, h in manifest["sources"].items()),
                "CRITIC_SCOPE_SOURCE_CHANGED",
                "sources changed during pilot",
            )
    result = assess(root)
    write_json_atomic(
        root / "complete.json",
        {
            "contract_digest": manifest["contract_digest"],
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "responses": len(list((root / "responses").glob("*.json"))),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    elif args.command == "run":
        require(args.env_file is not None, "CRITIC_SCOPE_CREDENTIAL", "explicit existing env file required")
        result = run(args.root, args.env_file)
    else:
        result = assess(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"sources", "artifacts", "cells", "response_artifacts", "main_replay"}
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
