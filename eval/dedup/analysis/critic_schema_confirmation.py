# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Isolated confirmation of the observed schema mode and the actual critic contract."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

from eval.dedup.analysis import critic_schema_probe as probe
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_retention_v3 as critic
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic


def planned_requests(session: Path) -> tuple[list[dict], dict]:
    requests_to_make = [
        {"id": f"synthetic-{i}", "mode": mode, "sentinel": sentinel, "body": probe.request_body(mode, sentinel)}
        for i, (mode, sentinel) in enumerate(
            (
                ("json_object", "UNUSED"),
                ("json_schema_standard", "ONLY_SCHEMA_FIRST"),
                ("json_schema_standard", "ONLY_SCHEMA_SECOND"),
            ),
            1,
        )
    ]
    prior = session / "retention-v3-pilot"
    rows = json.loads((prior / "panel_private.json").read_text())
    prior_requests = json.loads((prior / "requests_frozen.json").read_text())
    selected_rows = {r["review_id"]: r for r in rows if r["review_id"] in ("H0701", "H0893", "H0850")}
    for rid, row in selected_rows.items():
        original = next(
            r
            for r in prior_requests
            if r["canonical_pair_id"] == row["canonical_pair_id"] and r["arm"] == "candidate" and r["repeat"] == 1
        )
        body = {
            **original["body"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "critic_retention_v3", "strict": True, "schema": critic.response_schema()},
            },
        }
        requests_to_make.append(
            {
                "id": rid,
                "mode": "actual_critic_schema",
                "body": body,
                "original_request_sha256": original["request_sha256"],
            }
        )
    for request in requests_to_make:
        request["request_sha256"] = sha256_json(request["body"])
    return requests_to_make, selected_rows


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    require(not root.exists(), "SCHEMA_CONFIRM_ROOT", "new root required")
    require(
        time.time() < probe.base.DEADLINE - probe.base.STOP_ADMISSION_MARGIN,
        "SCHEMA_CONFIRM_DEADLINE",
        "no late calls",
    )
    source_root = root.parent / "schema-capability-v1"
    sources = probe.base.previous.reference.verify_freeze(source_root / "manifest.json")
    for path in (
        source_root / "assessment.json",
        source_root / "upstream_audit.json",
        Path(__file__).resolve(),
        Path(__file__).with_name("critic_schema_confirmation_protocol.md").resolve(),
        Path(critic.__file__).resolve(),
    ):
        sources[str(path)] = sha256_file(path)
    sources.update(probe.base.previous.reference.verify_freeze(root.parent / "retention-v3-pilot/manifest.json"))
    frozen, rows = planned_requests(root.parent)
    write_json_atomic(root / "requests_frozen.json", frozen)
    manifest = {
        "purpose": "ISOLATED_SCHEMA_CONFIRMATION_NOT_VERSION_SELECTION",
        "logical_requests": 6,
        "deadline_utc_epoch": probe.base.DEADLINE,
        "sources": sources,
        "artifacts": {str(root / "requests_frozen.json"): sha256_file(root / "requests_frozen.json")},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SCHEMA_CONFIRM_CREDENTIAL", "existing credential unavailable")
    profile = probe.base.previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=1,
        max_attempts=1,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=1,
    )
    receipts, last_start = [], time.monotonic() - 2
    for request in frozen:
        time.sleep(max(0, 2 - (time.monotonic() - last_start)))
        record = {k: request[k] for k in ("id", "mode", "request_sha256")}
        if time.time() >= probe.base.DEADLINE - probe.base.STOP_ADMISSION_MARGIN:
            record["status"] = "DEADLINE_NOT_SUBMITTED"
        else:
            generation = {k: request["body"][k] for k in probe.base.previous.GENERATION}
            with probe.base.previous.PacedRelay(
                profile=profile,
                logical_model="Qwen/Qwen3.8-27B-FP8",
                upstream_base_url="https://inference-api.nvidia.com/v1",
                upstream_model="nvidia/qwen/qwen3.8-27b",
                upstream_api_key=credential,
                timeout_seconds=160,
                expected_generation_parameters=generation,
            ) as relay:
                relay.set_context(
                    probe.base.previous.RelayContext(request["id"], 0, root / "transport" / f"{request['id']}.jsonl")
                )
                last_start = time.monotonic()
                try:
                    response = requests.post(relay.endpoint + "/chat/completions", json=request["body"], timeout=200)
                    record["http_status"] = response.status_code
                    if response.status_code != 200:
                        record["status"] = "HTTP_REJECTED"
                    else:
                        raw = response.json()
                        record["raw_response"] = raw
                        write_json_atomic(root / "raw_received" / f"{request['id']}.json", record)
                        if request["mode"] != "actual_critic_schema":
                            record["status"] = probe.classify(request["mode"], request["sentinel"], raw)
                        else:
                            choice = raw["choices"][0]
                            require(
                                choice.get("finish_reason") == "stop",
                                "SCHEMA_CONFIRM_FINISH",
                                "complete output required",
                            )
                            content = choice["message"]["content"]
                            public, rule, value = selected.parse_output(
                                "candidate",
                                content,
                                rows[request["id"]],
                                {"candidate": {"adapter": "eval.dedup.judging.critic_retention_v3"}},
                            )
                            record.update(
                                status="CRITIC_SCHEMA_AND_PROOF_VALID",
                                assistant_content=content,
                                public=public,
                                rule=rule,
                                parsed_review=value,
                            )
                except DedupEvaluationError as exc:
                    record.update(status="VALIDATION_FAILURE", error_code=exc.issue.code)
                except Exception as exc:  # noqa: BLE001 - only safe error types, never credential-bearing messages
                    record.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_type=type(exc).__name__)
        write_json_atomic(root / "responses" / f"{request['id']}.json", record)
        receipts.append(record)
        print(
            json.dumps({k: record.get(k) for k in ("id", "mode", "http_status", "status", "error_code")}), flush=True
        )
    events = [json.loads(line) for p in (root / "transport").glob("*.jsonl") for line in p.read_text().splitlines()]
    result = {
        "logical_requests": len(frozen),
        "external_requests": sum(e.get("external_request") is True for e in events),
        "local_rejections": sum(e.get("external_request") is False for e in events),
        "sentinel_constraints_observed": all(
            r["status"] == "SCHEMA_CONSTRAINT_OBSERVED" for r in receipts if r["mode"] == "json_schema_standard"
        ),
        "actual_critic_schema_valid": all(
            r["status"] == "CRITIC_SCHEMA_AND_PROOF_VALID" for r in receipts if r["mode"] == "actual_critic_schema"
        ),
        "results": [{k: r.get(k) for k in ("id", "mode", "http_status", "status", "error_code")} for r in receipts],
        "judgment_performance_scored": False,
        "limitation": "Small compatibility probe only, not evidence of full-population semantic improvement.",
        "response_artifacts": {str(p): sha256_file(p) for p in (root / "responses").glob("*.json")},
    }
    probe.base.previous.reference.verify_freeze(root / "manifest.json")
    write_json_atomic(root / "assessment.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.env_file)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
