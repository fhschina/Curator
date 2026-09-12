# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Isolate unsupported schema keywords without changing local acceptance rules."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

from eval.dedup.analysis import critic_schema_confirmation as prior
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

base = prior.probe.base


def planned_requests(session: Path) -> tuple[list[dict], dict]:
    originals, rows = prior.planned_requests(session)
    planned = []
    for mode in transport.MODES:
        for original in originals:
            if original["mode"] != "actual_critic_schema":
                continue
            body = {
                **original["body"],
                "response_format": transport.response_format(prior.critic.response_schema(), mode),
            }
            planned.append({"id": original["id"], "mode": mode, "body": body, "request_sha256": sha256_json(body)})
    return planned, rows


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    require(not root.exists(), "SCHEMA_PROJECT_ROOT", "new probe root required")
    predecessor = root.parent / "schema-confirmation-v1"
    sources = base.previous.reference.verify_freeze(predecessor / "manifest.json")
    for path in (
        Path(__file__).resolve(),
        Path(transport.__file__).resolve(),
        predecessor / "assessment.json",
        Path(__file__).with_name("critic_schema_projection_protocol.md").resolve(),
    ):
        sources[str(path)] = sha256_file(path)
    frozen, rows = planned_requests(root.parent)
    write_json_atomic(root / "requests_frozen.json", frozen)
    manifest = {
        "purpose": "SCHEMA_KEYWORD_COMPATIBILITY_NOT_SEMANTIC_VERSION_SELECTION",
        "logical_requests": len(frozen),
        "sources": sources,
        "local_contract_unchanged": prior.critic.CONTRACT,
        "artifacts": {str(root / "requests_frozen.json"): sha256_file(root / "requests_frozen.json")},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SCHEMA_PROJECT_CREDENTIAL", "existing credential unavailable")
    profile = base.previous.TransportProfile(
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
        record = {key: request[key] for key in ("id", "mode", "request_sha256")}
        if time.time() >= base.DEADLINE - base.STOP_ADMISSION_MARGIN:
            record["status"] = "DEADLINE_NOT_SUBMITTED"
        else:
            with base.previous.PacedRelay(
                profile=profile,
                logical_model="Qwen/Qwen3.8-27B-FP8",
                upstream_base_url="https://inference-api.nvidia.com/v1",
                upstream_model="nvidia/qwen/qwen3.8-27b",
                upstream_api_key=credential,
                timeout_seconds=160,
                expected_generation_parameters=base.previous.GENERATION,
            ) as relay:
                relay.set_context(
                    base.previous.RelayContext(
                        request["id"], 0, root / "transport" / f"{request['mode']}-{request['id']}.jsonl"
                    )
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
                        write_json_atomic(root / "raw_received" / f"{request['mode']}-{request['id']}.json", record)
                        choice = raw["choices"][0]
                        require(
                            choice.get("finish_reason") == "stop", "SCHEMA_PROJECT_FINISH", "complete output required"
                        )
                        content = choice["message"]["content"]
                        record["assistant_content"] = content
                        public, rule, value = prior.selected.parse_output(
                            "candidate",
                            content,
                            rows[request["id"]],
                            {"candidate": {"adapter": "eval.dedup.judging.critic_retention_v3"}},
                        )
                        record.update(status="LOCAL_CONTRACT_VALID", public=public, rule=rule, parsed_review=value)
                except DedupEvaluationError as exc:
                    record.update(status="VALIDATION_FAILURE", error_code=exc.issue.code)
                except Exception as exc:  # noqa: BLE001 - never persist credential-bearing exception details
                    record.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_type=type(exc).__name__)
        write_json_atomic(root / "responses" / f"{request['mode']}-{request['id']}.json", record)
        receipts.append(record)
        print(
            json.dumps({k: record.get(k) for k in ("id", "mode", "http_status", "status", "error_code")}), flush=True
        )
    events = [json.loads(line) for p in (root / "transport").glob("*.jsonl") for line in p.read_text().splitlines()]
    result = {
        "logical_requests": len(frozen),
        "external_requests": sum(e.get("external_request") is True for e in events),
        "local_rejections": sum(e.get("external_request") is False for e in events),
        "supported_local_valid_modes": [
            mode
            for mode in transport.MODES
            if all(r["status"] == "LOCAL_CONTRACT_VALID" for r in receipts if r["mode"] == mode)
        ],
        "judgment_performance_scored": False,
        "results": [{k: r.get(k) for k in ("id", "mode", "http_status", "status", "error_code")} for r in receipts],
        "response_artifacts": {str(p): sha256_file(p) for p in (root / "responses").glob("*.json")},
    }
    base.previous.reference.verify_freeze(root / "manifest.json")
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
