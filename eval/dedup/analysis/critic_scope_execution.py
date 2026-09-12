# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Failure-durable execution for the unchanged, frozen critic-scope v1 experiment."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests

from eval.dedup.analysis import critic_scope_experiment as experiment
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

EXECUTION_CONTRACT = "dedup-critic-scope-execution-v2"
PRIOR = Path("/raid/hfang/ihb/runs/v06212-critic-scope-v1")


def prepare(root: Path) -> dict:
    root = root.resolve()
    require(not root.exists(), "CRITIC_SCOPE_ROOT_EXISTS", "new execution root required")
    base_root = root / "base_freeze"
    manifest = experiment.prepare(base_root)
    for name in ("requests_frozen.json", "panel_private.json"):
        write_text_atomic(root / name, (base_root / name).read_text())
    require(
        sha256_file(root / "requests_frozen.json") == sha256_file(PRIOR / "requests_frozen.json"),
        "CRITIC_SCOPE_EXECUTION_ONLY",
        "fresh execution must retain every request, setting and semantic prompt",
    )
    manifest.pop("contract_digest")
    manifest.update(
        execution_contract=EXECUTION_CONTRACT,
        prior_stopped_root=str(PRIOR),
        prior_requests_sha256=sha256_file(PRIOR / "requests_frozen.json"),
        semantic_change_from_first_pilot=False,
    )
    for path in (Path(__file__), Path(__file__).with_name("critic_scope_execution_v2_protocol.md")):
        manifest["sources"][str(path.resolve())] = sha256_file(path)
    manifest["artifacts"].update(
        {
            str(p): sha256_file(p)
            for p in (base_root / "manifest.json", root / "requests_frozen.json", root / "panel_private.json")
        }
    )
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def collect(request: dict, row: dict, *, endpoint: str, output: Path) -> dict:
    """Save the raw response before any schema/evidence parser can raise."""
    require(
        sha256_json(request["body"]) == request["request_sha256"], "CRITIC_SCOPE_REQUEST", "frozen request changed"
    )
    receipt = {k: request[k] for k in ("canonical_pair_id", "repeat", "arm", "request_sha256")}
    receipt["status"] = "REQUEST_STARTED"
    write_json_atomic(output.parent / "request_started" / output.name, receipt)
    try:
        response = requests.post(endpoint + "/chat/completions", json=request["body"], timeout=200)
        receipt["http_status"] = response.status_code
        if response.status_code != 200:
            receipt.update(status="TRANSPORT_FAILURE", error_code=f"HTTP_{response.status_code}")
        else:
            raw = response.json()
            receipt.update(raw_response=raw, status="RECEIVED_UNVALIDATED")
            write_json_atomic(output.parent / "raw_received" / output.name, receipt)
            choice = raw["choices"][0]
            require(choice.get("finish_reason") == "stop", "CRITIC_SCOPE_FINISH", "completion must not be truncated")
            content = choice["message"]["content"]
            receipt["assistant_content"] = content
            public, rule, value = experiment.parse_output(request["arm"], content, row)
            receipt.update(status="VALID", public=public, rule=rule, parsed_review=value)
    except DedupEvaluationError as exc:
        receipt.update(status="VALIDATION_FAILURE", error_code=exc.issue.code)
    except Exception as exc:  # noqa: BLE001 - exception messages may contain request or credential data
        receipt.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_code=type(exc).__name__)
    finally:
        write_json_atomic(output, receipt)
    return receipt


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    experiment.reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest.get("execution_contract") == EXECUTION_CONTRACT,
        "CRITIC_SCOPE_EXECUTION",
        "new execution contract required",
    )
    require(not (root / "started.json").exists(), "CRITIC_SCOPE_STARTED", "observed runs cannot restart")
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "CRITIC_SCOPE_CREDENTIAL", "existing NVIDIA credential is unavailable")
    rows = experiment._index(json.loads((root / "panel_private.json").read_text()), "fixed panel")
    frozen = json.loads((root / "requests_frozen.json").read_text())
    write_json_atomic(
        root / "started.json",
        {
            "at_utc": datetime.now(UTC).isoformat(),
            "contract_digest": manifest["contract_digest"],
            "credential_value_stored": False,
        },
    )
    with experiment.PacedRelay(
        profile=experiment.TransportProfile(**manifest["transport_profile"]),
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=experiment.GENERATION,
    ) as relay:
        relay.set_context(
            experiment.RelayContext("critic-scope-pilot19-execution-v2", 0, root / "transport_events.jsonl")
        )
        for repeat in (1, 2):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        collect,
                        r,
                        rows[r["canonical_pair_id"]],
                        endpoint=relay.endpoint,
                        output=root / "responses" / f"{r['repeat']}-{r['arm']}-{r['canonical_pair_id']}.json",
                    )
                    for r in frozen
                    if r["repeat"] == repeat
                ]
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
            experiment.reference.verify_freeze(root / "manifest.json")
    result = experiment.assess(root)
    write_json_atomic(
        root / "complete.json",
        {
            "execution_contract": EXECUTION_CONTRACT,
            "contract_digest": manifest["contract_digest"],
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "responses": len(list((root / "responses").glob("*.json"))),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    else:
        require(args.env_file is not None, "CRITIC_SCOPE_CREDENTIAL", "explicit existing env file required")
        result = run(args.root, args.env_file)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"sources", "artifacts", "cells", "response_artifacts", "main_replay", "saved_baselines"}
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
