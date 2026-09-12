# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Frozen critic experiments with explicit versioned adapters and a hard deadline."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import requests
import yaml
from jinja2 import Environment, StrictUndefined

from eval.dedup.analysis import critic_retention_experiment as base
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

DEFAULT_CONFIG = base.previous.RESOURCES / "v06212_retention_v3.yaml"
DEFAULT_ADAPTER = "eval.dedup.judging.critic_retention_v3"
DEFAULT_PROTOCOL = base.HERE / "critic_retention_v3_protocol.md"


def specification(adapter_name: str, config_path: Path) -> tuple[dict, dict]:
    module = importlib.import_module(adapter_name)
    config = yaml.safe_load(config_path.read_text())
    require(config["contract_version"] == module.CONTRACT, "SELECTED_CONTRACT", "adapter and prompt contract agree")
    paths = {
        "config": config_path.resolve(),
        "system": (config_path.parent / config["system_prompt_path"]).resolve(),
        "pair": (config_path.parent / config["prompt_path"]).resolve(),
        "adapter_file": Path(module.__file__).resolve(),
    }
    sources = {str(p): sha256_file(p) for p in paths.values()}
    if config.get("predecessor_review_path"):
        review_path = Path(config["predecessor_review_path"])
        review = json.loads(review_path.read_text())
        assessment_path = review_path.parent / "assessment.json"
        assessment = json.loads(assessment_path.read_text())
        errors = {
            row["review_id"]
            for cell in assessment["cells"].values()
            for row in cell["pairs"]
            if row["draft_error"] != "CORRECT" or row["status"] == "ENGINEERING_FAILURE"
        }
        require(
            review.get("all_observed_disagreements_reviewed") is True
            and review.get("assessment_sha256") == sha256_file(assessment_path)
            and errors <= set(review.get("cases", {})),
            "SELECTED_PREDECESSOR_REVIEW",
            "every previous disagreement needs a cause review bound to the actual assessment",
        )
        sources.update(base.previous.reference.verify_freeze(review_path.parent / "manifest.json"))
        sources.update({str(p): sha256_file(p) for p in (review_path, assessment_path)})
    return {"adapter": adapter_name, **{k: str(v) for k, v in paths.items()}}, sources


def renderer(spec: dict) -> Callable[[dict], list[dict]]:
    module = importlib.import_module(spec["adapter"])
    config = yaml.safe_load(Path(spec["config"]).read_text())
    system = Path(spec["system"]).read_text()
    env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701 - plain text
    pair = env.from_string(Path(spec["pair"]).read_text())

    def render(payload: dict) -> list[dict]:
        base.previous.assert_blind_payload(payload)
        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": pair.render(
                    payload=payload,
                    rubric=json.dumps(config["rubric"], ensure_ascii=False),
                    schema=json.dumps(module.response_schema()),
                ),
            },
        ]

    return render


def parse_output(arm: str, text: str, row: dict, specs: dict) -> tuple[dict, str, dict]:
    if arm == "control":
        return base.previous.parse_output(arm, text, row)
    require(arm in specs, "SELECTED_ARM", "unknown frozen critic arm")
    module = importlib.import_module(specs[arm]["adapter"])
    value = base.previous.strict_json(text)
    public, rule = module.apply_review(row["main_public"], row["payload"], value)
    return public, rule, value


def prepare(
    root: Path,
    *,
    stage: str,
    parent: Path | None,
    adapter: str,
    config: Path,
    protocol: Path,
    control_adapter: str,
    control_config: Path,
) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root = root.resolve()
    require(not root.exists(), "SELECTED_ROOT_EXISTS", "new contract needs a new root")
    original = base.prepare(root / "input_freeze", stage=stage, parent=parent)
    candidate, sources = specification(adapter, config)
    encoding, more_sources = specification(control_adapter, control_config)
    sources.update(more_sources)
    sources.update(base.previous.reference.verify_freeze(root / "input_freeze/manifest.json"))
    for path in (Path(__file__).resolve(), protocol.resolve()):
        sources[str(path)] = sha256_file(path)
    rows = json.loads((root / "input_freeze/panel_private.json").read_text())
    render = {
        "control": base.previous.renderers()["control"],
        "candidate": renderer(candidate),
        "encoding_control": renderer(encoding),
    }
    specs = {"candidate": candidate, "encoding_control": encoding}
    arms = ("control", "encoding_control", "candidate") if stage == "pilot" else ("control", "candidate")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    frozen = []
    for repeat in range(1, original["repeat_count"] + 1):
        for row in rows:
            if base.critic.veto.route(row["main_public"], row["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
                continue
            for arm in arms if repeat % 2 else tuple(reversed(arms)):
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **base.previous.GENERATION,
                    "messages": render[arm](row["payload"]),
                    "response_format": {"type": "json_object"},
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "SELECTED_CONTEXT", "full original visible input must fit unchanged")
                frozen.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": repeat,
                        "arm": arm,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                        "deadline_utc_epoch": base.DEADLINE,
                    }
                )
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", frozen)
    manifest = {
        k: original[k]
        for k in (
            "stage",
            "deadline_utc_epoch",
            "stop_admission_margin_seconds",
            "model",
            "endpoint",
            "repeat_count",
            "population",
            "fixed_main",
            "main_online_calls",
            "reference_changed",
            "pending_adjudication",
        )
    }
    manifest.update(
        contract_version=importlib.import_module(adapter).CONTRACT,
        arms=list(arms),
        specs=specs,
        logical_requests=len(frozen),
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "panel_private.json", root / "requests_frozen.json")},
    )
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def collect(request: dict, row: dict, *, endpoint: str, output: Path, specs: dict) -> dict:
    require(sha256_json(request["body"]) == request["request_sha256"], "SELECTED_REQUEST", "frozen body changed")
    receipt = {k: request[k] for k in ("canonical_pair_id", "repeat", "arm", "request_sha256")}
    receipt["status"] = "REQUEST_STARTED"
    write_json_atomic(output.parent / "request_started" / output.name, receipt)
    try:
        if time.time() >= request["deadline_utc_epoch"] - base.STOP_ADMISSION_MARGIN:
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
                receipt["assistant_content"] = choice["message"]["content"]
                require(choice.get("finish_reason") == "stop", "SELECTED_FINISH", "truncated completions are failures")
                public, rule, parsed = parse_output(request["arm"], receipt["assistant_content"], row, specs)
                receipt.update(status="VALID", public=public, rule=rule, parsed_review=parsed)
    except DedupEvaluationError as exc:
        receipt.update(status="VALIDATION_FAILURE", error_code=exc.issue.code)
    except Exception as exc:  # noqa: BLE001 - exception details may expose credentials
        receipt.update(status="TRANSPORT_OR_FORMAT_FAILURE", error_code=type(exc).__name__)
    finally:
        write_json_atomic(output, receipt)
    return receipt


def assess(root: Path) -> dict:
    base.previous.reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    requests_frozen = json.loads((root / "requests_frozen.json").read_text())
    requested = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in requests_frozen}
    receipts = [json.loads(p.read_text()) for p in sorted((root / "responses").glob("*.json"))]
    found = {(r["repeat"], r["arm"], r["canonical_pair_id"]): r for r in receipts}
    require(
        len(found) == len(receipts) and found.keys() <= requested.keys(), "SELECTED_RECEIPTS", "unique known responses"
    )
    for key, receipt in found.items():
        require(receipt["request_sha256"] == requested[key]["request_sha256"], "SELECTED_BINDING", "request binding")
        if receipt["status"] == "VALID":
            choice = receipt["raw_response"]["choices"][0]
            require(
                choice.get("finish_reason") == "stop" and choice["message"]["content"] == receipt["assistant_content"],
                "SELECTED_RAW_BINDING",
                "accepted text must equal complete raw output",
            )
    result = {
        "contract_digest": manifest["contract_digest"],
        "stage": manifest["stage"],
        "population": len(rows),
        "release_eligible": False,
        "reference_status": base.previous.reference.DRAFT_STATUS,
        "main_online_calls": 0,
        "responses_saved": len(receipts),
        "logical_requests": len(requested),
        "cells": {},
    }
    for repeat in range(1, manifest["repeat_count"] + 1):
        for arm in manifest["arms"]:
            predictions, diagnostics = [], []
            for row in rows:
                pid, rid = row["canonical_pair_id"], row["review_id"]
                rule = base.critic.veto.route(row["main_public"], row["payload"])
                receipt = found.get((repeat, arm, pid))
                if rule != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
                    public, status = row["main_public"], "DETERMINISTIC_BYPASS"
                elif receipt is None or receipt["status"] != "VALID":
                    public, status, rule = (
                        base.critic.veto.unresolved_judge_output_v3(),
                        "ENGINEERING_FAILURE",
                        "MISSING_OR_INVALID_OUTPUT",
                    )
                else:
                    public, rule, _ = parse_output(arm, receipt["assistant_content"], row, manifest["specs"])
                    require(public == receipt["public"], "SELECTED_REPLAY", "public output must replay")
                    status = "VALID"
                predictions.append(
                    {
                        "canonical_pair_id": pid,
                        **public,
                        **({"metric_only_missing_output": True} if status == "ENGINEERING_FAILURE" else {}),
                    }
                )
                error = base.previous.reference.classify_primary_error(row["draft_label"], public)
                diagnostics.append(
                    {
                        "canonical_pair_id": pid,
                        "review_id": rid,
                        "primary": base.previous.reference.primary(public),
                        "draft_error": error,
                        "status": status,
                        "rule": rule,
                        "error_code": (receipt or {}).get("error_code"),
                        "pending_policy_question": base.DISPUTES.get(rid),
                        "requires_cause_review": error != "CORRECT" or status == "ENGINEERING_FAILURE",
                    }
                )
            result["cells"][f"{repeat}/{arm}"] = {
                "scores": {
                    name: base.previous.reference.score([r[field] for r in rows], predictions)
                    for name, field in (("historical", "historical_label"), ("partial_draft", "draft_label"))
                },
                "pairs": diagnostics,
                "engineering_failures": sum(d["status"] == "ENGINEERING_FAILURE" for d in diagnostics),
                "guard_failures": [
                    d["review_id"]
                    for d in diagnostics
                    if (d["review_id"] in base.NEGATIVES and d["primary"]["same_duplicate_group"] != "NO")
                    or (d["review_id"] in (*base.POSITIVES, *base.EQUIVALENT) and d["draft_error"] != "CORRECT")
                ],
            }
    candidates = [result["cells"][f"{i}/candidate"] for i in range(1, manifest["repeat_count"] + 1)]
    disagreements = []
    if len(candidates) == 2:
        disagreements = [
            a["review_id"]
            for a, b in zip(candidates[0]["pairs"], candidates[1]["pairs"], strict=True)
            if a["primary"] != b["primary"] or a["status"] != b["status"]
        ]
    worse = any(
        c["scores"]["partial_draft"]["weighted"]["primary_decision_exact"]
        < result["cells"][f"{i}/control"]["scores"]["partial_draft"]["weighted"]["primary_decision_exact"]
        for i, c in enumerate(candidates, 1)
    )
    result.update(
        candidate_repeat_disagreements=disagreements,
        next_step="STOP_REVIEW_AND_REPAIR"
        if disagreements or worse or any(c["engineering_failures"] or c["guard_failures"] for c in candidates)
        else "REVIEW_BEFORE_EXPANSION",
    )
    score = candidates[0]["scores"]["partial_draft"]["weighted"]
    result["full_development_75_gate"] = (
        manifest["stage"] == "full"
        and result["next_step"] == "REVIEW_BEFORE_EXPANSION"
        and score["duplicate_precision"] >= 0.75
        and score["duplicate_recall"] >= 0.75
        and score["primary_decision_exact"] >= 0.79
    )
    result["response_artifacts"] = {str(p): sha256_file(p) for p in (root / "responses").rglob("*.json")}
    write_json_atomic(root / "assessment.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    base.previous.reference.verify_freeze(root / "manifest.json")
    require(not (root / "started.json").exists(), "SELECTED_STARTED", "never restart an observed run")
    require(
        time.time() < base.DEADLINE - base.STOP_ADMISSION_MARGIN, "SELECTED_DEADLINE", "do not start near deadline"
    )
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "SELECTED_CREDENTIAL", "existing credential unavailable")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = base.previous._index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen = json.loads((root / "requests_frozen.json").read_text())
    write_json_atomic(
        root / "started.json", {"at_utc": datetime.now(UTC).isoformat(), "deadline_utc_epoch": base.DEADLINE}
    )
    profile = base.previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=2 * len(frozen),
    )
    with base.previous.PacedRelay(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=base.previous.GENERATION,
    ) as relay:
        relay.set_context(base.previous.RelayContext(manifest["contract_version"], 0, root / "transport_events.jsonl"))
        for repeat in range(1, manifest["repeat_count"] + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        collect,
                        r,
                        rows[r["canonical_pair_id"]],
                        endpoint=relay.endpoint,
                        output=root / "responses" / f"{repeat}-{r['arm']}-{r['canonical_pair_id']}.json",
                        specs=manifest["specs"],
                    )
                    for r in frozen
                    if r["repeat"] == repeat
                ]
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
            base.previous.reference.verify_freeze(root / "manifest.json")
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
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--control-adapter", default="eval.dedup.judging.critic_retention_v2")
    parser.add_argument("--control-config", type=Path, default=base.previous.RESOURCES / "v06212_retention_v2.yaml")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.root,
            stage=args.stage,
            parent=args.parent,
            adapter=args.adapter,
            config=args.config,
            protocol=args.protocol,
            control_adapter=args.control_adapter,
            control_config=args.control_config,
        )
    elif args.command == "run":
        require(args.env_file is not None, "SELECTED_CREDENTIAL", "explicit existing env file required")
        result = run(args.root, args.env_file)
    else:
        result = assess(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("sources", "artifacts", "response_artifacts", "cells", "specs")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
