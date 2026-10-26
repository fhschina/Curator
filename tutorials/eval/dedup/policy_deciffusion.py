# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-panel native SystemOne comparison after the Qwen acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from http import HTTPStatus
from pathlib import Path

import requests
from jinja2 import Environment, StrictUndefined
from policy_cases import digest
from policy_eval import (
    FIELDS,
    acceptance_state,
    checked_report,
    decision,
    exact_index,
    read_json,
    read_rows,
    verify_files,
    write_json,
)

MODEL_ID = "nemotron-deciffusion/Nemotron-Deciffusion-8B-v2"
MODEL_REVISION = "1dea38015e86e101eb978a33fa493818fc71e366"
OPTIONS = {"YES", "NO", "UNRESOLVED"}


def parse_response(response: dict) -> dict:
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(FIELDS):
        msg = "Native response must contain both directional answers"
        raise ValueError(msg)
    result = {}
    for field in FIELDS:
        probabilities = answers[field].get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != OPTIONS:
            msg = "Expected the full YES/NO/UNRESOLVED probability distribution"
            raise ValueError(msg)
        if not all(
            isinstance(p, (float, int)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1
            for p in probabilities.values()
        ):
            msg = "Probabilities must be finite numbers in [0, 1]"
            raise ValueError(msg)
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5):
            msg = "Probabilities must sum to one"
            raise ValueError(msg)
        winners = [key for key, p in probabilities.items() if p == max(probabilities.values())]
        result[field] = (winners[0] if len(winners) == 1 else "UNRESOLVED").lower()
    return result


def request_body(row: dict, system: str, pair: str) -> dict:
    material = Environment(undefined=StrictUndefined, autoescape=False).from_string(pair).render(**row)  # noqa: S701
    start, end = material.index("<semantic_diff "), material.index("</semantic_diff>") + len("</semantic_diff>")
    evidence = material[start:end]
    policy = system.split("SPAN PACKET AND LEDGER", 1)[0]
    policy += "\nIncomplete or truncated evidence is UNRESOLVED. Answer only the requested replacement direction using the native choice options."
    return {
        "model": MODEL_ID.rsplit("/", 1)[1],
        "state": {"visible_material": evidence, "source_truncated": row["truncated"]},
        "questions": {
            field: {
                "type": "choice",
                "instructions": policy + "\n" + question,
                "criteria": {
                    "YES": "The keeper preserves all useful contributions under this policy",
                    "NO": "A useful substantive or linguistic contribution would be lost",
                    "UNRESOLVED": "Evidence or policy significance is insufficient",
                },
            }
            for field, question in zip(
                FIELDS, ("Can A replace B? Keep A, delete B.", "Can B replace A? Keep B, delete A."), strict=True
            )
        },
        "elimination": False,
    }


def verify_service(receipt: dict) -> None:
    if receipt.get("model_revision") != MODEL_REVISION or receipt.get("quantization") != "none":
        msg = "The service receipt must pin the BF16 Deciffusion v2 checkpoint"
        raise ValueError(msg)
    live = json.loads(
        subprocess.check_output([shutil.which("docker"), "inspect", receipt["container_id"]], text=True)  # noqa: S603
    )[0]
    env = dict(item.split("=", 1) for item in live["Config"]["Env"] if "=" in item)
    if (
        not live["State"]["Running"]
        or live["Image"] != receipt["image_id"]
        or any(
            env.get(key) != value
            for key, value in {
                "MODEL_ID": MODEL_ID,
                "MODEL_REVISION": MODEL_REVISION,
                "QUANTIZATION": "none",
                "ATTENTION": "noncausal",
                "COMPACT_OVER": "0",
                "COMPACT_BUDGET": "0",
            }.items()
        )
    ):
        msg = "The live container does not match its pinned native service receipt"
        raise ValueError(msg)


def require_qwen_acceptance(root: Path) -> None:
    if not acceptance_state(root)["passed"]:
        msg = "Deciffusion comparison follows Qwen human acceptance and the complete 20K run"
        raise ValueError(msg)
    for variant in ("baseline", "candidate"):
        result = checked_report(root, f"{variant}-full20k-1")
        if result["integrity_failures"] or result["unmapped_attempts"]:
            msg = "Qwen 20K provenance and engineering checks must pass before comparison"
            raise ValueError(msg)


def compare(args: argparse.Namespace) -> dict:  # noqa: PLR0915
    root = args.root.resolve()
    verify_files(root, read_json(root / "manifest.json")["files"])
    require_qwen_acceptance(root)
    service = read_json(args.service_receipt)
    verify_service(service)
    output = root / "deciffusion" / args.panel
    output.mkdir(parents=True, exist_ok=False)
    inputs = read_rows(root / "panels" / args.panel)
    qwen = exact_index(
        read_rows(root / "runs" / f"candidate-{args.panel}-1" / "critics"), {row["pair_id"] for row in inputs}
    )
    labels = (
        read_json(root / "independent_votes.json")
        if args.panel == "holdout"
        else read_json(root / "labels" / f"{args.panel}.json")
    )
    confirmed = {
        row["pair_id"]: row
        for row in labels
        if row.get("label_status") in {"consensus", "controlled", "human_independent"}
        and all(row.get(f) in {"yes", "no"} for f in FIELDS)
    }
    system = (root / "variants/candidate/judge_config/system.jinja").read_text()
    pair = (root / "variants/candidate/judge_config/pair.jinja").read_text()
    results = []
    started = time.time()
    for row in inputs:
        request = request_body(row, system, pair)
        folder = output / row["pair_id"]
        folder.mkdir()
        write_json(folder / "request.json", request)
        result = {
            "pair_id": row["pair_id"],
            "request_sha256": digest(json.dumps(request, sort_keys=True)),
            "status": "error",
        }
        for attempt in range(1, 4):
            before = time.time()
            receipt = {"attempt": attempt, "started": before}
            retry = False
            try:
                response = requests.post(args.endpoint.rstrip("/") + "/v1/systemone", json=request, timeout=600)
                receipt.update(http_status=response.status_code, response_text=response.text)
                retry = (
                    response.status_code == HTTPStatus.TOO_MANY_REQUESTS
                    or response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
                )
                response.raise_for_status()
                result.update(parse_response(response.json()), status="valid")
            except (ValueError, requests.RequestException) as error:
                receipt["error"] = str(error)
                retry = retry or isinstance(error, (requests.Timeout, requests.ConnectionError))
            receipt["elapsed_s"] = time.time() - before
            write_json(folder / f"attempt-{attempt}.json", receipt)
            if result["status"] == "valid" or not retry:
                break
        main = decision(qwen[row["pair_id"]], "main")
        result["qwen_main"] = {f: getattr(main, f) if main else "invalid" for f in FIELDS}
        results.append(result)
    exact_index(results, set(qwen))
    gold = [row for row in results if row["pair_id"] in confirmed]
    report = {
        "panel": args.panel,
        "model_revision": MODEL_REVISION,
        "total": len(results),
        "valid": sum(row["status"] == "valid" for row in results),
        "elapsed_s": time.time() - started,
        "scored_pairs": len(gold),
        "service_receipt_sha256": digest(args.service_receipt.read_bytes()),
        "candidate_policy_sha256": digest(system),
        "qwen_interface": "native structured judge, FP8",
        "deciffusion_interface": "native SystemOne choice probabilities, BF16",
        "interpretation": "Shared policy and evidence; interfaces and precision differ. This does not validate a Deciffusion critic pipeline.",
    }
    for model in ("deciffusion", "qwen_main"):
        correct = harmful = unresolved = 0
        for row in gold:
            answers = row if model == "deciffusion" else row["qwen_main"]
            label = confirmed[row["pair_id"]]
            for field in FIELDS:
                correct += answers.get(field) == label[field]
                harmful += answers.get(field) == "yes" and label[field] == "no"
                unresolved += answers.get(field) not in {"yes", "no"}
        report[model] = {
            "direction_accuracy": correct / (2 * len(gold)) if gold else None,
            "unsafe_allows": harmful,
            "unresolved_or_invalid_directions": unresolved,
        }
    write_json(output / "results.json", results)
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("gate", "diagnostic", "holdout"), required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--service-receipt", type=Path, required=True)
    print(json.dumps(compare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
