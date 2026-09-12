# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Repeat the unchanged subject stage, then verify all newly proposed in-scope vetoes."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_subject_proof_experiment as proof
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

subject = proof.previous
reference = proof.reference
CONTRACT = "dedup-subject-proof-v4-fresh-upstream-repeat-v1"


def freeze(root: Path, result: dict, rows: list, requests: list) -> dict:
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests)
    result["artifacts"] = {
        str(root / name): sha256_file(root / name) for name in ("panel_private.json", "requests_frozen.json")
    }
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def prepare(root: Path, parent: Path) -> dict:
    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "SUBJECT_REFRESH_ROOT", "new root required")
    sources = proof.kinds_trial.reviewed_sources(parent, "FRESH_SUBJECT_PROOF_STABILITY_FULL1000")
    assessment = json.loads((parent / "assessment.json").read_text())
    require(
        assessment["population"] == 1000
        and not assessment["candidate_repeat_disagreements"]
        and not any(assessment["clear_scope_guard_failures"].values())
        and all(assessment["cells"][f"{r}/candidate"]["engineering_failures"] == 0 for r in (1, 2)),
        "SUBJECT_REFRESH_GATE",
        "both full conditional repeats must preserve the predeclared clear protections",
    )
    origin = Path(json.loads((parent / "manifest.json").read_text())["coverage_origin"])
    sources.update(reference.verify_freeze(origin / "specialist/manifest.json"))
    old = json.loads((origin / "manifest.json").read_text())
    rows = json.loads((origin / "specialist/panel_private.json").read_text())
    requests = json.loads((origin / "specialist/requests_frozen.json").read_text())
    require(len(rows) == 1000, "SUBJECT_REFRESH_POPULATION", "all original 1000 cases required")
    require(
        not any(r["coverage_component_engineering_failure"] for r in rows),
        "SUBJECT_REFRESH_COVERAGE",
        "saved coverage failures cannot become safe bypasses",
    )
    eligible = {
        r["canonical_pair_id"]
        for r in rows
        if subject.subject.route(r["coverage_base"]["1"], r["payload"]) == "REVIEW_BILATERAL_SUBJECTS"
    }
    require(
        len(requests) == len(eligible)
        and {r["canonical_pair_id"] for r in requests} == eligible
        and all(r["repeat"] == 1 and r["arm"] == "candidate" for r in requests),
        "SUBJECT_REFRESH_REQUESTS",
        "the exact original admitted request bodies are repeated once",
    )
    for path in (
        Path(__file__).resolve(),
        Path(__file__).with_suffix(".md"),
        Path(proof.__file__).resolve(),
        Path(proof.candidate.__file__).resolve(),
        proof.SYSTEM,
        origin / "specialist/panel_private.json",
        origin / "specialist/requests_frozen.json",
    ):
        sources[str(path)] = sha256_file(path)
    return freeze(
        root,
        {
            "contract_version": CONTRACT,
            "stage": "fresh_subject_component_of_fixed_proof_pipeline",
            "population": 1000,
            "repeat_count": 1,
            "logical_requests": len(requests),
            "model": old["model"],
            "endpoint": old["endpoint"],
            "transport_projection": old["transport_projection"],
            "specs": {"candidate": subject.SPEC},
            "parent": str(parent),
            "coverage_origin": str(origin),
            "main_online_calls": 0,
            "coverage_online_calls": 0,
            "reference_changed": False,
            "release_eligible": False,
            "final_action_contract": proof.candidate.CONTRACT,
            "verification_repeats": 2,
            "deadline_utc_epoch": proof.selected.base.DEADLINE,
            "sources": sources,
        },
        rows,
        requests,
    )


def prepare_verifier(origin: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    origin = origin.resolve()
    root = origin / "verification"
    require(not root.exists(), "SUBJECT_REFRESH_VERIFIER_ROOT", "new verification root required")
    sources = reference.verify_freeze(origin / "manifest.json")
    old = json.loads((origin / "manifest.json").read_text())
    require(old["final_action_contract"] == proof.candidate.CONTRACT, "SUBJECT_REFRESH_CONTRACT", "fixed action")
    require((origin / "complete.json").exists(), "SUBJECT_REFRESH_COMPLETE", "all upstream calls must finish")
    rows = json.loads((origin / "panel_private.json").read_text())
    original_requests = {r["canonical_pair_id"]: r for r in json.loads((origin / "requests_frozen.json").read_text())}
    receipts = {}
    for path in (origin / "responses").glob("*.json"):
        value = json.loads(path.read_text())
        pid = value["canonical_pair_id"]
        require(pid not in receipts and pid in original_requests, "SUBJECT_REFRESH_RECEIPTS", "unique known receipt")
        require(
            value["status"] == "VALID"
            and value["repeat"] == 1
            and value["arm"] == "candidate"
            and value["request_sha256"] == original_requests[pid]["request_sha256"],
            "SUBJECT_REFRESH_VALID",
            "failed or mismatched upstream proposals cannot be silently bypassed",
        )
        receipts[pid] = value
        sources[str(path)] = sha256_file(path)
    require(receipts.keys() == original_requests.keys(), "SUBJECT_REFRESH_MISSING", "every upstream receipt required")
    for row in rows:
        row["original_main_public"] = deepcopy(row["main_public"])
        row["main_public"] = deepcopy(row["coverage_base"]["1"])
        base, payload = row["main_public"], row["payload"]
        if subject.subject.route(base, payload) != "REVIEW_BILATERAL_SUBJECTS":
            continue
        receipt = receipts[row["canonical_pair_id"]]
        choice = receipt["raw_response"]["choices"][0]
        require(
            choice["finish_reason"] == "stop" and choice["message"]["content"] == receipt["assistant_content"],
            "SUBJECT_REFRESH_RAW",
            "complete exact fresh proposal required",
        )
        proposal = proof.selected.base.previous.strict_json(receipt["assistant_content"])
        require(
            subject.subject.apply_review(base, payload, proposal)[0] == receipt["public"],
            "SUBJECT_REFRESH_REPLAY",
            "fresh proposal must replay exactly",
        )
        payload["subject_proposal"] = proposal
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests = []
    for rep in (1, 2):
        for row in rows:
            if proof.candidate.route(row["main_public"], row["payload"]) != "VERIFY_FIXED_SUBJECT_VETO":
                continue
            body = {
                "model": "Qwen/Qwen3.8-27B-FP8",
                **proof.selected.base.previous.GENERATION,
                "messages": proof.candidate.messages(row["payload"], proof.SYSTEM.read_text()),
                "response_format": subject.transport.response_format(
                    proof.candidate.response_schema(), old["transport_projection"]
                ),
            }
            tokens = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(tokens + 6144 <= 32768, "SUBJECT_REFRESH_CONTEXT", "all original text must fit without truncation")
            requests.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    "repeat": rep,
                    "arm": "candidate",
                    "body": body,
                    "request_sha256": sha256_json(body),
                    "input_tokens": tokens,
                    "deadline_utc_epoch": proof.selected.base.DEADLINE,
                }
            )
    for name in ("manifest.json", "assessment.json", "complete.json"):
        sources[str(origin / name)] = sha256_file(origin / name)
    return freeze(
        root,
        {
            "contract_version": CONTRACT + "-verification",
            "stage": "full1000_fresh_subject_conditional_proof_verification",
            "population": 1000,
            "repeat_count": 2,
            "logical_requests": len(requests),
            "specs": {"candidate": {"adapter": proof.candidate.__name__}},
            "model": old["model"],
            "endpoint": old["endpoint"],
            "coverage_origin": old["coverage_origin"],
            "subject_origin": str(origin),
            "parent": old["parent"],
            "main_online_calls": 0,
            "coverage_online_calls": 0,
            "fresh_subject_logical_requests": len(original_requests),
            "reference_changed": False,
            "release_eligible": False,
            "deadline_utc_epoch": proof.selected.base.DEADLINE,
            "sources": sources,
        },
        rows,
        requests,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.parent is not None, "SUBJECT_REFRESH_PARENT", "reviewed conditional full trial required")
        result = prepare(args.root, args.parent)
    else:
        require(args.env_file is not None, "SUBJECT_REFRESH_ENV", "existing credential source required")
        subject.run(args.root, args.env_file)
        verification = prepare_verifier(args.root)
        if verification["logical_requests"]:
            result = proof.run(args.root / "verification", args.env_file)
        else:
            result = proof.assess(args.root / "verification")
            write_json_atomic(
                args.root / "verification/complete.json",
                {
                    "assessment_sha256": sha256_file(args.root / "verification/assessment.json"),
                    "external_attempts": 0,
                    "local_rejections": 0,
                    "http_statuses": {},
                },
            )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cells", "sources", "artifacts", "response_artifacts")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
