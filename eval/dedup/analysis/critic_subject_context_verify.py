# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compose intact-context proposals with the unchanged fixed-proof action gate."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_subject_context_trial as context
from eval.dedup.analysis import critic_subject_proof_experiment as proof
from eval.dedup.analysis import critic_subject_refresh as refresh
from eval.dedup.validation import require, sha256_file, sha256_json


def prepare(root: Path, parent: Path, *, full: bool = False) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "CONTEXT_VERIFY_ROOT", "new root required")
    sources = context.paired.reviewed_sources(
        parent, "CONTIGUOUS_SUBJECT_FULL1000" if full else "VERIFY_CONTIGUOUS_PROPOSALS_PILOT"
    )
    prior = json.loads((parent / "manifest.json").read_text())
    origin = Path(prior["coverage_origin"])
    settings = json.loads((origin / "manifest.json").read_text())
    sources.update(proof.reference.verify_freeze(origin / "specialist/manifest.json"))
    if full:
        require(
            json.loads((parent / "assessment.json").read_text())["proof_verifier_pilot_gate"],
            "CONTEXT_VERIFY_GATE",
            "reviewed composed pilot protections must pass",
        )
        rows = json.loads((origin / "specialist/panel_private.json").read_text())
        originals = json.loads((origin / "specialist/requests_frozen.json").read_text())
    else:
        rows = json.loads((parent / "panel_private.json").read_text())
        originals = None
        receipts = {}
        for path in (parent / "responses").glob("*.json"):
            receipt = json.loads(path.read_text())
            if receipt["arm"] == "candidate" and receipt["repeat"] == 1:
                require(receipt["status"] == "VALID", "CONTEXT_VERIFY_SOURCE", "all first-repeat proposals valid")
                receipts[receipt["canonical_pair_id"]] = receipt
                sources[str(path)] = sha256_file(path)
        require(len(receipts) == len(rows) == 32, "CONTEXT_VERIFY_MEMBERSHIP", "all first-repeat pilot proposals")
        for row in rows:
            receipt = receipts[row["canonical_pair_id"]]
            choice = receipt["raw_response"]["choices"][0]
            require(
                choice["finish_reason"] == "stop" and choice["message"]["content"] == receipt["assistant_content"],
                "CONTEXT_VERIFY_RAW",
                "complete original subject output required",
            )
            parsed, _, _ = proof.selected.parse_output("candidate", receipt["assistant_content"], row, prior["specs"])
            require(parsed == receipt["public"], "CONTEXT_VERIFY_REPLAY", "exact original scoped replay required")
            row["payload"]["subject_proposal"] = proof.selected.base.previous.strict_json(receipt["assistant_content"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)

    def request(row: dict, rep: int, body: dict) -> dict:
        tokens = len(
            tokenizer.apply_chat_template(
                body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        )
        require(tokens + 6144 <= 32768, "CONTEXT_VERIFY_LENGTH", "original sources must fit without truncation")
        return {
            "canonical_pair_id": row["canonical_pair_id"],
            "repeat": rep,
            "arm": "candidate",
            "body": body,
            "request_sha256": sha256_json(body),
            "input_tokens": tokens,
            "deadline_utc_epoch": proof.selected.base.DEADLINE,
        }

    requests = []
    if full:
        require(len(rows) == 1000, "CONTEXT_VERIFY_POPULATION", "all original 1000 cases required")
        by = {r["canonical_pair_id"]: r for r in rows}
        for original in originals:
            row = by[original["canonical_pair_id"]]
            body = deepcopy(original["body"])
            body["messages"] = context.candidate.messages(row["payload"], proof.previous.SYSTEM.read_text())
            requests.append(request(row, 1, body))
        specs = {"candidate": proof.previous.SPEC}
    else:
        for rep in (1, 2):
            for row in rows:
                if proof.candidate.route(row["main_public"], row["payload"]) != "VERIFY_FIXED_SUBJECT_VETO":
                    continue
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **proof.selected.base.previous.GENERATION,
                    "messages": proof.candidate.messages(row["payload"], proof.SYSTEM.read_text()),
                    "response_format": proof.previous.transport.response_format(
                        proof.candidate.response_schema(), settings["transport_projection"]
                    ),
                }
                requests.append(request(row, rep, body))
        specs = {"candidate": {"adapter": proof.candidate.__name__}}
    for path in (Path(__file__).resolve(), Path(__file__).with_suffix(".md")):
        sources[str(path)] = sha256_file(path)
    return refresh.freeze(
        root,
        {
            "contract_version": "dedup-contiguous-subject-fixed-proof-v1" + ("-full1000" if full else "-pilot32"),
            "stage": "full_contiguous_subject_component" if full else "contiguous_subject_fixed_proof_pilot",
            "population": len(rows),
            "repeat_count": 1 if full else 2,
            "logical_requests": len(requests),
            "specs": specs,
            "model": settings["model"],
            "endpoint": settings["endpoint"],
            "transport_projection": settings["transport_projection"],
            "parent": str(parent),
            "coverage_origin": str(origin),
            "main_online_calls": 0,
            "coverage_online_calls": 0,
            "reference_changed": False,
            "release_eligible": False,
            "final_action_contract": proof.candidate.CONTRACT,
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
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.parent is not None, "CONTEXT_VERIFY_PARENT", "reviewed parent required")
        result = prepare(args.root, args.parent, full=args.full)
    else:
        require(args.env_file is not None, "CONTEXT_VERIFY_ENV", "existing credential source required")
        if args.full:
            proof.previous.run(args.root, args.env_file)
            refresh.prepare_verifier(args.root)
            result = proof.run(args.root / "verification", args.env_file)
        else:
            result = proof.run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cells", "sources", "artifacts", "response_artifacts")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
