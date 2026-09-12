# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A paired intact-context diagnostic with unchanged subject rules and scope."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_subject_kinds_experiment as paired
from eval.dedup.analysis import critic_subject_refresh as refresh
from eval.dedup.judging import critic_subject_contiguous as candidate
from eval.dedup.validation import require, sha256_file, sha256_json


def prepare(root: Path, parent: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "SUBJECT_CONTEXT_ROOT", "new root required")
    sources = paired.reviewed_sources(parent, "CONTIGUOUS_SUBJECT_CONTEXT_PILOT_ONLY")
    old = json.loads((parent / "manifest.json").read_text())
    origin = Path(old["subject_origin"])
    sources.update(paired.reference.verify_freeze(origin / "manifest.json"))
    settings = json.loads((origin / "manifest.json").read_text())
    rows = json.loads((origin / "panel_private.json").read_text())
    rows = [r for r in rows if r["review_id"] in paired.PANEL]
    require(len(rows) == 32, "SUBJECT_CONTEXT_PANEL", "same complete 32-case panel required")
    for row in rows:
        row["original_main_public"] = deepcopy(row["main_public"])
        row["main_public"] = deepcopy(row["coverage_base"]["1"])
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    original = {r["canonical_pair_id"]: r for r in json.loads((origin / "requests_frozen.json").read_text())}
    requests = []
    arms = {"subject_control": paired.control, "candidate": candidate}
    for rep in (1, 2):
        for row in rows:
            if candidate.route(row["main_public"], row["payload"]) != "REVIEW_BILATERAL_SUBJECTS":
                continue
            for arm, module in arms.items():
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **paired.selected.base.previous.GENERATION,
                    "messages": module.messages(row["payload"], paired.previous.SYSTEM.read_text()),
                    "response_format": paired.previous.transport.response_format(
                        module.response_schema(row["payload"]), settings["transport_projection"]
                    ),
                }
                digest = sha256_json(body)
                if arm == "subject_control":
                    require(
                        digest == original[row["canonical_pair_id"]]["request_sha256"],
                        "SUBJECT_CONTEXT_CONTROL",
                        "unchanged original subject request body required",
                    )
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "SUBJECT_CONTEXT_LENGTH", "no source truncation allowed")
                requests.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": rep,
                        "arm": arm,
                        "body": body,
                        "request_sha256": digest,
                        "input_tokens": tokens,
                        "deadline_utc_epoch": paired.selected.base.DEADLINE,
                    }
                )
    for path in (Path(__file__).resolve(), Path(__file__).with_suffix(".md"), Path(candidate.__file__).resolve()):
        sources[str(path)] = sha256_file(path)
    return refresh.freeze(
        root,
        {
            "contract_version": candidate.CONTRACT,
            "stage": "paired_intact_subject_context_pilot32",
            "population": 32,
            "repeat_count": 2,
            "logical_requests": len(requests),
            "specs": {name: {"adapter": module.__name__} for name, module in arms.items()},
            "model": settings["model"],
            "endpoint": settings["endpoint"],
            "coverage_origin": settings["coverage_origin"],
            "parent": str(parent),
            "main_online_calls": 0,
            "coverage_online_calls": 0,
            "reference_changed": False,
            "release_eligible": False,
            "action_contract": paired.control.CONTRACT,
            "proof_verifier_in_this_component_trial": False,
            "deadline_utc_epoch": paired.selected.base.DEADLINE,
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
        require(args.parent is not None, "SUBJECT_CONTEXT_PARENT", "reviewed stability iteration required")
        result = prepare(args.root, args.parent)
    else:
        require(args.env_file is not None, "SUBJECT_CONTEXT_ENV", "existing credential source required")
        result = paired.run(args.root, args.env_file)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("cells", "sources", "artifacts", "response_artifacts")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
