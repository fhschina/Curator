# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Freeze a deterministic context-budget fallback without dropping any original case."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import critic_subject_context_verify as previous
from eval.dedup.validation import require, sha256_file, sha256_json


def prepare(root: Path, parent: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, parent = root.resolve(), parent.resolve()
    require(not root.exists(), "BOUNDED_SUBJECT_ROOT", "new root required")
    sources = previous.context.paired.reviewed_sources(parent, "CONTIGUOUS_SUBJECT_FULL1000")
    require(
        json.loads((parent / "assessment.json").read_text())["proof_verifier_pilot_gate"],
        "BOUNDED_SUBJECT_GATE",
        "reviewed composed context pilot required",
    )
    origin = Path(json.loads((parent / "manifest.json").read_text())["coverage_origin"])
    sources.update(previous.proof.reference.verify_freeze(origin / "specialist/manifest.json"))
    settings = json.loads((origin / "manifest.json").read_text())
    rows = json.loads((origin / "specialist/panel_private.json").read_text())
    requests = json.loads((origin / "specialist/requests_frozen.json").read_text())
    require(len(rows) == 1000, "BOUNDED_SUBJECT_POPULATION", "all original rows required")
    by = {r["canonical_pair_id"]: r for r in rows}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    fallback = []
    for request in requests:
        row = by[request["canonical_pair_id"]]
        body = deepcopy(request["body"])
        body["messages"] = previous.context.candidate.messages(
            row["payload"], previous.proof.previous.SYSTEM.read_text()
        )
        tokens = len(
            tokenizer.apply_chat_template(
                body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
        )
        request["intact_context_input_tokens"] = tokens
        if tokens + 6144 > 32768:
            require(request["input_tokens"] + 6144 <= 32768, "BOUNDED_SUBJECT_FALLBACK", "original request must fit")
            request["presentation_route"] = "EXACT_ORIGINAL_SPAN_PRESENTATION"
            fallback.append(row["review_id"])
        else:
            request.update(
                body=body,
                request_sha256=sha256_json(body),
                input_tokens=tokens,
                presentation_route="INTACT_SOURCE_PLUS_LOCATORS",
            )
    for path in (Path(__file__).resolve(), Path(__file__).with_suffix(".md")):
        sources[str(path)] = sha256_file(path)
    return previous.refresh.freeze(
        root,
        {
            "contract_version": "dedup-contiguous-subject-budget-gated-v1",
            "stage": "full_context_budget_gated_subject_component",
            "population": 1000,
            "repeat_count": 1,
            "logical_requests": len(requests),
            "specs": {"candidate": previous.proof.previous.SPEC},
            "model": settings["model"],
            "endpoint": settings["endpoint"],
            "transport_projection": settings["transport_projection"],
            "parent": str(parent),
            "coverage_origin": str(origin),
            "main_online_calls": 0,
            "coverage_online_calls": 0,
            "reference_changed": False,
            "release_eligible": False,
            "final_action_contract": previous.proof.candidate.CONTRACT,
            "fallback_review_ids_private": fallback,
            "presentation_gate_uses": "input tokens only, never labels or predictions",
            "deadline_utc_epoch": previous.proof.selected.base.DEADLINE,
            "sources": sources,
        },
        rows,
        requests,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.parent)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
