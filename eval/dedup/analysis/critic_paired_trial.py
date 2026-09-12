# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Same-material critic trials with a fresh immediate-predecessor control."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

TARGETS = ("H0232", "H0312", "H0907")
GUARDS = (
    "H0007",
    "H0108",
    "H0126",
    "H0208",
    "H0347",
    "H0417",
    "H0419",
    "H0514",
    "H0537",
    "H0546",
    "H0606",
    "H0641",
    "H0672",
    "H0711",
    "H0810",
    "H0850",
    "H0893",
    "H0957",
    "H0998",
)


def prepare(root: Path, predecessor: Path, adapter: str, config: Path, protocol: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "PAIRED_TRIAL_ROOT", "new trial root required")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    old = json.loads((predecessor / "manifest.json").read_text())
    assessment = json.loads((predecessor / "assessment.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    completion = json.loads((predecessor / "complete.json").read_text())
    errors = {r["review_id"] for c in assessment["cells"].values() for r in c["pairs"] if r["requires_cause_review"]}
    require(
        old["population"] == 96
        and completion["assessment_sha256"] == sha256_file(predecessor / "assessment.json")
        and review["assessment_sha256"] == completion["assessment_sha256"]
        and review["all_observed_disagreements_reviewed"] is True
        and errors <= set(review["cases"]),
        "PAIRED_TRIAL_REVIEW",
        "complete same-inventory predecessor and all cause reviews required",
    )
    candidate, extra = selected.specification(adapter, config)
    sources.update(extra)
    for path in (
        Path(__file__).resolve(),
        Path(transport.__file__).resolve(),
        protocol.resolve(),
        predecessor / "assessment.json",
        predecessor / "review_complete.json",
        predecessor / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    specs = {"candidate": candidate, "encoding_control": old["specs"]["candidate"]}
    render = {
        "control": selected.base.previous.renderers()["control"],
        **{arm: selected.renderer(spec) for arm, spec in specs.items()},
    }
    rows = json.loads((predecessor / "panel_private.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    frozen = []
    arms = ("control", "encoding_control", "candidate")
    for repeat in (1, 2):
        for row in rows:
            if (
                selected.base.critic.veto.route(row["main_public"], row["payload"])
                != "REVIEW_POSITIVE_DIRECTIONS_ONLY"
            ):
                continue
            for arm in arms if repeat == 1 else tuple(reversed(arms)):
                module = importlib.import_module(specs[arm]["adapter"]) if arm in specs else None
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **selected.base.previous.GENERATION,
                    "messages": render[arm](row["payload"]),
                    "response_format": transport.response_format(module.response_schema(), old["transport_projection"])
                    if module
                    else {"type": "json_object"},
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "PAIRED_TRIAL_CONTEXT", "original input must fit without truncation")
                frozen.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "arm": arm,
                        "repeat": repeat,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                        "deadline_utc_epoch": selected.base.DEADLINE,
                    }
                )
    before = {
        (r["canonical_pair_id"], r["repeat"], r["arm"]): r
        for r in json.loads((predecessor / "requests_frozen.json").read_text())
    }
    require(
        all(
            request["request_sha256"]
            == before[
                (
                    request["canonical_pair_id"],
                    request["repeat"],
                    "candidate" if request["arm"] == "encoding_control" else "control",
                )
            ]["request_sha256"]
            for request in frozen
            if request["arm"] != "candidate"
        ),
        "PAIRED_TRIAL_CONTROL_IDENTITY",
        "immediate-predecessor and legacy controls must remain byte-identical",
    )
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", frozen)
    result = {k: v for k, v in old.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version=importlib.import_module(adapter).CONTRACT + "-paired96",
        stage="paired96",
        purpose="MATERIALITY_BOUNDARY_SINGLE_CRITERION_PAIRED_TRIAL",
        arms=list(arms),
        specs=specs,
        constrained_arms=["candidate", "encoding_control"],
        immediate_predecessor=str(predecessor),
        targets=list(TARGETS),
        guards=list(GUARDS),
        logical_requests=len(frozen),
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "panel_private.json", root / "requests_frozen.json")},
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def paired_assessment(root: Path) -> dict:
    assessment = json.loads((root / "assessment.json").read_text())
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    cells = {}
    for repeat in (1, 2):
        candidate = assessment["cells"][f"{repeat}/candidate"]
        control = assessment["cells"][f"{repeat}/encoding_control"]
        by_id = {r["review_id"]: r for r in candidate["pairs"]}
        cells[str(repeat)] = {
            "candidate_engineering_failures": candidate["engineering_failures"],
            "target_failures": [rid for rid in TARGETS if by_id[rid]["draft_error"] != "CORRECT"],
            "guard_failures": [rid for rid in GUARDS if by_id[rid]["draft_error"] != "CORRECT"],
            "weighted_primary_delta": candidate["scores"]["partial_draft"]["weighted"]["primary_decision_exact"]
            - control["scores"]["partial_draft"]["weighted"]["primary_decision_exact"],
        }
    result = {
        "assessment_sha256": sha256_file(root / "assessment.json"),
        "cells": cells,
        "candidate_repeat_disagreements": assessment["candidate_repeat_disagreements"],
        "all_errors_need_review_before_next_action": True,
        "full_development_75_gate": False,
        "release_eligible": False,
        "limitation": "Local target repair is not full-population performance; pending references and all denominators remain unchanged.",
    }
    write_json_atomic(root / "paired_assessment.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--adapter", default="eval.dedup.judging.critic_retention_v4")
    parser.add_argument("--config", type=Path, default=selected.base.previous.RESOURCES / "v06212_retention_v4.yaml")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("critic_retention_v4_protocol.md"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.predecessor is not None, "PAIRED_TRIAL_PREDECESSOR", "reviewed predecessor required")
        result = prepare(args.root, args.predecessor, args.adapter, args.config, args.protocol)
    else:
        if args.command == "run":
            require(
                args.env_file is not None, "PAIRED_TRIAL_CREDENTIAL", "explicit existing credential source required"
            )
            selected.run(args.root, args.env_file)
        else:
            selected.assess(args.root)
        result = paired_assessment(args.root)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("sources", "artifacts", "response_artifacts", "specs")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
