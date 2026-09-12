# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Non-promotional diagnostic on the complete pre-existing critic material inventory."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def prepare(root: Path, predecessor: Path, protocol: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "MATERIAL_ROOT", "never overwrite an observed diagnostic")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    manifest = json.loads((predecessor / "manifest.json").read_text())
    assessment = json.loads((predecessor / "assessment.json").read_text())
    completion = json.loads((predecessor / "complete.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    digest = sha256_file(predecessor / "assessment.json")
    errors = {
        row["review_id"]
        for cell in assessment["cells"].values()
        for row in cell["pairs"]
        if row["requires_cause_review"]
    }
    require(
        completion["assessment_sha256"] == digest == review.get("assessment_sha256")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == "DIAGNOSTIC96_NOT_PASSED_EXPANSION_GATE",
        "MATERIAL_REVIEW",
        "complete assessment-bound cause review and explicit non-promotional revision required",
    )
    require(
        manifest["population"] == 19
        and manifest["repeat_count"] == 2
        and all(assessment["cells"][f"{i}/candidate"]["engineering_failures"] == 0 for i in (1, 2)),
        "MATERIAL_ENGINEERING",
        "do not broaden while known candidate output contract is incomplete",
    )
    inventory = selected.base.PRIOR
    sources.update(selected.base.previous.reference.verify_freeze(inventory / "manifest.json"))
    for path in (
        Path(__file__).resolve(),
        Path(transport.__file__).resolve(),
        protocol.resolve(),
        predecessor / "assessment.json",
        predecessor / "complete.json",
        predecessor / "review_complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    rows = json.loads((inventory / "panel_private.json").read_text())
    old_rows = json.loads((predecessor / "panel_private.json").read_text())
    by_id = selected.base.previous._index(rows, "inventory")
    require(
        len(rows) == 96 and sum(bool(r["in_original64"]) for r in rows) == 64,
        "MATERIAL_MEMBERSHIP",
        "all original 64 and all 96 inventory rows remain",
    )
    require(
        all(by_id[r["canonical_pair_id"]] == r for r in old_rows),
        "MATERIAL_PILOT_UNCHANGED",
        "preserve every pilot payload, prediction, label and weight",
    )
    specs = manifest["specs"]
    candidate = importlib.import_module(specs["candidate"]["adapter"])
    render = {
        "control": selected.base.previous.renderers()["control"],
        "candidate": selected.renderer(specs["candidate"]),
    }
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    frozen = []
    for repeat in (1, 2):
        for row in rows:
            if (
                selected.base.critic.veto.route(row["main_public"], row["payload"])
                != "REVIEW_POSITIVE_DIRECTIONS_ONLY"
            ):
                continue
            for arm in ("control", "candidate") if repeat == 1 else ("candidate", "control"):
                body = {
                    "model": "Qwen/Qwen3.8-27B-FP8",
                    **selected.base.previous.GENERATION,
                    "messages": render[arm](row["payload"]),
                    "response_format": transport.response_format(
                        candidate.response_schema(), manifest["transport_projection"]
                    )
                    if arm == "candidate"
                    else {"type": "json_object"},
                }
                tokens = len(
                    tokenizer.apply_chat_template(
                        body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                    )
                )
                require(tokens + 6144 <= 32768, "MATERIAL_CONTEXT", "never shorten evidence to fit")
                frozen.append(
                    {
                        "canonical_pair_id": row["canonical_pair_id"],
                        "repeat": repeat,
                        "arm": arm,
                        "body": body,
                        "request_sha256": sha256_json(body),
                        "input_tokens": tokens,
                        "deadline_utc_epoch": selected.base.DEADLINE,
                    }
                )
    old_requests = json.loads((predecessor / "requests_frozen.json").read_text())
    current = {(r["canonical_pair_id"], r["repeat"], r["arm"]): r for r in frozen}
    require(
        all(
            current[(r["canonical_pair_id"], r["repeat"], r["arm"])]["request_sha256"] == r["request_sha256"]
            for r in old_requests
            if r["arm"] in render
        ),
        "MATERIAL_REQUEST_IDENTITY",
        "diagnostic cannot silently change the critic or its existing requests",
    )
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", frozen)
    result = {k: v for k, v in manifest.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version=manifest["contract_version"] + "-diagnostic96",
        stage="diagnostic96",
        purpose="CAUSE_FINDING_NOT_PILOT_PROMOTION",
        release_eligible=False,
        predecessor_gate_passed=False,
        predecessor_next_step=assessment["next_step"],
        predecessor_repeat_disagreements=assessment["candidate_repeat_disagreements"],
        pending_adjudication={
            **manifest["pending_adjudication"],
            "H0701": "membership/gift/cookie substantive-X boundary, no label change",
            "H0760": "third-party consent attestation versus substantive-X boundary, no label change",
        },
        arms=["control", "candidate"],
        population=len(rows),
        logical_requests=len(frozen),
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "panel_private.json", root / "requests_frozen.json")},
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("critic_material_diagnostic_v1.md"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.predecessor is not None, "MATERIAL_PREDECESSOR", "reviewed source run required")
        result = prepare(args.root, args.predecessor, args.protocol)
    elif args.command == "run":
        require(args.env_file is not None, "MATERIAL_CREDENTIAL", "explicit existing credential source required")
        result = selected.run(args.root, args.env_file)
    else:
        result = selected.assess(args.root)
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
