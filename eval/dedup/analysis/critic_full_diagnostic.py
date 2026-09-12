# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full original development diagnostics after reviewed paired critic repairs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from eval.dedup.analysis import critic_ordered_experiment as ordered
from eval.dedup.analysis import critic_paired_trial as paired
from eval.dedup.analysis import critic_resilient_execution as resilient
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def reviewed_predecessor(predecessor: Path) -> tuple[dict, dict]:
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    manifest = json.loads((predecessor / "manifest.json").read_text())
    assessment = json.loads((predecessor / "assessment.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    completion = json.loads((predecessor / "complete.json").read_text())
    errors = {r["review_id"] for c in assessment["cells"].values() for r in c["pairs"] if r["requires_cause_review"]}
    digest = sha256_file(predecessor / "assessment.json")
    require(
        manifest["population"] == 96
        and manifest["repeat_count"] == 2
        and completion["assessment_sha256"] == digest == review.get("assessment_sha256")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {})),
        "FULL_CRITIC_REVIEW",
        "completed paired trial and all assessment-bound error reviews required",
    )
    for repeat in (1, 2):
        cell = assessment["cells"][f"{repeat}/candidate"]
        by_id = {r["review_id"]: r for r in cell["pairs"]}
        require(
            cell["engineering_failures"] == 0
            and all(by_id[rid]["draft_error"] == "CORRECT" for rid in (*paired.TARGETS, *paired.GUARDS)),
            "FULL_CRITIC_PROTECTION",
            "do not expand with a clear paired target or guard failure",
        )
    require(
        review.get("next_stage") == "FULL1000_DIAGNOSTIC_NOT_RELEASE"
        and set(assessment["candidate_repeat_disagreements"]) <= set(review.get("pending_repeat_disagreements", [])),
        "FULL_CRITIC_DECISION",
        "explicit diagnostic decision and reviewed pending repeat variation required",
    )
    for name in ("assessment.json", "review_complete.json", "complete.json", "paired_assessment.json"):
        path = predecessor / name
        sources[str(path)] = sha256_file(path)
    if manifest.get("candidate_schema_property_order"):
        path = predecessor / "order_assessment.json"
        order_audit = json.loads(path.read_text())
        require(
            order_audit["assessment_sha256"] == digest
            and order_audit["order_intervention_empirically_applied"] is True,
            "FULL_CRITIC_ORDER_APPLIED",
            "an ordering candidate must show the requested order in actual completed outputs",
        )
        sources[str(path)] = sha256_file(path)
    return manifest, sources


def original_rows() -> tuple[list[dict], dict]:
    previous = selected.base.previous
    sources = previous.reference.verify_freeze(previous.REFERENCE_ROOT / "summary.json")
    mains, finals, packets, _ = previous.historical.component_replay(
        previous.historical.FULL_ROOT, "v6", main_policy="v6-route"
    )
    raw, more = previous.matched_raw_outputs(previous.historical.FULL_ROOT, finals)
    sources.update(more)
    main, final, payloads = (
        previous._index(values, name) for values, name in ((mains, "main"), (finals, "final"), (packets, "packets"))
    )
    historical = previous._index(
        previous.historical._read_csv(previous.REFERENCE_ROOT / "reference_historical.csv"), "historical"
    )
    draft = previous.historical._read_csv(previous.REFERENCE_ROOT / "reference_policy_v2_partial_draft.csv")
    reference_ids = set(previous._index(draft, "draft"))
    require(
        len(draft) == 1000 and reference_ids == main.keys() == final.keys() == payloads.keys() == raw.keys(),
        "FULL_CRITIC_MEMBERSHIP",
        "all 1,000 original labels, raw mains and payloads must match exactly",
    )
    rows = []
    for label in sorted(draft, key=lambda r: r["review_id"]):
        pid = label["canonical_pair_id"]
        rows.append(
            {
                "canonical_pair_id": pid,
                "review_id": label["review_id"],
                "payload": payloads[pid]["payload"],
                "raw_main": raw[pid][0],
                "main_public": {k: main[pid][k] for k in previous.JUDGE_FIELDS_V3},
                "saved_final": {k: final[pid][k] for k in previous.JUDGE_FIELDS_V3},
                "draft_label": label,
                "historical_label": historical[pid],
            }
        )
    return rows, sources


def prepare(root: Path, predecessor: Path, protocol: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.analysis.presentation_diagnostic import TOKENIZER

    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "FULL_CRITIC_ROOT", "new full diagnostic root required")
    old, sources = reviewed_predecessor(predecessor)
    rows, more = original_rows()
    sources.update(more)
    candidate = old["specs"]["candidate"]
    module = importlib.import_module(candidate["adapter"])
    render = {"control": selected.base.previous.renderers()["control"], "candidate": selected.renderer(candidate)}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    requests = []
    for row in rows:
        if selected.base.critic.veto.route(row["main_public"], row["payload"]) != "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            continue
        for arm in ("control", "candidate"):
            body = {
                "model": "Qwen/Qwen3.8-27B-FP8",
                **selected.base.previous.GENERATION,
                "messages": render[arm](row["payload"]),
                "response_format": transport.response_format(module.response_schema(), old["transport_projection"])
                if arm == "candidate"
                else {"type": "json_object"},
            }
            tokens = len(
                tokenizer.apply_chat_template(
                    body["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            require(tokens + 6144 <= 32768, "FULL_CRITIC_CONTEXT", "never truncate full diagnostic inputs")
            requests.append(
                {
                    "canonical_pair_id": row["canonical_pair_id"],
                    "arm": arm,
                    "repeat": 1,
                    "body": body,
                    "request_sha256": sha256_json(body),
                    "input_tokens": tokens,
                    "deadline_utc_epoch": selected.base.DEADLINE,
                }
            )
    if old.get("candidate_schema_property_order"):
        requests = [
            ordered.with_property_order(r, old["candidate_schema_property_order"]) if r["arm"] == "candidate" else r
            for r in requests
        ]
    prior_requests = json.loads((predecessor / "requests_frozen.json").read_text())
    current = {(r["canonical_pair_id"], r["arm"]): r for r in requests}
    require(
        all(
            current[(r["canonical_pair_id"], r["arm"])]["request_sha256"] == r["request_sha256"]
            for r in prior_requests
            if r["repeat"] == 1 and r["arm"] in render
        ),
        "FULL_CRITIC_REQUEST_IDENTITY",
        "all prior 96-case requests must remain byte-identical; outputs are not reused",
    )
    for path in (
        Path(__file__).resolve(),
        protocol.resolve(),
        Path(transport.__file__).resolve(),
        Path(ordered.__file__).resolve(),
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", requests)
    result = {k: v for k, v in old.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version=module.CONTRACT + "-full1000-diagnostic",
        stage="full_diagnostic",
        purpose="ORIGINAL_FULL_DEVELOPMENT_WEIGHTED_DIAGNOSTIC_NOT_RELEASE",
        population=1000,
        repeat_count=1,
        arms=["control", "candidate"],
        specs={"candidate": candidate},
        constrained_arms=["candidate"],
        immediate_predecessor=str(predecessor),
        logical_requests=len(requests),
        predecessor_old_gate_passed=False,
        clear_paired_targets_and_guards_passed=True,
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "panel_private.json", root / "requests_frozen.json")},
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def full_assessment(root: Path) -> dict:
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    assessment = json.loads((root / "assessment.json").read_text())
    rows = json.loads((root / "panel_private.json").read_text())
    require(len(rows) == 1000 and assessment["population"] == 1000, "FULL_CRITIC_SCORE", "full population only")
    candidate = assessment["cells"]["1/candidate"]
    scores = candidate["scores"]["partial_draft"]["weighted"]
    complete = assessment["responses_saved"] == assessment["logical_requests"]
    by_id = {r["review_id"]: r for r in candidate["pairs"]}
    result = {
        "assessment_sha256": sha256_file(root / "assessment.json"),
        "population": len(rows),
        "complete": complete,
        "target_weighted_pr75_met": complete
        and candidate["engineering_failures"] == 0
        and scores["duplicate_precision"] >= 0.75
        and scores["duplicate_recall"] >= 0.75,
        "primary_agreement79_met": scores["primary_decision_exact"] >= 0.79,
        "clear_target_or_guard_failures": [
            rid for rid in (*paired.TARGETS, *paired.GUARDS) if by_id[rid]["draft_error"] != "CORRECT"
        ],
        "candidate_scores": candidate["scores"],
        "fresh_legacy_control_scores": assessment["cells"]["1/control"]["scores"],
        "saved_baselines": {},
        "reference_changed": False,
        "release_eligible": False,
        "all_errors_need_review": True,
        "limitation": "One fresh full-development run, unchanged provisional reference, not holdout or independent gold. Old disputed-policy gates remain reported in assessment.json.",
    }
    for name, field in (("v06212_main", "main_public"), ("v06212_final", "saved_final")):
        predictions = [{"canonical_pair_id": r["canonical_pair_id"], **r[field]} for r in rows]
        result["saved_baselines"][name] = {
            reference: selected.base.previous.reference.score([r[label] for r in rows], predictions)
            for reference, label in (("partial_draft", "draft_label"), ("historical", "historical_label"))
        }
    write_json_atomic(root / "full_assessment.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("critic_full_diagnostic_v1.md"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.predecessor is not None, "FULL_CRITIC_PREDECESSOR", "reviewed predecessor required")
        result = prepare(args.root, args.predecessor, args.protocol)
    else:
        if args.command == "run":
            require(
                args.env_file is not None, "FULL_CRITIC_CREDENTIAL", "explicit existing credential source required"
            )
            manifest = json.loads((args.root / "manifest.json").read_text())
            if manifest.get("transport_contract") == resilient.TRANSPORT_CONTRACT:
                resilient.run(args.root, args.env_file)
                if manifest.get("candidate_schema_property_order"):
                    ordered.order_assessment(args.root)
            elif manifest.get("candidate_schema_property_order"):
                ordered.run(args.root, args.env_file)
                ordered.order_assessment(args.root)
            else:
                selected.run(args.root, args.env_file)
        else:
            selected.assess(args.root)
        result = full_assessment(args.root)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts", "specs", "saved_baselines")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
