# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference-frozen local iteration gates and digest-matched arbitration replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.dedup.analysis.development_diagnostic import (
    _index,
    classify_primary_error,
    evaluate_primary_guards,
    load_run,
    matched_raw_outputs,
)
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight, evaluate_predictions
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.analysis.residual_audit import CONTAINMENT_IDS, PROTECTED_IDS
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, write_json_atomic


def validate_projection(labels: list[dict], reference: list[dict]) -> None:
    """An identity-only selection must not silently replace original stratum weights."""
    selected, original = _index(labels, "selection"), _index(reference, "reference")
    require(bool(selected) and selected.keys() <= original.keys(), "LOCAL_MEMBERSHIP", "unknown or empty selection")
    for pair_id, label in selected.items():
        source = original[pair_id]
        fields = (
            "review_id",
            *(f"human_{key}" for key in (*PRIMARY_FIELDS, "relation_type", "material_difference", "reason_code")),
        )
        require(
            all(label.get(key) == source[key] for key in fields) and _weight(label) == _weight(source),
            "LOCAL_REFERENCE_CHANGED",
            "selection must preserve semantic reference fields, cohorts and original weights",
            pair_id=pair_id,
        )


def _guards(labels: list[dict], ids: set[str]) -> list[dict]:
    rows = [
        {**label, "final": {key: label[f"human_{key}"] for key in PRIMARY_FIELDS}}
        for label in labels
        if label["review_id"] in ids
    ]
    require(len(rows) == len(ids), "LOCAL_GUARD_MISSING", "guard subset is incomplete")
    return rows


def guard_report(labels: list[dict], predictions: list[dict], baseline: list[dict], negatives: list[dict]) -> dict:
    baseline_by_pair = _index(baseline, "baseline")
    benign = _guards(labels, PROTECTED_IDS)

    def preserved_by_baseline(guards: list[dict]) -> list[dict]:
        return [
            guard
            for guard in guards
            if all(baseline_by_pair[guard["canonical_pair_id"]][key] == guard["final"][key] for key in PRIMARY_FIELDS)
        ]

    return {
        name: evaluate_primary_guards(guards, predictions)
        for name, guards in {
            "all_negative": negatives,
            "baseline_negative_floor": preserved_by_baseline(negatives),
            "all_benign": benign,
            "baseline_benign_floor": preserved_by_baseline(benign),
            "true_containment": _guards(labels, CONTAINMENT_IDS),
            "identical": _guards(labels, {r["review_id"] for r in labels if r["human_reason_code"] == "identical"}),
        }.items()
    }


def local_gates(candidate: dict, baseline: dict, guards: dict, operations: dict, expected: int) -> dict:
    translation_delta = (
        candidate["cohorts"]["translation"]["weighted_primary_decision_exact"]
        - baseline["cohorts"]["translation"]["weighted_primary_decision_exact"]
    )
    checks = {
        "all_valid_no_terminal_error": operations["requested"] == operations["valid"] == expected
        and operations["errors"] == 0,
        "outer_retry_at_most_one_percent": operations["retried"] / expected <= 0.01,
        **{
            key: guards[key]["passed"]
            for key in ("identical", "true_containment", "baseline_negative_floor", "baseline_benign_floor")
        },
        "translation_noninferior_three_pp": translation_delta >= -0.03,
    }
    if "additional_critic_repair_guards" in guards:
        checks["additional_critic_repair_guards"] = guards["additional_critic_repair_guards"]["passed"]
    return {"passed": all(checks.values()), "checks": checks, "translation_weighted_primary_delta": translation_delta}


def replay_predictions(
    root: Path, predictions: list[dict], payloads: list[dict], source_policy: str, target_policy: str
) -> tuple[list[dict], dict]:
    raw, digests = matched_raw_outputs(root, predictions)
    packets = _index(payloads, "payloads")
    replayed = []
    for published in predictions:
        pair_id = published["canonical_pair_id"]
        main, critic = raw[pair_id]
        payload = packets[pair_id]["payload"]
        old = adapt_ndd_judge_output(
            main,
            "dedup-judge-output-v3",
            payload=payload,
            record_binding_critic=critic,
            record_binding_policy=source_policy,
        )
        require(
            all(published[key] == value for key, value in old.items()),
            "LOCAL_REPLAY_CHANGED",
            "historical public replay changed",
            pair_id=pair_id,
        )
        new = adapt_ndd_judge_output(
            main,
            "dedup-judge-output-v3",
            payload=payload,
            record_binding_critic=critic,
            record_binding_policy=target_policy,
        )
        validate_evidence_offsets(new, payload)
        replayed.append({"canonical_pair_id": pair_id, **new})
    return replayed, digests


def compare_rows(labels: list[dict], before: list[dict], after: list[dict]) -> list[dict]:
    old, new = _index(before, "before"), _index(after, "after")
    return [
        {
            "review_id": label["review_id"],
            "canonical_pair_id": label["canonical_pair_id"],
            "sample_weight": _weight(label),
            "human_reason_code": label["human_reason_code"],
            "reference": {key: label[f"human_{key}"] for key in PRIMARY_FIELDS},
            **{
                name: {
                    **{
                        key: rows[label["canonical_pair_id"]][key]
                        for key in (*PRIMARY_FIELDS, "relation_type", "confidence_tier")
                    },
                    "error": classify_primary_error(label, rows[label["canonical_pair_id"]]),
                    "reason_codes": rows[label["canonical_pair_id"]].get("reason_codes", []),
                }
                for name, rows in (("source", old), ("candidate", new))
            },
        }
        for label in labels
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "reference", "guard-replay", "baseline-root", "source-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-policy", required=True)
    parser.add_argument("--candidate-policy")
    parser.add_argument("--additional-guards", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidate-root", type=Path)
    mode.add_argument("--replay-policy")
    args = parser.parse_args()
    require(not args.output.exists(), "LOCAL_OUTPUT_EXISTS", "use a fresh output path")
    labels, reference = _read_csv(args.labels), _read_csv(args.reference)
    validate_projection(labels, reference)
    require(len(labels) == 258, "LOCAL_MEMBERSHIP", "this protocol requires the frozen 258-pair subset")
    baseline_manifest, _, baseline, baseline_payloads = load_run(args.baseline_root)
    source_manifest, source_complete, source, payloads = load_run(args.source_root)
    ids = set(_index(labels, "labels"))
    require(ids == set(_index(source, "source")), "LOCAL_MEMBERSHIP", "source membership differs")
    baseline = [row for row in baseline if row["canonical_pair_id"] in ids]
    baseline_packets = _index(baseline_payloads, "baseline payloads")
    require(
        all(row["payload"] == baseline_packets[row["canonical_pair_id"]]["payload"] for row in payloads),
        "LOCAL_PAYLOAD_CHANGED",
        "baseline payload differs",
    )
    raw_digests = {}
    if args.replay_policy:
        candidate, raw_digests = replay_predictions(
            args.source_root, source, payloads, args.source_policy, args.replay_policy
        )
        manifest, complete = source_manifest, source_complete
    else:
        require(
            bool(args.candidate_policy), "LOCAL_POLICY_REQUIRED", "online replay requires the actual candidate policy"
        )
        manifest, complete, candidate, candidate_payloads = load_run(args.candidate_root)
        require(
            manifest["pair_selection"]["sha256"] == sha256_file(args.labels),
            "LOCAL_REFERENCE_CHANGED",
            "online selector differs",
        )
        require(ids == set(_index(candidate, "candidate")), "LOCAL_MEMBERSHIP", "candidate membership differs")
        packets = _index(payloads, "source payloads")
        require(
            all(row["payload"] == packets[row["canonical_pair_id"]]["payload"] for row in candidate_payloads),
            "LOCAL_PAYLOAD_CHANGED",
            "candidate payload differs",
        )
        replay_predictions(args.source_root, source, payloads, args.source_policy, args.source_policy)
        replay_predictions(
            args.candidate_root, candidate, candidate_payloads, args.candidate_policy, args.candidate_policy
        )
    negatives = json.loads(args.guard_replay.read_text())["critic_repaired_negative_guards"]
    require(len(negatives) == 38, "LOCAL_GUARD_MISSING", "all 38 negative guards are required")
    metrics = {
        name: evaluate_predictions(labels, rows)
        for name, rows in (("baseline", baseline), ("source", source), ("candidate", candidate))
    }
    guards = {
        name: guard_report(labels, rows, baseline, negatives)
        for name, rows in (("baseline", baseline), ("source", source), ("candidate", candidate))
    }
    if args.additional_guards:
        additional = json.loads(args.additional_guards.read_text())
        references = _index(labels, "labels")
        require(
            all(
                g["canonical_pair_id"] in references
                and all(
                    references[g["canonical_pair_id"]][f"human_{key}"] == g["final"][key] for key in PRIMARY_FIELDS
                )
                for g in additional
            ),
            "LOCAL_REFERENCE_CHANGED",
            "additional guards must agree with the unchanged reference",
        )
        for name, predictions in (("baseline", baseline), ("source", source), ("candidate", candidate)):
            guards[name]["additional_critic_repair_guards"] = evaluate_primary_guards(additional, predictions)
    require(
        guards["baseline"]["baseline_negative_floor"]["expected"] == 20
        and guards["baseline"]["baseline_benign_floor"]["expected"] == 3,
        "LOCAL_FLOOR_CHANGED",
        "the floor must remain the frozen v12 floor",
    )
    rows = compare_rows(labels, source, candidate)
    gates = local_gates(metrics["candidate"], metrics["baseline"], guards["candidate"], complete, len(labels))
    result: dict[str, Any] = {
        "schema_version": "dedup-local-iteration-v1",
        "mode": "OFFLINE_ARBITRATION_ONLY" if args.replay_policy else "ONLINE_LOCAL_DEVELOPMENT",
        "interpretation": "Error-enriched development, not full/holdout or independent human calibration. Offline replay makes zero calls and cannot approve an online prompt candidate. Frozen reference taxonomy is historical; primary tuple drives selection.",
        "eligible_for_release": False,
        "full_development_authorized_by_local_gates": not args.replay_policy and gates["passed"],
        "reference_sha256": sha256_file(args.reference),
        "labels_sha256": sha256_file(args.labels),
        "guard_sha256": sha256_file(args.guard_replay),
        "additional_guards_sha256": sha256_file(args.additional_guards) if args.additional_guards else None,
        "arbitration_policies": {
            "source": args.source_policy,
            "candidate": args.replay_policy or args.candidate_policy,
        },
        "baseline_manifest": baseline_manifest,
        "source_manifest": source_manifest,
        "candidate_manifest": manifest if not args.replay_policy else None,
        "operations": complete,
        "raw_source_digests": raw_digests,
        "metrics": metrics,
        "guards": guards,
        "local_gates": gates,
        "error_distribution": dict(Counter(row["candidate"]["error"] for row in rows)),
        "changed_primary": [
            row for row in rows if any(row["source"][key] != row["candidate"][key] for key in PRIMARY_FIELDS)
        ],
        "errors": [row for row in rows if row["candidate"]["error"] != "CORRECT"],
    }
    write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("mode", "error_distribution", "local_gates", "full_development_authorized_by_local_gates")
            }
        )
    )
    print(json.dumps({"weighted": metrics["candidate"]["weighted"], "guards": guards["candidate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
