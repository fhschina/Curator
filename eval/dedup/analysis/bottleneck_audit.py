# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline error-cause accounting with immutable references and component replay."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.dedup.analysis.development_diagnostic import (
    _index,
    classify_primary_error,
    load_run,
    matched_raw_outputs,
)
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight, evaluate_predictions
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import assert_blind_payload, validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

REPO = Path(__file__).resolve().parents[3]
REFERENCE = REPO / "eval/dedup/analysis/v0628_policy_reconciled_labels_1000.csv"
REFERENCE_SHA256 = "6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096"
FULL_ROOT = Path("/raid/hfang/ihb/runs/v0.6.2.12-full-development")
LOCAL_ROOT = Path("/raid/hfang/ihb/runs/v0.6.2.28-sequencing-diagnostic")
CAUSES = {
    "SEMANTIC_ERROR",
    "POLICY_REFERENCE_DISPUTE",
    "ENGINEERING_ERROR",
    "INPUT_LIMITATION",
    "PENDING_REVIEW",
}


def exact_conflicts(labels: list[dict], payloads: list[dict]) -> list[dict]:
    """Compare complete original strings; normalize A/B direction, never text meaning."""
    references, packets = _index(labels, "labels"), _index(payloads, "payloads")
    require(references.keys() == packets.keys(), "BOTTLENECK_MEMBERSHIP", "identical memberships required")
    groups = defaultdict(list)
    for pair_id, packet in packets.items():
        payload = packet["payload"]
        a, b = (payload[f"document_{side}"]["text"] for side in ("a", "b"))
        if payload.get("long_document_evidence", {}).get("truncated") is not False:
            continue
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            continue
        label = references[pair_id]
        reverse = a > b
        directions = [label[f"human_{k}"] for k in PRIMARY_FIELDS[1:]]
        if reverse:
            directions.reverse()
        groups[sha256_json(sorted([a, b]))].append(
            {
                "review_id": label["review_id"],
                "canonical_pair_id": pair_id,
                "normalized_primary": [label["human_same_duplicate_group"], *directions],
                "reversed_for_comparison": reverse,
                "payload_sha256": sha256_json(payload),
                "weight": _weight(label),
            }
        )
    result = []
    for digest, members in sorted(groups.items()):
        by_answer = Counter()
        for member in members:
            by_answer[tuple(member["normalized_primary"])] += member["weight"]
        if len(by_answer) > 1:
            result.append(
                {
                    "document_pair_sha256": digest,
                    "members": sorted(members, key=lambda r: r["review_id"]),
                    "identical_visible_payload": len({r["payload_sha256"] for r in members}) == 1,
                    "has_both_orientations": len({r["reversed_for_comparison"] for r in members}) > 1,
                    "minimum_error_weight_for_orientation_consistent_answer": sum(by_answer.values())
                    - max(by_answer.values()),
                }
            )
    return result


def component_replay(root: Path, policy: str, *, main_policy: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Keep main routing, explicitly disable the critic-dependent arbitration contract."""
    manifest, _, finals, payloads = load_run(root)
    packets = _index(payloads, "payloads")
    raw, digests = matched_raw_outputs(root, finals)
    mains = []
    for final in finals:
        pair_id = final["canonical_pair_id"]
        main, critic = raw[pair_id]
        payload = packets[pair_id]["payload"]
        replay = adapt_ndd_judge_output(
            main, "dedup-judge-output-v3", payload=payload, record_binding_critic=critic, record_binding_policy=policy
        )
        require(
            all(final[key] == value for key, value in replay.items()),
            "BOTTLENECK_REPLAY_CHANGED",
            "all original public fields must replay exactly",
            pair_id=pair_id,
        )
        public = adapt_ndd_judge_output(
            main, "dedup-judge-output-v3", payload=payload, record_binding_policy=main_policy
        )
        validate_evidence_offsets(public, payload)
        mains.append({"canonical_pair_id": pair_id, **public})
    return (
        mains,
        finals,
        payloads,
        {
            "root": str(root),
            "policy": policy,
            "main_only_policy": main_policy,
            "contract_digest": manifest["judge_contract_digest"],
            "all_public_fields_replayed": True,
            "raw_digests": digests,
            "result_sha256": sha256_file(root / "data/judge_results.jsonl"),
            "payload_sha256": sha256_file(root / "data/judge_payloads.jsonl"),
        },
    )


def transition_matrix(labels: list[dict], mains: list[dict], finals: list[dict]) -> dict:
    references, before, after = (
        _index(rows, name) for rows, name in ((labels, "labels"), (mains, "main"), (finals, "final"))
    )
    require(references.keys() == before.keys() == after.keys(), "BOTTLENECK_MEMBERSHIP", "exact joins required")
    bins = defaultdict(lambda: {"pairs": 0, "weight": 0.0, "review_ids": []})
    group_deltas = Counter()
    for pair_id, label in references.items():
        a, b = (classify_primary_error(label, rows[pair_id]) for rows in (before, after))
        key = f"{a} -> {b}"
        bins[key]["pairs"] += 1
        bins[key]["weight"] += _weight(label)
        bins[key]["review_ids"].append(label["review_id"])
        if label["human_same_duplicate_group"] == "YES":
            was, now = (rows[pair_id]["same_duplicate_group"] == "YES" for rows in (before, after))
            if was != now:
                group_deltas["lost_true_positive_weight" if was else "recovered_true_positive_weight"] += _weight(
                    label
                )
        elif label["human_same_duplicate_group"] == "NO":
            was, now = (rows[pair_id]["same_duplicate_group"] == "YES" for rows in (before, after))
            if was != now:
                group_deltas["removed_false_positive_weight" if was else "added_false_positive_weight"] += _weight(
                    label
                )
    corrected = sum(
        v["weight"] for k, v in bins.items() if not k.startswith("CORRECT ->") and k.endswith("-> CORRECT")
    )
    regressed = sum(
        v["weight"] for k, v in bins.items() if k.startswith("CORRECT ->") and not k.endswith("-> CORRECT")
    )
    return {
        "main_metrics": evaluate_predictions(labels, mains),
        "final_metrics": evaluate_predictions(labels, finals),
        "transitions": dict(sorted(bins.items())),
        "corrected_weight": corrected,
        "regressed_weight": regressed,
        "net_primary_gain_pp": 100 * (corrected - regressed) / sum(_weight(r) for r in labels),
        "group_weight_changes": dict(group_deltas),
        "interpretation": "Same responses; critic plus its arbitration effect, not pure model capability or a future-prompt forecast.",
    }


def review_ledger(
    labels: list[dict], mains: list[dict], finals: list[dict], payloads: list[dict], reviews: dict
) -> list[dict]:
    references, before, after, packets = (
        _index(rows, name)
        for rows, name in ((labels, "labels"), (mains, "main"), (finals, "final"), (payloads, "payloads"))
    )
    require(
        references.keys() == before.keys() == after.keys() == packets.keys(),
        "BOTTLENECK_MEMBERSHIP",
        "exact joins required",
    )
    error_ids = {r["review_id"] for pid, r in references.items() if classify_primary_error(r, after[pid]) != "CORRECT"}
    require(set(reviews) == error_ids, "BOTTLENECK_REVIEW_COVERAGE", "one explicit review per final primary error")
    result = []
    for pair_id, label in sorted(references.items(), key=lambda item: item[1]["review_id"]):
        public, payload = after[pair_id], packets[pair_id]["payload"]
        error = classify_primary_error(label, public)
        review = reviews.get(label["review_id"])
        if review:
            require(
                set(review) == {"cause", "cluster", "note", "evidence_ids", "locus"},
                "BOTTLENECK_REVIEW_FIELDS",
                "explicit review fields required",
            )
            require(
                review["cause"] in CAUSES and bool(review["note"]),
                "BOTTLENECK_REVIEW_CAUSE",
                "known cause and rationale required",
            )
            require(
                review["locus"] in {"MAIN", "CRITIC_OR_ADAPTER", "REFERENCE", "INPUT", "MIXED", "UNDETERMINED"},
                "BOTTLENECK_REVIEW_LOCUS",
                "known locus required",
            )
            spans = {s["span_id"]: s for s in payload.get("semantic_diff_evidence", {}).get("spans", [])}
            require(
                set(review["evidence_ids"]) <= spans.keys(),
                "BOTTLENECK_REVIEW_EVIDENCE",
                "unknown evidence ID",
                review_id=label["review_id"],
            )
            incomplete = (
                payload.get("long_document_evidence", {}).get("truncated")
                or payload.get("semantic_diff_evidence", {}).get("status") != "COMPLETE"
            )
            require(
                review["cause"] != "INPUT_LIMITATION" or bool(incomplete),
                "BOTTLENECK_INPUT_LIMIT",
                "input limitation requires recorded incomplete evidence",
            )
            require(
                bool(review["evidence_ids"]) or review["cause"] in {"INPUT_LIMITATION", "PENDING_REVIEW"},
                "BOTTLENECK_REVIEW_EVIDENCE",
                "reviewed claim needs source evidence",
            )
        result.append(
            {
                "review_id": label["review_id"],
                "canonical_pair_id": pair_id,
                "weight": _weight(label),
                "error_type": error,
                "cause": review["cause"] if review else "NOT_ERROR_NOT_SEMANTICALLY_REVIEWED",
                "review_status": "PREDICTION_AWARE_AI_DEVELOPMENT_REVIEW" if review else "NOT_REVIEWED",
                "locus": review["locus"] if review else "NOT_APPLICABLE",
                "cluster": review["cluster"] if review else "NOT_APPLICABLE",
                "note": review["note"]
                if review
                else "Agreement with frozen reference is not independent semantic verification.",
                "evidence_ids": review["evidence_ids"] if review else [],
                "payload_sha256": sha256_json(payload),
                "reference": {k: label[f"human_{k}"] for k in PRIMARY_FIELDS},
                "main": {k: before[pair_id][k] for k in PRIMARY_FIELDS},
                "final": {k: public[k] for k in PRIMARY_FIELDS},
                "main_error_type": classify_primary_error(label, before[pair_id]),
                "reason_codes": public.get("reason_codes", []),
                "truncated": payload.get("long_document_evidence", {}).get("truncated"),
                "diff_status": payload.get("semantic_diff_evidence", {}).get("status"),
            }
        )
    return result


def cause_accounting(ledger: list[dict]) -> dict:
    """Additive confusion weights, with fixed-denominator loss contributions."""
    total = sum(r["weight"] for r in ledger)
    positive = sum(r["weight"] for r in ledger if r["reference"]["same_duplicate_group"] == "YES")
    tp = sum(
        r["weight"]
        for r in ledger
        if r["reference"]["same_duplicate_group"] == r["final"]["same_duplicate_group"] == "YES"
    )
    fp = sum(
        r["weight"]
        for r in ledger
        if r["reference"]["same_duplicate_group"] == "NO" and r["final"]["same_duplicate_group"] == "YES"
    )
    bins = defaultdict(
        lambda: {
            "pairs": 0,
            "error_weight": 0.0,
            "fp_weight": 0.0,
            "fn_weight": 0.0,
            "direction_weight": 0.0,
            "unresolved_weight": 0.0,
            "review_ids": [],
        }
    )
    for row in ledger:
        if row["error_type"] == "CORRECT":
            continue
        item, w = bins[row["cause"]], row["weight"]
        item["pairs"] += 1
        item["error_weight"] += w
        item["review_ids"].append(row["review_id"])
        truth, prediction = row["reference"]["same_duplicate_group"], row["final"]["same_duplicate_group"]
        item["fp_weight"] += w if truth == "NO" and prediction == "YES" else 0
        item["fn_weight"] += w if truth == "YES" and prediction != "YES" else 0
        item["fp_pairs"] = item.get("fp_pairs", 0) + int(truth == "NO" and prediction == "YES")
        item["fn_pairs"] = item.get("fn_pairs", 0) + int(truth == "YES" and prediction != "YES")
        item["direction_weight"] += w if row["error_type"] == "DIRECTION_ONLY" else 0
        item["unresolved_weight"] += w if prediction == "UNRESOLVED" else 0
    for item in bins.values():
        item["primary_loss_pp"] = 100 * item["error_weight"] / total
        item["precision_loss_contribution_pp"] = 100 * item["fp_weight"] / (tp + fp) if tp + fp else None
        item["recall_loss_contribution_pp"] = 100 * item["fn_weight"] / positive if positive else None
    return {
        "total_weight": total,
        "reference_positive_weight": positive,
        "predicted_positive_resolved_reference_weight": tp + fp,
        "by_cause": dict(sorted(bins.items())),
        "definition": "Precision loss contributions sum to 100-P; recall loss contributions sum to 100-R. They are not additive repair gains or attribution of just the gap below 75. Direction-only errors affect primary, not duplicate P/R. Unresolved weights overlap FN weights and must not be summed again.",
    }


def oracle_sensitivity(labels: list[dict], finals: list[dict], ledger: list[dict]) -> dict:
    """Reference-directed repair bounds; never change, drop or rescore labels."""
    refs, rows = _index(labels, "labels"), _index(ledger, "ledger")
    result = {}
    for cause in sorted(CAUSES):
        repaired = [
            {**p, **{k: refs[p["canonical_pair_id"]][f"human_{k}"] for k in PRIMARY_FIELDS}}
            if rows[p["canonical_pair_id"]]["cause"] == cause
            else p
            for p in finals
        ]
        metrics = evaluate_predictions(labels, repaired)["weighted"]
        result[cause] = {k: metrics[k] for k in ("duplicate_precision", "duplicate_recall", "primary_decision_exact")}
    return {
        "not_candidate_scores": True,
        "not_label_revision": True,
        "assumption": "Oracle matches old reference on every error in ONE cause, with no other changes or regressions. Dispute row is a reference-chasing ceiling, NOT proof that disputed references are correct. Bounds cannot be added.",
        "by_cause": result,
    }


def _csv_text(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                for k, v in row.items()
            }
        )
    return output.getvalue()


def mixed_panel(labels: list[dict], payloads: list[dict], spec: dict, reviews: dict) -> tuple[list[dict], dict]:
    """Freeze a reviewed diagnostic selection without exporting answers to the reviewer."""
    by_review = {r["review_id"]: r for r in labels}
    packets = _index(payloads, "payloads")
    require(len(by_review) == len(labels), "BOTTLENECK_PANEL_ID", "unique review IDs required")
    ids = [r["review_id"] for r in spec["cases"]]
    require(
        bool(ids) and len(set(ids)) == len(ids) and set(ids) <= by_review.keys(),
        "BOTTLENECK_PANEL_ID",
        "nonempty unique known panel IDs required",
    )
    require(
        all(
            reviews.get(i, {}).get("cause") not in {"POLICY_REFERENCE_DISPUTE", "PENDING_REVIEW", "INPUT_LIMITATION"}
            for i in ids
        ),
        "BOTTLENECK_PANEL_DISPUTE",
        "do not turn disputed cases into clear probes",
    )
    conflict_ids = {m["review_id"] for group in exact_conflicts(labels, payloads) for m in group["members"]}
    require(
        not (set(ids) & conflict_ids),
        "BOTTLENECK_PANEL_DISPUTE",
        "exact reference conflicts require arbitration first",
    )
    selected, keys, seen = [], [], set()
    for case in spec["cases"]:
        label = by_review[case["review_id"]]
        payload = packets[label["canonical_pair_id"]]["payload"]
        texts = [payload[f"document_{s}"]["text"] for s in ("a", "b")]
        require(
            all(isinstance(t, str) and bool(t) for t in texts),
            "BOTTLENECK_PANEL_INPUT",
            "complete original texts required",
        )
        require(
            payload.get("long_document_evidence", {}).get("truncated") is False
            and payload.get("semantic_diff_evidence", {}).get("status") == "COMPLETE",
            "BOTTLENECK_PANEL_INPUT",
            "complete visible evidence required",
        )
        digest = sha256_json(sorted(texts))
        require(digest not in seen, "BOTTLENECK_PANEL_DUPLICATE", "one instance per unordered exact text pair")
        seen.add(digest)
        selected.append(label)
        keys.append(
            {
                **case,
                "canonical_pair_id": label["canonical_pair_id"],
                "payload_sha256": sha256_json(payload),
                "reference_primary": {k: label[f"human_{k}"] for k in PRIMARY_FIELDS},
            }
        )
    return selected, {
        "schema_version": "dedup-mixed-capability-panel-v1",
        "independent_human_confirmed": False,
        "eligible_for_version_selection": False,
        "selection": spec,
        "keys": keys,
        "reference_changed": False,
        "reference_sha256": REFERENCE_SHA256,
    }


def local_components(panel_labels: list[dict]) -> dict:
    """Strictly replay both local repeats; the fixed .14 main is not the .12 main."""
    from eval.dedup.analysis.sequencing_experiment import SequencingExperiment
    from eval.dedup.judging.coverage_routing import route_coverage

    experiment = SequencingExperiment(REPO / "eval/dedup/analysis/v06228_experiment.json")
    manifest, inputs, raw_mains, _ = experiment.frozen_inputs(LOCAL_ROOT)
    require((LOCAL_ROOT / "run_complete.json").is_file(), "BOTTLENECK_LOCAL_INCOMPLETE", "finished schedule required")
    labels = _read_csv(Path(manifest["paths"]["labels"]))
    mains = []
    for row in inputs[1]:
        pid, payload = row["canonical_pair_id"], row["payload"]
        route = route_coverage(raw_mains[pid], payload)
        public = (
            route.public_output
            if route.public_output is not None
            else adapt_ndd_judge_output(
                raw_mains[pid], "dedup-judge-output-v3", payload=payload, record_binding_policy="v6-route"
            )
        )
        mains.append({"canonical_pair_id": pid, **public})
    panel_ids = {r["canonical_pair_id"] for r in panel_labels}
    available_ids = {r["canonical_pair_id"] for r in labels}
    missing_ids = panel_ids - available_ids
    local_panel_labels = [r for r in panel_labels if r["canonical_pair_id"] in available_ids]
    require(bool(local_panel_labels), "BOTTLENECK_LOCAL_PANEL", "nonempty offline intersection required")
    result = {
        "scope": "258 enriched pairs, not a full-1000 estimate; different saved main from .12. Paired within-cell differences only.",
        "contract_digest": manifest["diagnostic_contract_digest"],
        "root": str(LOCAL_ROOT),
        "panel_requested": len(panel_labels),
        "panel_available": len(local_panel_labels),
        "panel_unavailable_review_ids": [
            r["review_id"] for r in panel_labels if r["canonical_pair_id"] in missing_ids
        ],
        "panel_scope_warning": "Report only the explicitly available intersection, not a score for the entire panel. Do not impute missing guards or launch calls.",
        "cells": {},
    }
    for repeat in (1, 2):
        for variant in ("control", "coverage"):
            name = f"repeat_{repeat}_{variant}"
            finals, ops = experiment.replay_cell(LOCAL_ROOT, name, variant, inputs[repeat], raw_mains)
            result["cells"][name] = {
                "all_pairs": transition_matrix(labels, mains, finals),
                "operations": ops,
                "mixed_panel": transition_matrix(
                    local_panel_labels,
                    [p for p in mains if p["canonical_pair_id"] in panel_ids],
                    [p for p in finals if p["canonical_pair_id"] in panel_ids],
                ),
            }
    return result


def blind_packets(payloads: list[dict], ids: set[str]) -> tuple[list[dict], list[dict]]:
    """Keep opaque per-pair IDs, even when two records have identical payloads."""
    packets = _index(payloads, "blind packets")
    require(bool(ids) and ids <= packets.keys(), "BOTTLENECK_BLIND_MEMBERSHIP", "all selected inputs required")
    blind, private = [], []
    for pair_id in sorted(ids):
        payload = packets[pair_id]["payload"]
        assert_blind_payload(payload)
        require(
            set(payload)
            <= {
                "document_a",
                "document_b",
                "long_document_evidence",
                "payload_schema_version",
                "semantic_diff_evidence",
            },
            "BOTTLENECK_BLIND_FIELDS",
            "export evidence only, not annotations or predictions",
        )
        require(
            all(set(payload[f"document_{s}"]) == {"text"} for s in ("a", "b")),
            "BOTTLENECK_BLIND_FIELDS",
            "reviewer receives original text without source metadata",
        )
        case_id = sha256_json(["bottleneck-blind-v1", pair_id, payload])
        blind.append({"case_id": case_id, "payload": payload})
        private.append({"case_id": case_id, "canonical_pair_id": pair_id, "payload_sha256": sha256_json(payload)})
    return sorted(blind, key=lambda r: r["case_id"]), private


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    require(
        sha256_file(REFERENCE) == REFERENCE_SHA256,
        "BOTTLENECK_REFERENCE_CHANGED",
        "original reference must remain frozen",
    )
    labels = _read_csv(REFERENCE)
    require(len(labels) == 1000, "BOTTLENECK_POPULATION", "full development population required")
    mains, finals, payloads, provenance = component_replay(FULL_ROOT, "v6", main_policy="v6-route")
    reviews = json.loads(args.reviews.read_text())
    ledger = review_ledger(labels, mains, finals, payloads, reviews["cases"])
    conflicts = exact_conflicts(labels, payloads)
    matrix = transition_matrix(labels, mains, finals)
    panel_labels, panel_manifest = mixed_panel(labels, payloads, json.loads(args.panel.read_text()), reviews["cases"])
    panel_ids = {r["canonical_pair_id"] for r in panel_labels}
    report = {
        "schema_version": "dedup-bottleneck-audit-v1",
        "external_model_calls": 0,
        "reference_changed": False,
        "reference_sha256": REFERENCE_SHA256,
        "reviews_sha256": sha256_file(args.reviews),
        "implementation_sha256": sha256_file(Path(__file__)),
        "scope": "Latest completed full-1000 .12; .28 local diagnostics are separate. No version selection or independent-human accuracy claim.",
        "provenance": provenance,
        "component_comparison": matrix,
        "cause_accounting": cause_accounting(ledger),
        "oracle_sensitivity": oracle_sensitivity(labels, finals, ledger),
        "exact_document_reference_conflicts": conflicts,
        "mixed_panel": transition_matrix(
            panel_labels,
            [p for p in mains if p["canonical_pair_id"] in panel_ids],
            [p for p in finals if p["canonical_pair_id"] in panel_ids],
        ),
        "panel_spec_sha256": sha256_file(args.panel),
    }
    for cause in sorted(CAUSES):
        ids = {r["canonical_pair_id"] for r in ledger if r["cause"] == cause}
        if ids:
            report.setdefault("component_on_final_error_cause_slices", {})[cause] = transition_matrix(
                [r for r in labels if r["canonical_pair_id"] in ids],
                [r for r in mains if r["canonical_pair_id"] in ids],
                [r for r in finals if r["canonical_pair_id"] in ids],
            )
    report["slice_warning"] = (
        "Cause slices are selected on FINAL errors, omit protected correct cases, and cannot estimate net critic benefit; use the full matrix or the separately reviewed mixed panel."
    )
    write_json_atomic(args.output / "local_258_components.json", local_components(panel_labels))
    write_json_atomic(args.output / "panel_manifest.json", panel_manifest)
    blind, private = blind_packets(payloads, panel_ids)
    write_json_atomic(args.output / "panel_private_key.json", private)
    write_text_atomic(
        args.output / "panel_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blind),
    )
    write_json_atomic(args.output / "summary.json", report)
    write_text_atomic(args.output / "ledger_1000.csv", _csv_text(ledger))
    write_text_atomic(args.output / "errors_124.csv", _csv_text([r for r in ledger if r["error_type"] != "CORRECT"]))
    review_ids = {r["canonical_pair_id"] for r in ledger if r["error_type"] != "CORRECT"}
    review_ids.update(m["canonical_pair_id"] for g in conflicts for m in g["members"])
    arbitration_ids = {
        r["canonical_pair_id"] for r in ledger if r["cause"] in {"POLICY_REFERENCE_DISPUTE", "PENDING_REVIEW"}
    }
    arbitration_ids.update(m["canonical_pair_id"] for g in conflicts for m in g["members"])
    arbitration_blind, arbitration_key = blind_packets(payloads, arbitration_ids)
    write_json_atomic(args.output / "arbitration_private_key.json", arbitration_key)
    write_text_atomic(
        args.output / "arbitration_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in arbitration_blind),
    )
    write_text_atomic(
        args.output / "review_payloads.jsonl",
        "".join(
            json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n"
            for p in payloads
            if p["canonical_pair_id"] in review_ids
        ),
    )
    print(
        json.dumps(
            {
                "rows": len(ledger),
                "errors": len(reviews["cases"]),
                "conflicts": len(conflicts),
                "external_model_calls": 0,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
