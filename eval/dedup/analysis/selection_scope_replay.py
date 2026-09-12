# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline scope-taxonomy counterfactual on accepted V2 selections; never a judge run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis.coverage_experiment import common, paired_checks
from eval.dedup.analysis.selection_experiment import SelectionExperiment
from eval.dedup.judging.coverage_routing import route_coverage
from eval.dedup.judging.coverage_selection import adapt_selection, compile_selection
from eval.dedup.judging.coverage_witness import _MATERIAL, COLUMN, parse_coverage
from eval.dedup.judging.local_ndd import _optional_v3_span_ledger, adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.schema_v3 import JUDGE_SCHEMA_V3, validate_judge_output_v3
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

POLICY = "dedup-selection-scope-counterfactual-v1"
PROTOCOL = Path(__file__).with_name("v06224_design.md")


def probe_scope(main: dict, selection: dict, payload: dict) -> tuple[dict, dict, dict]:
    """Taxonomy is not a semantic veto, but conflicting taxonomy cannot establish containment."""
    before = adapt_selection(main, selection, payload)
    compiled = compile_selection(selection, payload)
    review = parse_coverage(compiled, payload)
    route = route_coverage(main, payload)
    ledger = _optional_v3_span_ledger(main, payload)
    profiles = ledger[0] if ledger is not None and ledger[2] is None else {}
    non_main = profiles.get("span_content_profile_a") == profiles.get("span_content_profile_b") == "NON_MAIN_ONLY"
    scope = selection["record_scope"]
    sides = [selection[k] for k in ("a_meaning_in_b", "b_meaning_in_a")]
    mismatch = (scope == "SAME_SUBSTANTIVE_RECORD" and non_main) or (scope == "NON_MAIN_MESSAGES" and not non_main)
    checks = {
        "old_public_unresolved": before["same_duplicate_group"] == "UNRESOLVED",
        "critic_owned_positive_main": route.route == "NEEDS_COVERAGE",
        "supported_main_ledger": bool(profiles),
        "resolved_review": selection["input_status"] == "COMPLETE"
        and scope != "UNRESOLVED"
        and all(s["status"] != "UNRESOLVED" for s in sides),
        "scope_profile_mismatch": mismatch,
    }
    audit = {
        "checks": checks,
        "main_profiles": {k: profiles.get(k) for k in ("span_content_profile_a", "span_content_profile_b")},
        "critic_scope": scope,
        "direction_statuses": [s["status"] for s in sides],
        "loss_types": [s["uncovered_type"] for s in sides],
        "action": "UNCHANGED",
    }
    if not all(checks.values()):
        return before, before, audit
    losses = [s for s in sides if s["status"] == "UNCOVERED"]
    hard = next((s["uncovered_type"] for s in losses if s["uncovered_type"] in _MATERIAL), None)
    if not losses:
        chrome = any(s["coverage_mode"] in {"HARMLESS_ONLY", "MIXED"} for s in sides)
        target = {
            "same_duplicate_group": "YES",
            "a_can_replace_b": "YES",
            "b_can_replace_a": "YES",
            "relation_type": "NEAR_SURFACE",
            "material_difference": "MINOR" if chrome else "NONE",
            "primary_material_difference": "OTHER_MATERIAL" if chrome else "NONE",
            "primary_risk_factor": "NONE",
        }
        audit["action"] = "BILATERAL_COVERAGE_INDEPENDENT_OF_TAXONOMY"
    elif hard is not None:
        primary, risk = _MATERIAL[hard]
        target = {
            "same_duplicate_group": "NO",
            "a_can_replace_b": "NO",
            "b_can_replace_a": "NO",
            "relation_type": "VERSION_RELATED" if hard == "STATE_VERSION" else "RELATED_NON_DUPLICATE",
            "material_difference": "MAJOR",
            "primary_material_difference": primary,
            "primary_risk_factor": risk,
        }
        audit["action"] = "RETAINED_LOSS_INDEPENDENT_OF_TAXONOMY"
    else:
        audit["action"] = "MAIN_CONTENT_SCOPE_CONFLICT_STILL_UNRESOLVED"
        return before, before, audit
    evidence = [next(item for item in review.evidence if item["side"] == side) for side in ("A", "B")]
    for item in review.evidence:
        if item not in evidence and len(evidence) < 4:
            evidence.append(item)
    source = adapt_ndd_judge_output(main, JUDGE_SCHEMA_V3, payload=payload, record_binding_policy="v6-route")
    after = {
        **source,
        **target,
        "confidence_tier": "MEDIUM",
        "evidence": evidence,
        "reason_codes": [
            f"OFFLINE_SCOPE_POLICY:{POLICY}",
            f"OFFLINE_SCOPE_RULE:{audit['action']}",
            f"COVERAGE_A_IN_B:{sides[0]['status']}",
            f"COVERAGE_B_IN_A:{sides[1]['status']}",
            f"COVERAGE_RECORD_SCOPE:{scope}",
        ],
    }
    validate_judge_output_v3(after)
    validate_evidence_offsets(after, payload)
    return before, after, audit


def matched_selections(folder: Path, predictions: list[dict]) -> dict[str, dict]:
    """No earlier retry or merely similar response can replace the published raw row."""
    raw = {}
    for attempt in sorted(folder.glob("attempt_*")):
        number = int(attempt.name.split("_")[1])
        for row in common._read_output_rows(attempt / "output"):
            raw[(row["canonical_pair_id"], number, sha256_json(row))] = row
    selected = {}
    for p in predictions:
        if p["critic_request_status"] != "REQUESTED":
            continue
        key = p["canonical_pair_id"], p["attempts"], p["raw_output_sha256"]
        require(key in raw, "SCOPE_RAW_MISSING", "exact accepted attempt and raw digest required")
        value = raw[key][COLUMN]
        require(
            sha256_json(value) == p["selection_response_sha256"], "SCOPE_SELECTION_CHANGED", "selection digest differs"
        )
        selected[p["canonical_pair_id"]] = value
    return selected


def cell_report(labels: list[dict], rows: list[dict], baseline: list[dict], suites: dict, operations: dict) -> dict:
    require(
        common._index(labels, "labels").keys() == common._index(rows, "counterfactual").keys(),
        "SCOPE_MEMBERSHIP",
        "all original pairs remain in the score denominator",
    )
    metrics = common.evaluate_predictions(labels, rows)
    guards = common.guard_report(labels, rows, baseline, suites["negative"])
    guards["additional_critic_repair_guards"] = common.evaluate_primary_guards(suites["additional"], rows)
    gates = common.local_gates(metrics, common.evaluate_predictions(labels, baseline), guards, operations, len(labels))
    gates["checks"]["all_called_judge_retried_pairs_at_most_one_percent"] = (
        operations["called_pair_retry_rate"] is not None and operations["called_pair_retry_rate"] <= 0.01
    )
    gates["passed"] = all(gates["checks"].values())
    return {
        "metrics": metrics,
        "guards": guards,
        "local_gates": gates,
        "operations": operations,
        "operations_are_inherited_not_new_online_quality": True,
        "targets": common.evaluate_primary_guards(suites["targets"], rows),
    }


def analyze(root: Path) -> dict:
    manifest = common.validate_freeze(root)
    require((root / "run_complete.json").is_file(), "SCOPE_SOURCE_INCOMPLETE", "complete source schedule required")
    experiment = SelectionExperiment(Path(manifest["experiment_spec"]))
    _, inputs, mains, calls = experiment.frozen_inputs(root)
    labels = common._read_csv(Path(manifest["paths"]["labels"]))
    common.validate_projection(labels, common._read_csv(Path(manifest["paths"]["reference"])))
    require(len(labels) == 258, "SCOPE_LOCAL_SIZE", "all 258 original labels required")
    labels_by_id = common._index(labels, "labels")
    _, _, baseline, _ = common.load_run(Path(manifest["baseline_root"]))
    baseline = [p for p in baseline if p["canonical_pair_id"] in labels_by_id]
    suites = {
        "negative": json.loads(Path(manifest["paths"]["negative_guards"]).read_text())[
            "critic_repaired_negative_guards"
        ],
        "additional": json.loads(Path(manifest["paths"]["additional_guards"]).read_text()),
        "targets": [
            {**r, "final": {k: r[f"human_{k}"] for k in common.PRIMARY_FIELDS}}
            for r in labels
            if r["review_id"] in common.TARGETS
        ],
    }
    require(
        [len(suites[k]) for k in ("negative", "additional", "targets")] == [38, 22, 3],
        "SCOPE_GUARDS",
        "all frozen protections required",
    )
    first = common._index(inputs[1], "first inputs")
    preflights = {}
    for variant in ("control", "coverage"):
        _, preflights[variant] = experiment.replay_cell(
            root, f"preflight_{variant}", variant, [first[p] for p in calls["preflight_pair_ids"][variant]], mains
        )
    cells, predictions, proofs, changes = {}, {}, {}, {}
    for repeat in (1, 2):
        index = common._index(inputs[repeat], "original inputs")
        for variant in ("control", "coverage"):
            name = f"repeat_{repeat}_{variant}"
            rows, ops = experiment.replay_cell(root, name, variant, inputs[repeat], mains)
            require(ops["valid"] == 258 and ops["errors"] == 0, "SCOPE_SOURCE_FAILED", "all outputs must be valid")
            cells[name] = cell_report(labels, rows, baseline, suites, ops)
            predictions[name] = rows
            if variant == "control":
                continue
            raw = matched_selections(root / name, rows)
            after, audits = [], []
            for row in rows:
                pid = row["canonical_pair_id"]
                if pid not in raw:
                    after.append(row)
                    continue
                before, new, audit = probe_scope(mains[pid], raw[pid], index[pid]["payload"])
                require(
                    all(row[k] == v for k, v in before.items()), "SCOPE_SOURCE_CHANGED", "original adapter differs"
                )
                after.append({**row, **new})
                audits.append({"canonical_pair_id": pid, "review_id": labels_by_id[pid]["review_id"], **audit})
            key = f"repeat_{repeat}_counterfactual"
            cells[key] = cell_report(labels, after, baseline, suites, ops)
            predictions[key] = after
            proofs[str(repeat)] = {
                "called_pairs_checked": len(audits),
                "owned_pairs_unchanged": len(rows) - len(audits),
                "action_counts": dict(Counter(a["action"] for a in audits)),
                "rows": audits,
            }
            changes[str(repeat)] = common.primary_changes(labels, rows, after)
    repeats = {
        arm: common.primary_changes(labels, predictions[f"repeat_1_{arm}"], predictions[f"repeat_2_{arm}"])
        for arm in ("control", "coverage", "counterfactual")
    }
    return {
        "schema_version": POLICY,
        "diagnostic_only": True,
        "external_model_calls": 0,
        "reference_changed": False,
        "changes_old_run_completion": False,
        "eligible_for_release": False,
        "source_contract_digest": manifest["diagnostic_contract_digest"],
        "source_complete_sha256": sha256_file(root / "run_complete.json"),
        "implementation_artifacts": {str(p.resolve()): sha256_file(p) for p in (Path(__file__), PROTOCOL)},
        "preflights": preflights,
        "cells": cells,
        "proofs": proofs,
        "changes": changes,
        "within_arm_repeat_changes": repeats,
        "inherited_paired_checks": {
            str(r): paired_checks(cells[f"repeat_{r}_counterfactual"], cells[f"repeat_{r}_control"]) for r in (1, 2)
        },
        "repeat_changes_not_above_control": len(repeats["counterfactual"]) <= len(repeats["control"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "SCOPE_OUTPUT_EXISTS", "use a new immutable assessment path")
    result = analyze(args.root)
    write_json_atomic(args.output, result)
    print(json.dumps({"external_model_calls": 0, "checks": result["inherited_paired_checks"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
