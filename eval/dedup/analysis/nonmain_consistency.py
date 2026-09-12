# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline V0.6.2.19 equivalence/extension consistency probe, not a production policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dedup.analysis.critic_diagnostic import _jsonl, validate_freeze
from eval.dedup.analysis.development_diagnostic import _index, evaluate_primary_guards, load_run
from eval.dedup.analysis.judge_calibration import evaluate_predictions
from eval.dedup.analysis.local_iteration import compare_rows, guard_report, local_gates
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.analysis.presentation_diagnostic import presentation_record, replay_cell
from eval.dedup.judging.local_ndd import (
    RECORD_BINDING_CRITIC_COLUMN,
    _optional_record_binding_critic,
    _optional_v3_span_ledger,
    adapt_ndd_judge_output,
)
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.record_scope import complete_visible_equality, parse_record_scope
from eval.dedup.judging.retained_conflict import parse_retained_conflict
from eval.dedup.judging.schema_v3 import JUDGE_SCHEMA_V3, unresolved_judge_output_v3, validate_judge_output_v3
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

ANALYSIS = Path(__file__).resolve().parent
RUNS = Path("/raid/hfang/ihb/runs")
ROOTS = (
    RUNS / "v0.6.2.16-critic-diagnostic",
    RUNS / "v0.6.2.17-presentation-diagnostic",
    RUNS / "v0.6.2.18-local-context-diagnostic",
)
RULE = "NON_MAIN_EQUIVALENCE_EXTENSION_DISAGREEMENT"


def probe_nonmain_equivalence(main: dict, critic: dict, payload: dict) -> tuple[dict, dict, dict]:
    """A structural disagreement supports abstention, not an invented negative or direction."""
    before = adapt_ndd_judge_output(
        main, JUDGE_SCHEMA_V3, payload=payload, record_binding_critic=critic, record_binding_policy="v8"
    )
    validate_evidence_offsets(before, payload)
    aligned = presentation_record({"payload": payload})["presentation_full_available"]
    span_result = _optional_v3_span_ledger(main, payload)
    binding = _optional_record_binding_critic(critic, payload)
    scope = parse_record_scope(critic, payload)
    retained = parse_retained_conflict(critic, payload)
    ledger = span_result[0] if span_result is not None else {}
    checks = {
        "complete_aligned_nonempty_input": aligned
        and all(payload[f"document_{side}"]["text"].strip() for side in ("a", "b")),
        "not_exact_identity": not complete_visible_equality(payload),
        "main_ledger_supported": span_result is not None and span_result[2] is None,
        "both_non_main": ledger.get("span_content_profile_a")
        == ledger.get("span_content_profile_b")
        == "NON_MAIN_ONLY",
        "main_claims_complete_message_equivalence": ledger.get("span_shared_basis")
        == "VERIFIED_EQUIVALENT_NON_MAIN_MESSAGE",
        "public_bidirectional": before["a_can_replace_b"] == before["b_can_replace_a"] == "YES",
        "supported_atomic_extension": binding is not None
        and binding[0] == "ATOMIC_SAME_RECORD_EXTENSION"
        and binding[2] is None,
        "supported_bound_record": scope.score in {"SAME_SPECIFIC_RECORD", "SAME_ORGANIZATION_DESCRIPTION"}
        and not scope.issues,
        "no_retained_conflict": retained.score in {"NONE", "NOT_APPLICABLE"} and not retained.issues,
    }
    triggered = all(checks.values())
    after = before
    if triggered:
        after = unresolved_judge_output_v3()
        after["reason_codes"].append(f"OFFLINE_CONSISTENCY:{RULE}")
        validate_judge_output_v3(after)
        validate_evidence_offsets(after, payload)
    return (
        before,
        after,
        {
            "triggered": triggered,
            "checks": checks,
            "main_ledger": ledger,
            "citation_issues": {
                "main": span_result[2] if span_result is not None else "MISSING_LEDGER",
                "binding": binding[2] if binding is not None else "MISSING_BINDING",
                "record_scope": list(scope.issues),
                "retained_conflict": list(retained.issues),
            },
        },
    )


def matched_critics(root: Path, name: str, rows: list[dict]) -> dict[str, dict]:
    """Select only the raw response digest that produced each published result."""
    raw = {}
    for path in sorted((root / name).glob("attempt_*/output/*.jsonl")):
        for row in _jsonl(path):
            critic = row.get(RECORD_BINDING_CRITIC_COLUMN)
            if isinstance(critic, dict):
                raw[(row["canonical_pair_id"], sha256_json(critic))] = critic
    selected = {}
    for row in rows:
        pair_id = row["canonical_pair_id"]
        key = pair_id, row["critic_response_sha256"]
        require(key in raw, "CONSISTENCY_RAW_MISSING", "published response digest has no raw match", pair_id=pair_id)
        selected[pair_id] = raw[key]
    return selected


def score_cell(labels: list[dict], before: list[dict], after: list[dict], baseline: list[dict], suites: dict) -> dict:
    """Keep abstentions in every original denominator and preserve all guard suites."""
    keys = _index(labels, "labels").keys()
    require(
        keys == _index(before, "before").keys() == _index(after, "after").keys(),
        "CONSISTENCY_MEMBERSHIP",
        "counterfactual cannot drop or add pairs",
    )
    baseline_metrics = evaluate_predictions(labels, baseline)
    reports = {}
    for name, rows in (("source", before), ("counterfactual", after)):
        metrics = evaluate_predictions(labels, rows)
        guards = guard_report(labels, rows, baseline, suites["negative"])
        guards["additional_critic_repair_guards"] = evaluate_primary_guards(suites["additional"], rows)
        reports[name] = {
            "metrics": metrics,
            "guards": guards,
            "inherited_local_gates_not_new_online_quality": local_gates(
                metrics, baseline_metrics, guards, suites["source_operations"], len(labels)
            ),
        }
    changes = [row for row in compare_rows(labels, before, after) if row["source"] != row["candidate"]]
    return {**reports, "changes": changes}


def analyze(roots: tuple[Path, ...] = ROOTS) -> dict:
    require(len(roots) == 3, "CONSISTENCY_ROOTS", "all three historical diagnostic runs are required")
    cells, frozen, proofs = {}, {}, {}
    reference_digest = main_digest = membership_digest = payloads_digest = None
    for root, expected_cells in zip(roots, (6, 4, 4), strict=True):
        manifest = validate_freeze(root)
        require((root / "run_complete.json").is_file(), "CONSISTENCY_INCOMPLETE", "source schedule incomplete")
        require(len(manifest["schedule"]) == expected_cells, "CONSISTENCY_SCHEDULE", "source schedule differs")
        labels_path = Path(manifest["paths"]["labels"])
        labels = _read_csv(labels_path)
        labels_by_pair = _index(labels, "labels")
        inputs = _index(_jsonl(root / "input_repeat_1.jsonl"), "inputs")
        mains = json.loads((root / "fixed_main.json").read_text())
        require(
            len(inputs) == 258 and inputs.keys() == labels_by_pair.keys(), "CONSISTENCY_SIZE", "258 pairs required"
        )
        require(
            all(sha256_json(row["payload"]) == row["judge_payload_hash"] for row in inputs.values()),
            "CONSISTENCY_PAYLOAD_HASH",
            "payload digest must identify the actual original input",
        )
        current = (
            sha256_file(labels_path),
            manifest["fixed_main_sha256"],
            sha256_json(sorted(inputs)),
            sha256_json({pid: row["judge_payload_hash"] for pid, row in inputs.items()}),
        )
        if reference_digest is None:
            reference_digest, main_digest, membership_digest, payloads_digest = current
        require(
            current == (reference_digest, main_digest, membership_digest, payloads_digest),
            "CONSISTENCY_SOURCE_DRIFT",
            "all source cells must keep the same reference, main and pair membership",
        )
        _, _, baseline, _ = load_run(Path(manifest["baseline_root"]))
        baseline = [row for row in baseline if row["canonical_pair_id"] in inputs]
        suites = {
            "negative": json.loads(Path(manifest["paths"]["negative_guards"]).read_text())[
                "critic_repaired_negative_guards"
            ],
            "additional": json.loads(Path(manifest["paths"]["additional_guards"]).read_text()),
        }
        require(
            len(suites["negative"]) == 38 and len(suites["additional"]) == 22,
            "CONSISTENCY_GUARDS",
            "all frozen guards are required",
        )
        frozen[str(root)] = {
            "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
            "complete_sha256": sha256_file(root / "run_complete.json"),
        }
        for spec in manifest["schedule"]:
            name = f"repeat_{spec['repeat']}_{spec['variant']}"
            rows, operations = replay_cell(root, name)
            raw = matched_critics(root, name, rows)
            after, triggered_ids = [], []
            for row in rows:
                pid = row["canonical_pair_id"]
                old, new, check = probe_nonmain_equivalence(mains[pid], raw[pid], inputs[pid]["payload"])
                require(
                    all(row[key] == value for key, value in old.items()),
                    "CONSISTENCY_REPLAY_CHANGED",
                    "source public output or evidence changed",
                    pair_id=pid,
                )
                after.append({"canonical_pair_id": pid, **new})
                if check["triggered"]:
                    proof = {
                        "canonical_pair_id": pid,
                        "review_id": labels_by_pair[pid]["review_id"],
                        "payload_sha256": inputs[pid]["judge_payload_hash"],
                        "critic_response_sha256": row["critic_response_sha256"],
                        "original_critic": raw[pid],
                        "original_public": old,
                        "counterfactual_public": new,
                        **check,
                    }
                    digest = sha256_json(proof)
                    proofs[digest] = proof
                    triggered_ids.append({"review_id": proof["review_id"], "proof_sha256": digest})
            key = f"{root.name}/{name}"
            cells[key] = {
                "source_artifacts_sha256": sha256_file(root / name / "complete.json"),
                "source_operations_not_new_online_quality": operations,
                "triggered": triggered_ids,
                **score_cell(labels, rows, after, baseline, {**suites, "source_operations": operations}),
            }
    return {
        "schema_version": "dedup-nonmain-consistency-diagnostic-v1",
        "diagnostic_only": True,
        "exploratory_cached_response_counterfactual": True,
        "eligible_for_release": False,
        "new_model_calls": 0,
        "reference_changed": False,
        "pair_ids": 258,
        "distinct_visible_payloads": len({row["judge_payload_hash"] for row in inputs.values()}),
        "replayed_responses": sum(
            cell["source_operations_not_new_online_quality"]["valid"] for cell in cells.values()
        ),
        "reference_sha256": reference_digest,
        "fixed_main_sha256": main_digest,
        "implementation_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(ANALYSIS / "v06219_design.md"),
        "source_runs": frozen,
        "cells": cells,
        "proofs": proofs,
        "decision": "DIAGNOSTIC_ONLY_NO_AUTOMATIC_PROMOTION",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "CONSISTENCY_OUTPUT_EXISTS", "do not overwrite an observed diagnostic")
    result = analyze()
    write_json_atomic(args.output, result)
    print(json.dumps({key: result[key] for key in ("replayed_responses", "new_model_calls", "decision")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
