# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline, reference-frozen cross-version accounting and prospective probe export."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.dedup.analysis import bottleneck_audit as historical
from eval.dedup.analysis import paced_result_audit as paced
from eval.dedup.analysis.development_diagnostic import _index, classify_primary_error
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

NEW_ROOT = Path("/raid/hfang/ihb/runs/v0.6.2.32-paced-development")
HERE = Path(__file__).parent
VIEWS = ("old_main", "old_final", "new_main", "new_final")
TEMPLATE_OVERLAPS = {
    "COOKIE_CONSENT",
    "LEGAL_POLICY_TEMPLATE",
    "SHARED_PAGE_TEMPLATE",
    "ERROR_AUTH_PAYWALL",
    "SITE_CHROME",
}


def transition(before: str, after: str) -> str:
    if before == after == "CORRECT":
        return "STABLE_CORRECT"
    if before == "CORRECT":
        return "REGRESSED"
    if after == "CORRECT":
        return "IMPROVED"
    return "PERSISTENT_ERROR"


def observed_stage(errors: dict[str, str]) -> str:
    """An observable four-way pattern, not a causal attribution to prompt changes."""
    if errors["old_final"] == "CORRECT" and errors["new_final"] != "CORRECT":
        if errors["new_main"] == "CORRECT":
            return "NEW_POSTPROCESSING_REGRESSION"
        if errors["old_main"] != "CORRECT":
            return "PRIOR_POSTPROCESSING_REPAIR_NOT_RETAINED"
        return "NEW_MAIN_DISAGREEMENT"
    if errors["old_final"] != "CORRECT" and errors["new_final"] == "CORRECT":
        return "NEW_MAIN_CORRECT_FINAL_RETAINED" if errors["new_main"] == "CORRECT" else "NEW_POSTPROCESSING_REPAIR"
    return "NO_FINAL_CORRECTNESS_TRANSITION"


def confusion(label: dict, prediction: dict) -> dict:
    truth, answer = label["human_same_duplicate_group"], prediction["same_duplicate_group"]
    return {
        "tp": int(truth == answer == "YES"),
        "fp": int(truth == "NO" and answer == "YES"),
        "fn": int(truth == "YES" and answer != "YES"),
    }


def build_ledger(
    labels: list[dict], predictions: dict[str, list[dict]], payloads: list[dict], old_reviews: dict, composite: dict
) -> tuple[list[dict], list[dict]]:
    refs, packets = _index(labels, "labels"), _index(payloads, "payloads")
    require(set(predictions) == set(VIEWS), "INCREMENTAL_VIEWS", "four complete component views required")
    views = {name: _index(rows, name) for name, rows in predictions.items()}
    require(
        bool(refs) and all(rows.keys() == refs.keys() for rows in [packets, *views.values()]),
        "INCREMENTAL_MEMBERSHIP",
        "no intersection joins or dropped failures",
    )
    require(
        len({r["review_id"] for r in labels}) == len(labels), "INCREMENTAL_REVIEW_IDS", "unique review IDs required"
    )
    require(set(old_reviews) <= {r["review_id"] for r in labels}, "INCREMENTAL_REVIEWS", "unknown historical review")
    conflicts = historical.exact_conflicts(labels, payloads)
    conflict_ids = {m["review_id"] for group in conflicts for m in group["members"]}
    ledger = []
    for pid, label in sorted(refs.items(), key=lambda item: item[1]["review_id"]):
        rid, weight, payload = label["review_id"], _weight(label), packets[pid]["payload"]
        actual = {name: rows[pid] for name, rows in views.items()}
        errors = {name: classify_primary_error(label, row) for name, row in actual.items()}
        review, policy = old_reviews.get(rid), composite.get(rid)
        # A review of an old failure is not evidence for the cause of a new answer.
        require(
            not review or errors["old_final"] != "CORRECT",
            "INCREMENTAL_REVIEW_SCOPE",
            "old review must match old failure",
        )
        flags = []
        if rid in conflict_ids:
            flags.append("EXACT_INPUT_REFERENCE_CONTRADICTION")
        if review and review["cause"] in {"POLICY_REFERENCE_DISPUTE", "PENDING_REVIEW"}:
            flags.append("HISTORICAL_DISPUTE_OR_PENDING")
        if policy:
            flags.append("APPROVED_COMPOSITE_POLICY_REVIEW_SCOPE")
        old_c, new_c = (confusion(label, actual[name]) for name in ("old_final", "new_final"))
        missing = [name for name, row in actual.items() if row.get("metric_only_missing_output")]
        require(
            not missing or label["human_same_duplicate_group"] != "UNRESOLVED",
            "INCREMENTAL_MISSING_CREDIT",
            "engineering failure cannot earn unresolved reference credit",
        )
        row = {
            "review_id": rid,
            "canonical_pair_id": pid,
            "weight": weight,
            "sample_stratum": label.get("sample_stratum"),
            "reference_reason": label.get("human_reason_code"),
            "reference_relation": label.get("human_relation_type"),
            "reference": {k: label[f"human_{k}"] for k in PRIMARY_FIELDS},
            "transition": transition(errors["old_final"], errors["new_final"]),
            "observed_stage_pattern": observed_stage(errors),
            "errors": errors,
            "primary": {name: {k: r[k] for k in PRIMARY_FIELDS} for name, r in actual.items()},
            "relations": {name: r.get("relation_type") for name, r in actual.items()},
            "confusion_weight_delta": {k: weight * (new_c[k] - old_c[k]) for k in old_c},
            "primary_correct_weight_delta": weight
            * (int(errors["new_final"] == "CORRECT") - int(errors["old_final"] == "CORRECT")),
            "reference_review_flags": flags,
            "historical_review": review,
            "composite_policy_review": policy,
            "current_semantic_cause_status": "NOT_ADJUDICATED_FOR_NEW_OUTPUT",
            "engineering_missing_views": missing,
            "truncated": payload.get("long_document_evidence", {}).get("truncated"),
            "diff_status": payload.get("semantic_diff_evidence", {}).get("status"),
            "payload_sha256": sha256_json(payload),
            "new_model_shared_basis_claim": actual["new_main"].get("coverage_response", {}).get("shared_basis"),
            "new_model_overlap_claim": actual["new_main"].get("dominant_overlap_source"),
            "new_main_critic_primary_equal": all(
                actual["new_main"].get(k) == actual["new_final"].get(k) for k in PRIMARY_FIELDS
            ),
            "evidence": {name: r.get("evidence", []) for name, r in actual.items()},
            "reason_codes": {name: r.get("reason_codes", []) for name, r in actual.items()},
        }
        ledger.append(row)
    return ledger, conflicts


def aggregate(ledger: list[dict], field: str) -> dict:
    bins = defaultdict(
        lambda: {
            "pairs": 0,
            "weight": 0.0,
            "primary_correct_weight_delta": 0.0,
            "tp_weight_delta": 0.0,
            "fp_weight_delta": 0.0,
            "fn_weight_delta": 0.0,
            "review_ids": [],
        }
    )
    for row in ledger:
        key = str(row[field])
        bucket = bins[key]
        bucket["pairs"] += 1
        bucket["weight"] += row["weight"]
        bucket["primary_correct_weight_delta"] += row["primary_correct_weight_delta"]
        for axis in ("tp", "fp", "fn"):
            bucket[f"{axis}_weight_delta"] += row["confusion_weight_delta"][axis]
        bucket["review_ids"].append(row["review_id"])
    return dict(sorted(bins.items()))


def summarize(labels: list[dict], predictions: dict[str, list[dict]], ledger: list[dict]) -> dict:
    cross = historical.transition_matrix(labels, predictions["old_final"], predictions["new_final"])
    cross["interpretation"] = (
        "Historical same-payload comparison; prompt, route and execution differ. Not causal or contemporaneous."
    )
    components = {
        version: historical.transition_matrix(labels, predictions[f"{version}_main"], predictions[f"{version}_final"])
        for version in ("old", "new")
    }
    total = sum(row["weight"] for row in ledger)
    delta = sum(row["primary_correct_weight_delta"] for row in ledger)
    require(
        abs(100 * delta / total - cross["net_primary_gain_pp"]) < 1e-8,
        "INCREMENTAL_ACCOUNTING",
        "primary weighted changes must close",
    )
    regressions = [r for r in ledger if r["transition"] == "REGRESSED"]
    return {
        "rows": len(ledger),
        "total_weight": total,
        "by_transition": aggregate(ledger, "transition"),
        "regression_stage_patterns": aggregate(regressions, "observed_stage_pattern"),
        "regression_reference_reasons": aggregate(regressions, "reference_reason"),
        "new_overgroup_reference_reasons": aggregate(
            [r for r in regressions if r["errors"]["new_final"] == "OVER_GROUP"], "reference_reason"
        ),
        "new_overgroup_relations": dict(
            Counter(r["relations"]["new_final"] for r in regressions if r["errors"]["new_final"] == "OVER_GROUP")
        ),
        "historical_final_comparison": cross,
        "within_run_components": components,
        "reference_review_flags": {
            flag: aggregate([r for r in ledger if flag in r["reference_review_flags"]], "transition")
            for flag in sorted({f for r in ledger for f in r["reference_review_flags"]})
        },
        "flag_warning": "Flags overlap, are incomplete review coverage, and never constitute automatic corrections or exclusions from full metrics.",
        "cause_warning": "All old reviews retain their original scope. New semantic causes are unadjudicated; model overlap/basis are claims, not gold.",
    }


def select_panel(ledger: list[dict], payloads: list[dict], spec: dict) -> tuple[list[dict], list[dict]]:
    """Freeze mechanism-enriched inputs, not inferred gold labels for the X subtask."""
    packets, selected, seen = _index(payloads, "payloads"), [], set()
    by_review = {r["review_id"]: r for r in ledger}
    eligible = [
        r
        for r in ledger
        if r["truncated"] is False
        and r["diff_status"] == "COMPLETE"
        and all(
            isinstance(packets[r["canonical_pair_id"]]["payload"][f"document_{s}"]["text"], str)
            and packets[r["canonical_pair_id"]]["payload"][f"document_{s}"]["text"]
            for s in ("a", "b")
        )
        and not r["engineering_missing_views"]
        and "EXACT_INPUT_REFERENCE_CONTRADICTION" not in r["reference_review_flags"]
    ]
    eligible_ids = {r["review_id"] for r in eligible}

    def matches(row: dict, role: str) -> bool:
        if role == "repaired_negative_protection":
            return (
                row["errors"]["old_final"] == "OVER_GROUP"
                and row["errors"]["new_final"] == "CORRECT"
                and row["new_model_overlap_claim"] in TEMPLATE_OVERLAPS
            )
        if role == "suspected_template_containment":
            return (
                row["errors"]["new_final"] == "OVER_GROUP"
                and row["relations"]["new_final"] == "CONTAINMENT"
                and row["new_model_overlap_claim"] in TEMPLATE_OVERLAPS
            )
        if role == "containment_protection":
            return row["reference"]["same_duplicate_group"] == "YES" and row["reference_relation"] == "CONTAINMENT"
        if role in {"translation_protection", "chrome_protection"}:
            return row["reference"]["same_duplicate_group"] == "YES" and row["reference_reason"] == role.removesuffix(
                "_protection"
            ).replace("chrome", "chrome_only")
        if role == "identity_role_boundary":
            return row["reference"]["same_duplicate_group"] == "NO" and row["reference_reason"] in {
                "identity_slot",
                "page_role",
            }
        message = f"Unknown selection role: {role}"
        raise ValueError(message)

    for group in spec["selection"]:
        role, count, forced = group["role"], group["count"], group["required_review_ids"]
        require(
            len(forced) <= count and len(set(forced)) == len(forced),
            "INCREMENTAL_PANEL_QUOTA",
            "invalid quota or repeated required IDs",
        )
        candidates = sorted([r for r in eligible if matches(r, role)], key=lambda r: (-r["weight"], r["review_id"]))
        require(
            set(forced) <= eligible_ids and all(matches(by_review[rid], role) for rid in forced),
            "INCREMENTAL_PANEL_REQUIRED",
            "required case does not satisfy declared mechanism/input filter",
        )
        order = [by_review[rid] for rid in forced] + [r for r in candidates if r["review_id"] not in forced]
        chosen = []
        for row in order:
            if len(chosen) == count:
                break
            payload = packets[row["canonical_pair_id"]]["payload"]
            digest = sha256_json(sorted(payload[f"document_{s}"]["text"] for s in ("a", "b")))
            if digest in seen:
                require(
                    row["review_id"] not in forced,
                    "INCREMENTAL_PANEL_DUPLICATE",
                    "required case repeats an unordered exact input",
                )
                continue
            seen.add(digest)
            chosen.append(
                {
                    "role": role,
                    "review_id": row["review_id"],
                    "canonical_pair_id": row["canonical_pair_id"],
                    "selection_basis": "PREDICTION_AWARE_MECHANISM_OR_REFERENCE_HEURISTIC_NOT_X_GOLD",
                    "payload_sha256": row["payload_sha256"],
                    "historical_reference": row["reference"],
                    "weight": row["weight"],
                    "transition": row["transition"],
                    "reference_review_flags": row["reference_review_flags"],
                    "expected_shared_substantive_x": None,
                    "annotation_status": "PENDING_INDEPENDENT_REVIEW",
                }
            )
        require(
            len(chosen) == count,
            "INCREMENTAL_PANEL_SHORTFALL",
            "do not silently substitute mechanisms or reduce denominators",
            role=role,
        )
        selected.extend(chosen)
    blind, keys = historical.blind_packets(payloads, {r["canonical_pair_id"] for r in selected})
    key_index = _index(keys, "blind keys")
    for row in selected:
        row["case_id"] = key_index[row["canonical_pair_id"]]["case_id"]
    return blind, selected


def freeze_probe(output: Path, ledger: list[dict], payloads: list[dict], spec_path: Path) -> dict:
    spec = json.loads(spec_path.read_text())
    require(
        spec["online_execution_authorized"] is False and spec["automatic_veto_enabled"] is False,
        "INCREMENTAL_PROBE_SCOPE",
        "offline freeze only, no production gate",
    )
    blind, private = select_panel(ledger, payloads, spec)
    template = (spec_path.parent / spec["system_prompt_file"]).read_text()
    schema = json.loads((spec_path.parent / spec["output_schema_file"]).read_text())
    requests = [
        {
            "case_id": row["case_id"],
            "messages": [
                {"role": "system", "content": template},
                {
                    "role": "user",
                    "content": json.dumps({"payload": row["payload"]}, ensure_ascii=False, sort_keys=True),
                },
            ],
            "output_schema": schema,
        }
        for row in blind
    ]
    write_text_atomic(
        output / "probe/inputs_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blind),
    )
    write_text_atomic(
        output / "probe/requests_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in requests),
    )
    write_json_atomic(output / "probe/private_selection.json", private)
    write_json_atomic(
        output / "probe/annotation_template.json",
        [
            {
                "case_id": row["case_id"],
                "reviewer_type": None,
                "shared_substantive_x": None,
                "content_profile_a": None,
                "content_profile_b": None,
                "evidence": [],
                "rationale": None,
                "status": "PENDING_INDEPENDENT_REVIEW",
            }
            for row in blind
        ],
    )
    manifest = {
        "schema_version": "dedup-independent-x-probe-freeze-v1",
        "protocol": spec,
        "pair_count": len(blind),
        "selection_quotas": dict(Counter(r["role"] for r in private)),
        "reference_changed": False,
        "new_judge_version": False,
        "external_model_calls": 0,
        "independent_gold_available": False,
        "selection_is_holdout": False,
        "requests_sha256": sha256_file(output / "probe/requests_blind.jsonl"),
        "frozen_files": {
            str(p.resolve()): sha256_file(p)
            for p in [
                spec_path,
                spec_path.parent / spec["system_prompt_file"],
                spec_path.parent / spec["output_schema_file"],
                *[
                    output / "probe" / name
                    for name in (
                        "inputs_blind.jsonl",
                        "requests_blind.jsonl",
                        "private_selection.json",
                        "annotation_template.json",
                    )
                ],
            ]
        },
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(output / "probe/manifest.json", manifest)
    return {
        k: manifest[k]
        for k in (
            "contract_digest",
            "pair_count",
            "selection_quotas",
            "independent_gold_available",
            "external_model_calls",
        )
    }


def validate_probe_output(value: dict, payload: dict) -> dict:
    """Validate the small subtask contract without deriving any duplicate decision."""
    required = {"shared_substantive_x", "content_profile_a", "content_profile_b", "evidence", "rationale"}
    require(isinstance(value, dict) and set(value) == required, "X_PROBE_FIELDS", "exact fields required")
    require(value["shared_substantive_x"] in ("PRESENT", "ABSENT", "UNRESOLVED"), "X_PROBE_LABEL", "unknown X label")
    profiles = [value[f"content_profile_{s}"] for s in ("a", "b")]
    require(
        all(p in ("SUBSTANTIVE_MAIN", "NON_MAIN_ONLY", "UNRESOLVED") for p in profiles),
        "X_PROBE_PROFILE",
        "unknown profile",
    )
    require(
        value["shared_substantive_x"] != "PRESENT" or profiles == ["SUBSTANTIVE_MAIN"] * 2,
        "X_PROBE_CONSISTENCY",
        "shared substantive X needs substantive content on both sides",
    )
    evidence = value["evidence"]
    require(
        isinstance(value["rationale"], str) and bool(value["rationale"].strip()) and isinstance(evidence, list),
        "X_PROBE_EXPLANATION",
        "rationale and evidence list required",
    )
    for item in evidence:
        require(
            isinstance(item, dict)
            and set(item) == {"side", "start_char", "end_char", "quote"}
            and item["side"] in ("A", "B")
            and type(item["start_char"]) is int
            and type(item["end_char"]) is int
            and isinstance(item["quote"], str)
            and bool(item["quote"]),
            "X_PROBE_EVIDENCE",
            "exact typed source evidence required",
        )
    require(
        value["shared_substantive_x"] == "UNRESOLVED" or {r["side"] for r in evidence} == {"A", "B"},
        "X_PROBE_BILATERAL",
        "resolved decisions need bilateral evidence",
    )
    validate_evidence_offsets(value, payload)
    return value


def run(output: Path, spec_path: Path) -> dict:
    require(
        sha256_file(historical.REFERENCE) == historical.REFERENCE_SHA256,
        "INCREMENTAL_REFERENCE_CHANGED",
        "reference remains frozen",
    )
    labels = _read_csv(historical.REFERENCE)
    require(len(labels) == 1000, "INCREMENTAL_POPULATION", "all original 1000 cases required")
    old_main, old_final, payloads, provenance = historical.component_replay(
        historical.FULL_ROOT, "v6", main_policy="v6-route"
    )
    manifest = paced.experiment.validate(NEW_ROOT)
    require(
        (NEW_ROOT / "assessment.json").is_file() and not (NEW_ROOT / "stopped.json").exists(),
        "INCREMENTAL_INCOMPLETE",
        "completed new run required",
    )
    inputs = paced.experiment.common._jsonl(NEW_ROOT / "input_full.jsonl")
    old_packets, new_packets = _index(payloads, "old payload"), _index(inputs, "new payload")
    require(
        old_packets.keys() == new_packets.keys()
        and all(old_packets[pid]["payload"] == new_packets[pid]["payload"] for pid in old_packets),
        "INCREMENTAL_PAYLOAD_CHANGED",
        "exact full payload equality required",
    )
    fresh = paced.replay(NEW_ROOT, "full", inputs)
    predictions = {
        "old_main": old_main,
        "old_final": old_final,
        "new_main": fresh["scoring"]["main"],
        "new_final": fresh["scoring"]["final"],
    }
    review_path, composite_path = HERE / "bottleneck_reviews_v1.json", HERE / "composite_containment_review_v1.json"
    reviews, composite = json.loads(review_path.read_text())["cases"], json.loads(composite_path.read_text())["cases"]
    ledger, conflicts = build_ledger(labels, predictions, inputs, reviews, composite)
    report = summarize(labels, predictions, ledger)
    report.update(
        {
            "schema_version": "dedup-incremental-regression-audit-v1",
            "external_model_calls": 0,
            "reference_changed": False,
            "eligible_for_release": False,
            "reference_status": manifest["reference_status"],
            "reference_sha256": historical.REFERENCE_SHA256,
            "historical_provenance": provenance,
            "new_contract_digest": manifest["contract_digest"],
            "new_terminal_ids": fresh["terminal_ids"],
            "exact_document_reference_conflicts": conflicts,
        }
    )
    write_text_atomic(output / "ledger_1000.csv", historical._csv_text(ledger))
    for name in ("REGRESSED", "IMPROVED", "PERSISTENT_ERROR"):
        write_text_atomic(
            output / f"{name.lower()}.csv", historical._csv_text([r for r in ledger if r["transition"] == name])
        )
    # The queue is not a newly adjudicated dataset and remains separate from model requests.
    queue = sorted(
        [r for r in ledger if r["transition"] != "STABLE_CORRECT"],
        key=lambda r: (r["transition"] != "REGRESSED", -r["weight"], r["review_id"]),
    )
    write_json_atomic(output / "review_queue_private.json", queue)
    blind, key = historical.blind_packets(inputs, {r["canonical_pair_id"] for r in queue})
    write_text_atomic(
        output / "review_queue_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blind),
    )
    write_json_atomic(output / "review_queue_private_key.json", key)
    report["probe"] = freeze_probe(output, ledger, inputs, spec_path)
    source_files = [
        Path(__file__),
        review_path,
        composite_path,
        historical.REFERENCE,
        HERE / "composite_containment_policy_v1.md",
        NEW_ROOT / "manifest.json",
        NEW_ROOT / "input_full.jsonl",
        NEW_ROOT / "full/main_scoring_only.jsonl",
        NEW_ROOT / "full/final_scoring_only.jsonl",
    ]
    report["sources"] = {str(p.resolve()): sha256_file(p) for p in source_files}
    report["sources"].update(provenance["raw_digests"])
    report["sources"].update({str(p): sha256_file(p) for p in sorted((NEW_ROOT / "full").rglob("*.json*"))})
    report["artifacts"] = {
        str(p): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "summary.json"
    }
    write_json_atomic(output / "summary.json", report)
    print(
        json.dumps(
            {
                "rows": len(ledger),
                "transitions": {
                    k: {"pairs": v["pairs"], "weight": v["weight"]} for k, v in report["by_transition"].items()
                },
                "probe": report["probe"],
                "external_model_calls": 0,
            }
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-spec", type=Path, default=HERE / "independent_x_probe_v1.json")
    args = parser.parse_args()
    run(args.output, args.probe_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
