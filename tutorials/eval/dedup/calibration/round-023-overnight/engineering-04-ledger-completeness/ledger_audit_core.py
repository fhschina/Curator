"""Read-only validation of explicit ledger/summary constraints; retain the native scoring functions."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from policy_eval import FIELDS, LAYERS, exact_index, raw_directions, read_json, read_rows, validation, write_json
from policy_tuning import agreement, audit_run

MATERIAL_DELTAS = {
    "same_record_content_extension",
    "record_identity_role_or_state_change",
    "material_non_main_message_change",
    "other_substantive_content",
}
COVERED_DELTAS = {"none", "semantically_covered", "universal_ui_or_repetition"}
VERIFIED_BASES = {"verified_substantive_record", "verified_equivalent_non_main_message"}
AUDIT_VERSION = "engineering04-ledger-coherence"


def ledger_issues(row: dict) -> list[str]:
    """A Yes cannot simultaneously assert an uncovered removed-side contribution."""
    raw = row.get("pair_semantic_judgment") or {}
    values = {key: value.get("score") for key, value in raw.items() if isinstance(value, dict)}
    issues = []
    for field, delta in [("a_can_replace_b", "span_b_delta"), ("b_can_replace_a", "span_a_delta")]:
        if values.get(field) == "yes" and values.get(delta) in MATERIAL_DELTAS:
            issues.append("MAIN_LEDGER_CONTRADICTION:" + field + ":" + delta)
    if any(values.get(field) == "yes" for field in FIELDS):
        if values.get("span_hard_conflict") not in ["none", "unresolved", None]:
            issues.append("MAIN_LEDGER_CONTRADICTION:yes_with_material_hard_conflict")
        if values.get("span_shared_basis") in ["none", "unresolved"]:
            issues.append("MAIN_LEDGER_CONTRADICTION:yes_without_verified_basis")
    if (
        values.get("span_hard_conflict") == "none"
        and values.get("span_shared_basis") in VERIFIED_BASES
        and values.get("span_translation_status") == "not_translation"
    ):
        for field, delta in [("a_can_replace_b", "span_b_delta"), ("b_can_replace_a", "span_a_delta")]:
            if values.get(field) == "no" and values.get(delta) in COVERED_DELTAS:
                issues.append("MAIN_LEDGER_CONTRADICTION:" + field + ":covered_removed_side")
    return issues


def reference_groups(root: Path, panel: str) -> tuple[dict, dict]:
    """Read the same immutable lists and pair-coherent alternatives as the native audit."""
    evaluation = read_json(root / "evaluation.json")
    accepted = {}
    if panel == "diagnostic":
        labels = read_json(root / "labels/diagnostic.json")
        groups = {
            "personal120": evaluation["fixed_personal_reference"],
            "real_consensus6": [row for row in labels if row["pair_id"] in evaluation["consensus_ids"]],
        }
    elif panel == "holdout":
        original = [
            row
            for row in read_json(root / "agent_votes.json")
            if all(row.get(field) in ["yes", "no"] for field in FIELDS)
        ]
        revised = {row["pair_id"]: row for row in read_json(root / "adjudications.json")["entries"]}
        reviewed = [
            {**row, **{field: revised.get(row["pair_id"], row).get(field, row[field]) for field in FIELDS}}
            for row in original
        ]
        accepted = {
            key: value["accepted_pair_decisions"]
            for key, value in revised.items()
            if "accepted_pair_decisions" in value
        }
        groups = {
            "development99_original": original,
            "development99_reviewed": reviewed,
            "development97_strict": [row for row in reviewed if row["pair_id"] not in accepted],
        }
    else:
        labels = read_json(root / "labels" / (panel + ".json"))
        groups = {
            name: [row for row in labels if row["pair_id"] in group["ids"]]
            for name, group in evaluation["hard_groups"].items()
            if group["panel"] == panel
        }
    return groups, accepted


def verify_projection(before: dict, after: dict) -> None:
    if (
        before["pairs"] != after["pairs"]
        or before["directions"] != after["directions"]
        or after["correct_directions"] > before["correct_directions"]
        or after["correct_pairs"] > before["correct_pairs"]
        or before["reference_wrong_allows"] != after["reference_wrong_allows"]
        or before["reference_wrong_blocks"] != after["reference_wrong_blocks"]
    ):
        msg = "Ledger invalidity must preserve denominators and raw signals and cannot increase credit"
        raise ValueError(msg)


def strict_run(root: Path, name: str, native: dict | None = None, write: bool = True) -> dict:
    """Add invalidity flags without altering any generated decision, row, label or denominator."""
    if native is None:
        from argparse import Namespace

        native = audit_run(Namespace(root=root, name=name))
    run_dir = root / "runs" / name
    audit_dir = run_dir / "audits" / AUDIT_VERSION
    rows = exact_index(read_rows(run_dir / "critics"))
    flags = {key: found for key, row in rows.items() if (found := ledger_issues(row))}
    issues = copy.deepcopy(native["issues"])
    for key, found in flags.items():
        issues[key] = sorted(set(issues.get(key, []) + found))
    groups, accepted = reference_groups(root, native["panel"])
    result = copy.deepcopy(native)
    result.update(
        native_all_input_validation=copy.deepcopy(native["all_input_validation"]),
        native_failure_counts=copy.deepcopy(native["failure_counts"]),
        ledger_issues=flags,
        strict_audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        strict_artifacts={"json": str(audit_dir / "strict_audit.json"), "csv": str(audit_dir / "strict_audit.csv")},
        issues=issues,
        failure_counts=dict(Counter(issue for found in issues.values() for issue in found)),
        all_input_validation={
            layer: dict(
                Counter(validation(row, layer, issues.get(key, []))["validation_status"] for key, row in rows.items())
            )
            for layer in LAYERS
        },
        groups={
            group: {
                layer: agreement(rows, labels, issues, layer, accepted if group == "development99_reviewed" else None)
                for layer in LAYERS
            }
            for group, labels in groups.items()
        },
    )
    for group in result["groups"]:
        for layer in LAYERS:
            before, after = native["groups"][group][layer], result["groups"][group][layer]
            verify_projection(before, after)
    if write:
        audit_dir.mkdir(parents=True, exist_ok=True)
        temporary_json = audit_dir / ("strict_audit." + str(os.getpid()) + ".json.tmp")
        write_json(temporary_json, result)
        temporary_json.replace(audit_dir / "strict_audit.json")
        records = []
        memberships = {
            key: {
                group: [{field: label[field] for field in FIELDS}]
                for group, labels in groups.items()
                for label in labels
                if label["pair_id"] == key
            }
            for key in rows
        }
        for key, choices in memberships.items():
            if "development99_reviewed" in choices and key in accepted:
                choices["development99_reviewed"] = accepted[key]
        for key, row in rows.items():
            for layer in LAYERS:
                records.append(
                    {
                        "pair_id": key,
                        "layer": layer,
                        **validation(row, layer, issues.get(key, [])),
                        **{"raw_" + field: raw_directions(row, layer).get(field) for field in FIELDS},
                        "references_by_group": json.dumps(memberships[key], sort_keys=True),
                        "ledger_issues": json.dumps(flags.get(key, [])),
                    }
                )
        temporary_csv = audit_dir / ("strict_audit." + str(os.getpid()) + ".csv.tmp")
        with temporary_csv.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary_csv.replace(audit_dir / "strict_audit.csv")
    return result
