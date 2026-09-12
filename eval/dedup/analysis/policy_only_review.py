# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Record reviewed applications of an approved policy without rewriting historical gold."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis import regression_review as parent
from eval.dedup.analysis.composite_containment_review import regression_packets
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

HERE = Path(__file__).resolve().parent
REVIEW_ROOT = Path("/raid/hfang/ihb/runs/v06212-v06232-review121-v1")
POLICY = "dedup-composite-containment-policy-v2"
PROVENANCE = "PREDICTION_AWARE_AI_APPLICATION_NOT_INDEPENDENT_GOLD"


def apply_cases(spec: dict, ledger: list[dict], payloads: dict) -> tuple[list[dict], list[dict]]:
    """Validate manually declared applications; textual preservation is not a semantic judge."""
    require(
        spec["policy_version"] == POLICY
        and spec["policy_status"] == "USER_APPROVED"
        and spec["application_provenance"] == PROVENANCE
        and spec["reference_changed"] is False
        and spec["runtime_changed"] is False,
        "POLICY_ONLY_SCOPE",
        "no gold or runtime promotion",
    )
    declared = spec["supported"] + spec["deferred"]
    ids = [r["review_id"] for r in declared]
    index = {r["review_id"]: r for r in ledger}
    require(
        len(index) == len(ledger) and len(ids) == len(set(ids)) and set(ids) <= index.keys(),
        "POLICY_ONLY_MEMBERSHIP",
        "explicit unique known cases required",
    )
    supported, deferred = [], []
    for case in spec["supported"]:
        row = index[case["review_id"]]
        payload = payloads[row["canonical_pair_id"]]
        require(sha256_json(payload) == row["payload_sha256"], "POLICY_ONLY_PAYLOAD", "original input changed")
        require(
            not payload["long_document_evidence"]["truncated"],
            "POLICY_ONLY_TRUNCATED",
            "complete visible input required",
        )
        require(
            case["product_side"] in {"A", "B"} and bool(case["note"]),
            "POLICY_ONLY_DIRECTION",
            "explicit reviewed addition side required",
        )
        product_side = case["product_side"]
        policy_side = "B" if product_side == "A" else "A"
        small = payload[f"document_{policy_side.lower()}"]["text"]
        large = payload[f"document_{product_side.lower()}"]["text"]
        require(
            bool(case["policy_start"]) and case["policy_start"] in small,
            "POLICY_ONLY_START",
            "reviewed policy start must exist",
        )
        start = small.index(case["policy_start"])
        policy = small[start:]
        require(
            policy in large and small[:start] in large,
            "POLICY_ONLY_PRESERVATION",
            "complete reviewed policy and prefix must be present",
        )
        require(
            case["policy_witness"] in policy
            and bool(case["policy_witness"])
            and case["product_witness"] not in small
            and bool(case["product_witness"]),
            "POLICY_ONLY_WITNESS",
            "separate policy and product witnesses required",
        )
        evidence = [
            parent.exact_evidence(payload[f"document_{s.lower()}"]["text"], case["policy_witness"], s)
            for s in ("A", "B")
        ]
        evidence.append(parent.exact_evidence(large, case["product_witness"], product_side))
        validate_evidence_offsets({"evidence": evidence}, payload)
        directions = parent.PRIMARY[f"{product_side}_CONTAINS_{policy_side}"]
        proposal = {"a_can_replace_b": directions[0], "b_can_replace_a": directions[1], "same_duplicate_group": "YES"}
        supported.append(
            {
                **case,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": row["weight"],
                "payload_sha256": row["payload_sha256"],
                "status": "SUPPORTED_UNDER_APPROVED_POLICY_NOT_GOLD",
                "expected_shared_x_not_gold": "PRESENT",
                "proposed_primary_not_gold": proposal,
                "relation_type": "CONTAINMENT",
                "material_difference": "MAJOR",
                "primary_material_difference": "MAIN_CONTENT_ADDITION_DELETION",
                "evidence": evidence,
                "full_policy_evidence": [
                    parent.exact_evidence(t, policy, s) for s, t in ((policy_side, small), (product_side, large))
                ],
                "shared_policy_sha256": sha256_json(policy),
                "policy_only_text_sha256": sha256_json(small),
                "old_reference": row["reference"],
                "saved_primary": row["primary"],
                "old_reference_differs_from_proposal": row["reference"] != proposal,
                "in_review121": row["transition"] == "REGRESSED" and row["errors"]["new_final"] == "OVER_GROUP",
                "provenance": PROVENANCE,
                "independently_adjudicated": False,
            }
        )
    for case in spec["deferred"]:
        require(
            case["status"] in {"OUTSIDE_APPROVED_PRODUCT_SCOPE", "RECHECK_POLICY_COMPLETENESS"} and bool(case["note"]),
            "POLICY_ONLY_DEFERRED",
            "unsettled boundaries cannot acquire a forced decision",
        )
        row = index[case["review_id"]]
        deferred.append(
            {
                **case,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": row["weight"],
                "proposed_primary_not_gold": None,
                "old_reference": row["reference"],
                "provenance": PROVENANCE,
                "independently_adjudicated": False,
            }
        )
    return supported, deferred


def panel_overlay(panel: list[dict], supported: list[dict], deferred: list[dict]) -> list[dict]:
    updates = {r["review_id"]: r for r in supported + deferred}
    result = []
    for row in panel:
        change = updates.get(row["review_id"])
        suspend = bool(
            change and change["status"] in {"RECHECK_POLICY_COMPLETENESS", "SUPPORTED_UNDER_APPROVED_POLICY_NOT_GOLD"}
        )
        result.append(
            {
                "review_id": row["review_id"],
                "canonical_pair_id": row["canonical_pair_id"],
                "payload_sha256": row["payload_sha256"],
                "historical_role": row["role"],
                "historical_routing": row["routing"],
                "historical_review_flags": row["reference_review_flags"],
                "policy_application": change,
                "independent_shared_x_gold": None,
                "current_negative_guard_candidate": row["mandatory_supported_prior_repair"] and not suspend,
                "status": change["status"] if change else "UNCHANGED_PENDING_INDEPENDENT_ANNOTATION",
            }
        )
    require(
        len({r["review_id"] for r in result}) == len(panel), "POLICY_ONLY_PANEL", "retain unique original membership"
    )
    return result


def export(output: Path) -> dict:
    output = output.resolve()
    frozen_roots = (REVIEW_ROOT, parent.ROOT, parent.NEW, parent.old.FULL_ROOT)
    require(
        all(output != root and root not in output.parents for root in frozen_roots),
        "POLICY_ONLY_OUTPUT",
        "use a separate output root",
    )
    manifest = json.loads((REVIEW_ROOT / "summary.json").read_text())
    require(
        all(sha256_file(p) == h for p, h in (manifest["sources"] | manifest["artifacts"]).items()),
        "POLICY_ONLY_PARENT_CHANGED",
        "121-case review and 64-case freeze must remain unchanged",
    )
    _, payloads, _, _ = parent.sources()
    spec_path = HERE / "policy_only_review_v1.json"
    spec = json.loads(spec_path.read_text())
    supported, deferred = apply_cases(spec, parent.read_ledger(parent.ROOT), payloads)
    panel = json.loads((REVIEW_ROOT / "panel_v2/private_selection.json").read_text())
    overlay = panel_overlay(panel, supported, deferred)
    write_json_atomic(output / "applications_private.json", {"supported": supported, "deferred": deferred})
    write_json_atomic(output / "panel64_policy_overlay_private.json", overlay)
    prior_guards = manifest["supported_prior_repair_ids"]
    pending = {r["review_id"] for r in deferred if r["status"] == "RECHECK_POLICY_COMPLETENESS"}
    write_json_atomic(
        output / "development_guards.json",
        {
            "provenance": PROVENANCE,
            "active_negative_candidates": [rid for rid in prior_guards if rid not in pending],
            "suspended_negative_candidates": sorted(set(prior_guards) & pending),
            "positive_candidates": [r["review_id"] for r in supported],
            "independent_gold_available": False,
        },
    )
    parent.write_packets(output, "application_inputs", supported + deferred, payloads)
    synthetic, expectations = regression_packets(spec["examples"])
    blind, key = parent.old.blind_packets(synthetic, {r["canonical_pair_id"] for r in synthetic})
    write_text_atomic(
        output / "synthetic/inputs_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blind),
    )
    write_json_atomic(output / "synthetic/expectations_private.json", expectations)
    write_json_atomic(output / "synthetic/private_key.json", key)
    result = {
        "schema_version": "dedup-policy-only-application-freeze-v1",
        "policy_version": POLICY,
        "policy_status": "USER_APPROVED",
        "application_provenance": PROVENANCE,
        "reference_changed": False,
        "runtime_changed": False,
        "external_model_calls": 0,
        "new_scores_reported": False,
        "independently_adjudicated_pairs": 0,
        "supported_ids": [r["review_id"] for r in supported],
        "supported_review121_ids": [r["review_id"] for r in supported if r["in_review121"]],
        "supported_review121_weight": sum(r["weight"] for r in supported if r["in_review121"]),
        "deferred_counts": dict(Counter(r["status"] for r in deferred)),
        "panel_count": len(overlay),
        "parent_contract_digest": manifest["contract_digest"],
        "old_panel_inputs_sha256": sha256_file(REVIEW_ROOT / "panel_v2/inputs_blind.jsonl"),
        "execution_status": "NO_ONLINE_RUN_NEW_POLICY_REQUIRES_NEW_REQUEST_AND_ANNOTATION_FREEZE",
        "synthetic_pairs": len(synthetic),
        "semantic_model_test_performed": False,
    }
    source_files = [
        Path(__file__),
        spec_path,
        HERE / "composite_containment_policy_v2.md",
        REVIEW_ROOT / "summary.json",
    ]
    result["sources"] = (
        manifest["sources"] | manifest["artifacts"] | {str(p.resolve()): sha256_file(p) for p in source_files}
    )
    result["artifacts"] = {
        str(p): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "summary.json"
    }
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(output / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("policy_version", "supported_ids", "deferred_counts", "panel_count", "external_model_calls")
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
