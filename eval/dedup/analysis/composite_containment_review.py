# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Export a policy decision's review materials without relabeling or changing a judge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis.bottleneck_audit import REFERENCE, REFERENCE_SHA256, blind_packets
from eval.dedup.analysis.development_diagnostic import _index
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.payload import VISIBLE_PAYLOAD_V2, _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.judging.schema_v3 import unresolved_judge_output_v3, validate_judge_output_v3
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

POLICY = "dedup-composite-containment-policy-v1"
SPEC = Path(__file__).with_name("composite_containment_review_v1.json")
SOURCE_HASHES = {
    "ledger_1000.csv": "31f332dc230d96e184e4dee658afe54155048fcc60a8647ca27015366b5e327e",
    "review_payloads.jsonl": "aa7e26b3f64ca401d143b1e7a6e846cc4d8811e9ec0552aafe59d40acee1f2ce",
}
STATUSES = {"SUPPORTED_COMPOSITE", "NONEMPTY_GATE_REVIEW", "OTHER_BOUNDARY_REVIEW"}


def primary(direction: str) -> dict:
    require(direction in {"A_REPLACES_B", "B_REPLACES_A"}, "COMPOSITE_DIRECTION", "one safe direction required")
    return {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES" if direction == "A_REPLACES_B" else "NO",
        "b_can_replace_a": "YES" if direction == "B_REPLACES_A" else "NO",
    }


def review_packet(ledger: list[dict], payloads: list[dict], spec: dict) -> dict:
    require(
        spec["policy_version"] == POLICY
        and spec["policy_status"] == "USER_APPROVED"
        and spec["review_method"] == "PREDICTION_AWARE_AI_DEVELOPMENT_REVIEW_NOT_GOLD"
        and spec["reference_changed"] is False
        and spec["runtime_changed"] is False,
        "COMPOSITE_SCOPE",
        "this export cannot promote a review into gold or a deployed runtime",
    )
    _index(ledger, "ledger")
    require(len({r["review_id"] for r in ledger}) == len(ledger), "COMPOSITE_IDS", "unique review IDs required")
    selected = {r["review_id"]: r for r in ledger if r["cluster"] == "record_unit"}
    require(
        bool(selected) and selected.keys() == spec["cases"].keys(), "COMPOSITE_COVERAGE", "review the whole cluster"
    )
    packets = _index(payloads, "payloads")
    output, weights = [], Counter()
    for rid, row in sorted(selected.items()):
        review = spec["cases"][rid]
        require(
            set(review) == {"status", "direction", "evidence_ids", "note"}
            and review["status"] in STATUSES
            and bool(review["note"])
            and bool(review["evidence_ids"])
            and len(review["evidence_ids"]) == len(set(review["evidence_ids"])),
            "COMPOSITE_REVIEW",
            "an explicit status, rationale and unique evidence IDs are required",
        )
        require(row["canonical_pair_id"] in packets, "COMPOSITE_PAYLOAD", "missing original payload")
        payload = packets[row["canonical_pair_id"]]["payload"]
        require(sha256_json(payload) == row["payload_sha256"], "COMPOSITE_HASH", "original payload changed")
        spans = {s["span_id"]: s for s in payload["semantic_diff_evidence"]["spans"]}
        require(set(review["evidence_ids"]) <= spans.keys(), "COMPOSITE_EVIDENCE", "unknown evidence IDs")
        evidence = []
        for sid in review["evidence_ids"]:
            span = spans[sid]
            for side in ("A", "B") if span["kind"] == "SHARED" else (span["side"],):
                prefix = f"{side.lower()}_" if span["kind"] == "SHARED" else ""
                evidence.append(
                    {
                        "side": side,
                        "start_char": span[f"{prefix}start_char"],
                        "end_char": span[f"{prefix}end_char"],
                        "quote": span[f"{prefix}text"],
                    }
                )
        validate_evidence_offsets({"evidence": evidence}, payload)
        proposal = None
        if review["status"] == "SUPPORTED_COMPOSITE":
            proposal = primary(review["direction"])
            added_side = "A" if review["direction"] == "A_REPLACES_B" else "B"
            require(
                payload["long_document_evidence"]["truncated"] is False
                and payload["semantic_diff_evidence"]["status"] == "COMPLETE"
                and any(spans[s]["kind"] == "SHARED" for s in review["evidence_ids"])
                and any(spans[s]["kind"] == f"{added_side}_ONLY" for s in review["evidence_ids"]),
                "COMPOSITE_INPUT",
                "a supported review needs complete input, bilateral anchors and an addition on the stated side",
            )
        else:
            require(review["direction"] is None, "COMPOSITE_PENDING", "pending boundary reviews cannot set a label")
        old = {
            key: json.loads(row[key]) if isinstance(row[key], str) else row[key]
            for key in ("reference", "main", "final")
        }
        weights[review["status"]] += float(row["weight"])
        output.append(
            {
                "review_id": rid,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": float(row["weight"]),
                "payload_sha256": row["payload_sha256"],
                **review,
                "evidence": evidence,
                "historical_12": old,
                "proposed_primary_not_gold": proposal,
                "proposal_differs_from_reference": proposal is not None and proposal != old["reference"],
            }
        )
    return {
        "policy_version": POLICY,
        "review_method": spec["review_method"],
        "reference_changed": False,
        "runtime_changed": False,
        "external_model_calls": 0,
        "new_scores_reported": False,
        "selection": "Entire record_unit disagreement cluster from .12; not all policy-affected development pairs.",
        "counts_by_status": dict(Counter(r["status"] for r in output)),
        "weights_by_status": dict(weights),
        "proposals_differing_from_reference": [r["review_id"] for r in output if r["proposal_differs_from_reference"]],
        "cases": output,
    }


def regression_packets(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compile declared synthetic expectations, not semantic predictions from text."""
    require(
        bool(examples) and len({r["id"] for r in examples}) == len(examples),
        "COMPOSITE_EXAMPLES",
        "unique example IDs required",
    )
    packets, expectations = [], []
    for example in examples:
        for reverse in (False, True):
            pair_id = f"{example['id']}:{'ba' if reverse else 'ab'}"
            a, b = (example[k] for k in (("b", "a") if reverse else ("a", "b")))
            require(
                isinstance(a, str) and isinstance(b, str) and bool(a) and bool(b),
                "COMPOSITE_EXAMPLES",
                "visible text required",
            )
            truncated = example.get("truncated", False)
            require(isinstance(truncated, bool), "COMPOSITE_EXAMPLES", "truncation must be explicit boolean")
            payload = {
                "payload_schema_version": VISIBLE_PAYLOAD_V2,
                "document_a": {"text": a},
                "document_b": {"text": b},
                "long_document_evidence": {"truncated": truncated, "windows": []},
                "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=truncated),
            }
            directions = example["directions"][::-1] if reverse else example["directions"]
            require(len(directions) == 2, "COMPOSITE_DIRECTION", "two explicit directions required")
            same = "UNRESOLVED" if "UNRESOLVED" in directions else "YES" if "YES" in directions else "NO"
            expected = {
                "same_duplicate_group": same,
                "a_can_replace_b": directions[0],
                "b_can_replace_a": directions[1],
                "relation_type": example["relation"],
                "material_difference": example["material"],
                "primary_material_difference": example["primary_difference"],
            }
            # The scoring target excludes unspecified overlap/risk taxonomy. Complete
            # this temporary object only to reuse the public semantic consistency checks.
            check = unresolved_judge_output_v3() | expected
            if same != "UNRESOLVED":
                check.update(
                    dominant_overlap_source="NONE",
                    primary_risk_factor="NONE",
                    confidence_tier="MEDIUM",
                    reason_codes=[],
                    evidence=[
                        {"side": s, "start_char": 0, "end_char": len(t[:240]), "quote": t[:240]}
                        for s, t in (("A", a), ("B", b))
                    ],
                )
            require(
                not truncated or same == "UNRESOLVED",
                "COMPOSITE_INPUT",
                "incomplete synthetic case must stay unresolved",
            )
            validate_judge_output_v3(check)
            validate_evidence_offsets(check, payload)
            packets.append({"canonical_pair_id": pair_id, "payload": payload})
            expectations.append({"canonical_pair_id": pair_id, "expected": expected, "note": example["note"]})
    return packets, expectations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha256_file(REFERENCE) == REFERENCE_SHA256, "COMPOSITE_REFERENCE", "historical labels must remain frozen")
    for name, digest in SOURCE_HASHES.items():
        require(sha256_file(args.audit_root / name) == digest, "COMPOSITE_SOURCE", "audited source changed", file=name)
    spec = json.loads(SPEC.read_text())
    payloads = [json.loads(line) for line in (args.audit_root / "review_payloads.jsonl").read_text().splitlines()]
    report = review_packet(_read_csv(args.audit_root / "ledger_1000.csv"), payloads, spec)
    regression, expectations = regression_packets(spec["examples"])
    report["sources"] = {str(args.audit_root / name): digest for name, digest in SOURCE_HASHES.items()}
    report["sources"].update({str(SPEC): sha256_file(SPEC), str(REFERENCE): REFERENCE_SHA256})
    report["generator_sha256"] = sha256_file(__file__)
    report["synthetic_pairs"] = len(regression)
    report["synthetic_model_results_available"] = False
    write_json_atomic(args.output / "review.json", report)
    write_json_atomic(args.output / "regression_expectations_private.json", expectations)
    for name, rows, ids in (
        ("review", payloads, {r["canonical_pair_id"] for r in report["cases"]}),
        ("regression", regression, {r["canonical_pair_id"] for r in regression}),
    ):
        blind, key = blind_packets(rows, ids)
        write_text_atomic(
            args.output / f"{name}_blind.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind)
        )
        write_json_atomic(args.output / f"{name}_private_key.json", key)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "counts_by_status",
                    "weights_by_status",
                    "proposals_differing_from_reference",
                    "synthetic_pairs",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
