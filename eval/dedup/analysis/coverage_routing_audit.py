# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unscored proof checks for pre-critic ownership routing; never a replacement judge run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis.critic_diagnostic import _jsonl, validate_freeze
from eval.dedup.analysis.development_diagnostic import _index
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT, route_coverage
from eval.dedup.judging.coverage_witness import CONTRACT, adapt_coverage_witness
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def unresolved_branch_probe() -> dict:
    side = {
        "status": "UNRESOLVED",
        "reviewed_unique_ids": "",
        "coverage_mode": "NOT_COVERED",
        "coverage_counterpart_ids": "",
        "coverage_explanation": "Structural branch probe, not a semantic judgment.",
        "uncovered_type": "NONE",
        "source_ids": "",
        "source_quote": "",
        "counterpart_ids": "",
        "counterpart_quote": "",
        "counterpart_relation": "NOT_APPLICABLE",
        "retention_consequence": "",
    }
    return {
        "contract_version": CONTRACT,
        "input_status": "UNRESOLVED",
        "record_scope": "UNRESOLVED",
        "anchor_a_ids": "",
        "anchor_b_ids": "",
        "scope_explanation": "Structural branch probe, not a semantic judgment.",
        "a_meaning_in_b": dict(side),
        "b_meaning_in_a": dict(side),
    }


def audit_inputs(inputs: list[dict], mains: dict) -> dict:
    require(
        _index(inputs, "routing inputs").keys() == mains.keys(),
        "ROUTING_MAIN_MEMBERSHIP",
        "each original input needs its fixed main",
    )
    rows = []
    for packet in inputs:
        pid, payload = packet["canonical_pair_id"], packet["payload"]
        route = route_coverage(mains[pid], payload)
        item = {
            "canonical_pair_id": pid,
            "payload_sha256": sha256_json(payload),
            "fixed_main_sha256": sha256_json(mains[pid]),
            "route": route.route,
        }
        if route.public_output is not None:
            old = adapt_coverage_witness(mains[pid], unresolved_branch_probe(), payload)
            require(
                old == route.public_output,
                "ROUTING_OUTPUT_CHANGED",
                "owned branch public result differs from existing arbitration",
            )
            item.update(owned_branch_probe_equal=True, public_output=route.public_output)
        rows.append(item)
    counts = Counter(r["route"] for r in rows)
    return {
        "schema_version": "dedup-coverage-routing-audit-v1",
        "routing_contract": ROUTING_CONTRACT,
        "diagnostic_only": True,
        "external_model_calls": 0,
        "reference_read_or_changed": False,
        "eligible_for_release": False,
        "changes_old_run_completion": False,
        "input_pairs": len(inputs),
        "retained_output_population": len(rows),
        "route_counts": dict(counts),
        "requires_fresh_coverage": counts["NEEDS_COVERAGE"],
        "owned_branches_checked": sum(r.get("owned_branch_probe_equal", False) for r in rows),
        "proof_scope": "Direct branch ownership plus valid synthetic UNRESOLVED probes; no proof of main accuracy or of entailment in NEEDS_COVERAGE rows",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_freeze(args.root)
    inputs = _jsonl(args.root / "input_repeat_1.jsonl")
    require(len(inputs) == 258, "ROUTING_INPUT_COUNT", "all frozen local-development pairs are required")
    result = audit_inputs(inputs, json.loads((args.root / "fixed_main.json").read_text()))
    result["source_contract_digest"] = manifest["diagnostic_contract_digest"]
    result["implementation_artifacts"] = {
        str(p.resolve()): sha256_file(p)
        for p in (Path(__file__), Path(__file__).parent.parent / "judging/coverage_routing.py")
    }
    write_json_atomic(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "implementation_artifacts"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
