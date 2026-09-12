# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Evidence-bound development review of the complete incremental over-group cohort."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.dedup.analysis import bottleneck_audit as old
from eval.dedup.analysis.development_diagnostic import load_run, matched_raw_outputs
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

HERE = Path(__file__).resolve().parent
ROOT = Path("/raid/hfang/ihb/runs/v06212-v06232-incremental-v1")
NEW = Path("/raid/hfang/ihb/runs/v0.6.2.32-paced-development")
REPAIR = "PRIOR_POSTPROCESSING_REPAIR_NOT_RETAINED"
METHOD = "PREDICTION_AWARE_AI_DEVELOPMENT_REVIEW_NOT_HUMAN_GOLD"
COLUMNS = ["review_id", "assessment", "mechanism", "shared_x", "proposed_primary", "quote_a", "quote_b", "note"]
ASSESSMENTS = {"CLEAR_MODEL_ERROR", "POLICY_REFERENCE_DISPUTE", "PENDING_JUDGMENT"}
PRIMARY = {
    "NO_NO": ("NO", "NO"),
    "A_CONTAINS_B": ("YES", "NO"),
    "B_CONTAINS_A": ("NO", "YES"),
    "EQUIVALENT": ("YES", "YES"),
    "UNDECIDED": ("UNRESOLVED", "UNRESOLVED"),
}


def read_ledger(root: Path) -> list[dict]:
    with (root / "ledger_1000.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["weight"] = float(row["weight"])
        for key in ("errors", "primary", "relations", "reference", "reference_review_flags"):
            row[key] = json.loads(row[key])
    return rows


def sources() -> tuple[list[dict], dict, dict, dict]:
    summary = json.loads((ROOT / "summary.json").read_text())
    require(
        all(sha256_file(p) == h for p, h in (summary["sources"] | summary["artifacts"]).items()),
        "REGRESSION_SOURCE_CHANGED",
        "original audit and outputs must remain frozen",
    )
    probe = json.loads((ROOT / "probe/manifest.json").read_text())
    require(
        all(sha256_file(p) == h for p, h in probe["frozen_files"].items()),
        "REGRESSION_SOURCE_CHANGED",
        "original probe resources must remain frozen",
    )
    selected = [
        r for r in read_ledger(ROOT) if r["transition"] == "REGRESSED" and r["errors"]["new_final"] == "OVER_GROUP"
    ]
    require(len(selected) == 121, "REGRESSION_MEMBERSHIP", "all 121 incremental over-groups required")
    require(
        sum(r["observed_stage_pattern"] == REPAIR for r in selected) == 20,
        "REGRESSION_REPAIR_MEMBERSHIP",
        "all 20 prior repairs required",
    )
    payloads = {
        r["canonical_pair_id"]: r["payload"]
        for r in map(json.loads, (NEW / "input_full.jsonl").read_text().splitlines())
    }
    outputs = {
        r["canonical_pair_id"]: r
        for r in map(json.loads, (NEW / "full/final_scoring_only.jsonl").read_text().splitlines())
    }
    _, _, finals, _ = load_run(old.FULL_ROOT)
    raw, _ = matched_raw_outputs(old.FULL_ROOT, finals)
    return selected, payloads, outputs, raw


def exact_evidence(text: str, quote: str, side: str) -> dict:
    require(
        isinstance(quote, str) and bool(quote) and quote in text,
        "REGRESSION_QUOTE",
        "review evidence must be a nonempty exact source substring",
        side=side,
    )
    start = text.index(quote)
    end = start + len(quote)
    start_byte, end_byte = len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))
    require(
        text.encode("utf-8")[start_byte:end_byte].decode("utf-8") == quote,
        "REGRESSION_BYTE_ALIGNMENT",
        "UTF-8 and character evidence must agree",
    )
    return {
        "side": side,
        "quote": quote,
        "start_char": start,
        "end_char": end,
        "start_byte_utf8": start_byte,
        "end_byte_utf8": end_byte,
    }


def validate_reviews(spec: dict, rows: list[dict], payloads: dict) -> list[dict]:
    require(
        spec.get("review_method") == METHOD
        and spec.get("reference_changed") is False
        and spec.get("policy") == "dedup-composite-containment-policy-v1",
        "REGRESSION_PROVENANCE",
        "AI review is not a replacement reference",
    )
    require(
        spec.get("columns") == COLUMNS and isinstance(spec.get("cases"), list),
        "REGRESSION_COLUMNS",
        "explicit review fields required",
    )
    require(
        all(isinstance(r, list) and len(r) == len(COLUMNS) for r in spec["cases"]),
        "REGRESSION_COLUMNS",
        "every case must contain every review field",
    )
    index = {r["review_id"]: r for r in rows}
    ids = [r[0] for r in spec["cases"]]
    require(
        len(index) == len(rows) and len(ids) == len(set(ids)) and set(ids) == index.keys(),
        "REGRESSION_MEMBERSHIP",
        "no omitted, duplicate or out-of-cohort reviews",
    )
    reviewed = []
    for values in spec["cases"]:
        note = dict(zip(COLUMNS, values, strict=True))
        require(all(isinstance(v, str) and v.strip() for v in values), "REGRESSION_FIELDS", "no empty review fields")
        require(
            note["assessment"] in ASSESSMENTS
            and note["shared_x"] in {"PRESENT", "ABSENT", "UNRESOLVED"}
            and note["proposed_primary"] in PRIMARY,
            "REGRESSION_ENUM",
            "unknown review classification",
        )
        require(
            note["assessment"] != "CLEAR_MODEL_ERROR" or note["proposed_primary"] == "NO_NO",
            "REGRESSION_CLEAR_GROUP",
            "this cohort concerns false grouping, not relation-only errors",
        )
        require(
            note["assessment"] != "PENDING_JUDGMENT" or note["proposed_primary"] == "UNDECIDED",
            "REGRESSION_PENDING",
            "pending cases must not acquire a decided proposal",
        )
        row = index[note["review_id"]]
        require(
            row["transition"] == "REGRESSED"
            and row["errors"]["new_final"] == "OVER_GROUP"
            and row["reference"]["same_duplicate_group"] == "NO",
            "REGRESSION_COHORT",
            "review only the incremental over-group cohort",
        )
        payload = payloads[row["canonical_pair_id"]]
        require(
            sha256_json(payload) == row["payload_sha256"],
            "REGRESSION_PAYLOAD_CHANGED",
            "exact original input required",
        )
        evidence = [
            exact_evidence(payload[f"document_{s}"]["text"], note[f"quote_{s}"], s.upper()) for s in ("a", "b")
        ]
        validate_evidence_offsets({"evidence": evidence}, payload)
        a, b = (payload[f"document_{s}"]["text"] for s in ("a", "b"))
        reviewed.append(
            {
                **note,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": row["weight"],
                "payload_sha256": row["payload_sha256"],
                "unordered_text_digest": sha256_json(sorted([a, b])),
                "evidence": evidence,
                "observed_stage_pattern": row["observed_stage_pattern"],
                "reference": row["reference"],
                "reference_review_flags": row["reference_review_flags"],
                "saved_primary": row["primary"],
                "saved_errors": row["errors"],
                "proposal_in_sorted_text_order": list(PRIMARY[note["proposed_primary"]][:: 1 if a <= b else -1]),
                "provenance": METHOD,
                "independently_adjudicated": False,
            }
        )
    groups = defaultdict(list)
    for row in reviewed:
        groups[row["unordered_text_digest"]].append(row)
    for group in groups.values():
        require(
            len({(r["assessment"], r["shared_x"], tuple(r["proposal_in_sorted_text_order"])) for r in group}) == 1,
            "REGRESSION_DUPLICATE_DISAGREEMENT",
            "same unordered inputs need consistent assessments and swapped directions",
            review_ids=[r["review_id"] for r in group],
        )
    return sorted(reviewed, key=lambda r: r["review_id"])


def aggregate(rows: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {
        key: {
            "pairs": len(group),
            "weight": sum(r["weight"] for r in group),
            "unique_unordered_text_pairs": len({r["unordered_text_digest"] for r in group}),
            "review_ids": [r["review_id"] for r in group],
        }
        for key, group in sorted(groups.items())
    }


def repair_inventory(reviewed: list[dict], raw: dict) -> list[dict]:
    result = []
    for row in reviewed:
        if row["observed_stage_pattern"] != REPAIR:
            continue
        main, critic = raw[row["canonical_pair_id"]]
        require(bool(main) and bool(critic), "REGRESSION_REPAIR_RAW", "saved main and repair evidence required")
        valid = row["assessment"] == "CLEAR_MODEL_ERROR"
        result.append(
            {
                **row,
                "old_main_raw": main,
                "old_critic_raw": critic,
                "include_in_development_regression": valid,
                "expected_primary_is_human_gold": False,
                "old_reason_status": (
                    "BOUND_IDENTITY_REASON_STILL_SUPPORTED"
                    if row["mechanism"] == "BOUND_IDENTITY"
                    else "CONCLUSION_SUPPORTED_USE_NONEMPTY_X_NOT_SINGLE_RECORD_VETO"
                )
                if valid
                else "DISPUTED_OR_PENDING_DO_NOT_RESTORE_OLD_VETO",
            }
        )
    return result


def panel_revision(
    old_panel: list[dict], reviewed: list[dict], ledger: list[dict], payloads: dict, spec: dict
) -> list[dict]:
    index, reviews = {r["review_id"]: r for r in ledger}, {r["review_id"]: r for r in reviewed}
    additions = spec["additions"]
    old_ids = {r["review_id"] for r in old_panel}
    require(
        len(old_ids) == len(old_panel)
        and len({r["review_id"] for r in additions}) == len(additions)
        and not old_ids & {r["review_id"] for r in additions},
        "REGRESSION_PANEL_MEMBERSHIP",
        "panel delta must be explicit",
    )
    selected = [{"review_id": r["review_id"], "role": r["role"], "origin": "ORIGINAL_V1"} for r in old_panel]
    selected += [{**r, "origin": "REVIEW121_ADDITION"} for r in additions]
    require(
        all(r["review_id"] in index for r in selected), "REGRESSION_PANEL_MEMBERSHIP", "every selected pair must exist"
    )
    guards = {
        r["review_id"]
        for r in reviewed
        if r["observed_stage_pattern"] == REPAIR and r["assessment"] == "CLEAR_MODEL_ERROR"
    }
    require(
        guards <= {r["review_id"] for r in selected},
        "REGRESSION_PANEL_GUARDS",
        "all supported prior repairs are mandatory",
    )
    require(
        {r["mechanism"] for r in reviewed if r["assessment"] == "CLEAR_MODEL_ERROR"}
        <= {
            reviews[r["review_id"]]["mechanism"]
            for r in selected
            if r["review_id"] in reviews and reviews[r["review_id"]]["assessment"] == "CLEAR_MODEL_ERROR"
        },
        "REGRESSION_PANEL_MECHANISMS",
        "all observed clear-error mechanisms need an exemplar",
    )
    result = []
    for item in selected:
        row, review = index[item["review_id"]], reviews.get(item["review_id"])
        payload = payloads[row["canonical_pair_id"]]
        require(
            sha256_json(payload) == row["payload_sha256"], "REGRESSION_PANEL_PAYLOAD", "full original payload required"
        )
        require(
            not payload.get("long_document_evidence", {}).get("truncated", False),
            "REGRESSION_PANEL_TRUNCATED",
            "do not shorten this full-input panel",
        )
        routing = (
            "POLICY_ARBITRATION"
            if (review and review["assessment"] != "CLEAR_MODEL_ERROR")
            else (
                "HISTORICAL_REVIEW_FLAG_RETAINED"
                if row["reference_review_flags"]
                else "INDEPENDENT_ANNOTATION_PENDING"
            )
        )
        result.append(
            {
                **item,
                "canonical_pair_id": row["canonical_pair_id"],
                "weight": row["weight"],
                "payload_sha256": row["payload_sha256"],
                "historical_reference": row["reference"],
                "reference_review_flags": row["reference_review_flags"],
                "review121": review,
                "routing": routing,
                "mandatory_supported_prior_repair": row["review_id"] in guards,
                "independent_shared_x_gold": None,
            }
        )
    texts = [
        sha256_json(sorted(payloads[r["canonical_pair_id"]][f"document_{s}"]["text"] for s in ("a", "b")))
        for r in result
    ]
    require(
        len(texts) == len(set(texts)), "REGRESSION_PANEL_DUPLICATE", "panel uses unique text pairs, not repeated draws"
    )
    return result


def write_packets(output: Path, name: str, rows: list[dict], payloads: dict) -> list[dict]:
    packets = [{"canonical_pair_id": pid, "payload": payload} for pid, payload in payloads.items()]
    blind, private = old.blind_packets(packets, {r["canonical_pair_id"] for r in rows})
    index = {r["canonical_pair_id"]: r for r in rows}
    for item in private:
        item["review_id"] = index[item["canonical_pair_id"]]["review_id"]
    write_text_atomic(
        output / f"{name}/inputs_blind.jsonl",
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in blind),
    )
    write_json_atomic(output / f"{name}/private_key.json", private)
    return blind


def export(output: Path) -> dict:
    output = output.resolve()
    require(
        all(output != root and root not in output.parents for root in (ROOT, NEW, old.FULL_ROOT)),
        "REGRESSION_OUTPUT_SCOPE",
        "use a separate run root, never a frozen parent run",
    )
    rows, payloads, outputs, raw = sources()
    review_path, panel_path = HERE / "regression_reviews_121_v1.json", HERE / "independent_x_probe_v2.json"
    reviewed = validate_reviews(json.loads(review_path.read_text()), rows, payloads)
    inventory = repair_inventory(reviewed, raw)
    panel_spec = json.loads(panel_path.read_text())
    previous = json.loads((ROOT / "summary.json").read_text())
    require(
        panel_spec["parent_contract_digest"] == previous["probe"]["contract_digest"]
        and panel_spec["online_execution_started"] is False
        and panel_spec["automatic_veto_enabled"] is False,
        "REGRESSION_PANEL_CONTRACT",
        "preserve the parent freeze and offline-only scope",
    )
    old_panel = json.loads((ROOT / "probe/private_selection.json").read_text())
    selected = panel_revision(old_panel, reviewed, read_ledger(ROOT), payloads, panel_spec)
    require(len(selected) == panel_spec["pair_count"], "REGRESSION_PANEL_COUNT", "frozen panel size must match")
    for row in reviewed:
        row["new_saved_coverage_response"] = outputs[row["canonical_pair_id"]]["coverage_response"]
    write_json_atomic(output / "review_121_private.json", reviewed)
    write_text_atomic(
        output / "review_121.csv",
        old._csv_text(
            [
                {k: r[k] for k in [*COLUMNS, "weight", "evidence", "observed_stage_pattern", "payload_sha256"]}
                for r in reviewed
            ]
        ),
    )
    write_json_atomic(output / "prior_repairs_20_private.json", inventory)
    write_json_atomic(
        output / "supported_repairs_5_development_guards.json",
        [r for r in inventory if r["include_in_development_regression"]],
    )
    write_packets(output, "full_review", reviewed, payloads)
    arbitration = [r for r in reviewed if r["assessment"] != "CLEAR_MODEL_ERROR"]
    arbitration_blind = write_packets(output, "arbitration", arbitration, payloads)
    write_json_atomic(
        output / "arbitration/annotation_template.json",
        [
            {
                "case_id": r["case_id"],
                "reviewer_type": None,
                "prediction_exposure": None,
                "policy_version": None,
                "shared_substantive_x": None,
                "a_can_replace_b": None,
                "b_can_replace_a": None,
                "evidence": [],
                "rationale": None,
                "status": "PENDING_INDEPENDENT_REVIEW",
            }
            for r in arbitration_blind
        ],
    )
    blind = write_packets(output, "panel_v2", selected, payloads)
    write_json_atomic(output / "panel_v2/private_selection.json", selected)
    write_json_atomic(
        output / "panel_v2/annotation_template.json",
        [
            {
                "case_id": r["case_id"],
                "reviewer_type": None,
                "prediction_exposure": None,
                "shared_substantive_x": None,
                "content_profile_a": None,
                "content_profile_b": None,
                "evidence": [],
                "rationale": None,
                "status": "PENDING_INDEPENDENT_REVIEW",
            }
            for r in blind
        ],
    )
    parent_spec = json.loads((HERE / "independent_x_probe_v1.json").read_text())
    prompt_path, schema_path = HERE / parent_spec["system_prompt_file"], HERE / parent_spec["output_schema_file"]
    write_text_atomic(
        output / "panel_v2/requests_blind.jsonl",
        "".join(
            json.dumps(
                {
                    "case_id": r["case_id"],
                    "messages": [
                        {"role": "system", "content": prompt_path.read_text()},
                        {
                            "role": "user",
                            "content": json.dumps({"payload": r["payload"]}, ensure_ascii=False, sort_keys=True),
                        },
                    ],
                    "output_schema": json.loads(schema_path.read_text()),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for r in blind
        ),
    )
    weighted = previous["historical_final_comparison"]["final_metrics"]["weighted"]
    old_reviewed = [r for r in reviewed if r["review_id"] in {p["review_id"] for p in old_panel}]
    summary = {
        "schema_version": "dedup-incremental-review121-freeze-v1",
        "review_method": METHOD,
        "external_model_calls": 0,
        "reference_changed": False,
        "independently_adjudicated_pairs": 0,
        "eligible_for_release": False,
        "reviewed_pairs": len(reviewed),
        "reviewed_weight": sum(r["weight"] for r in reviewed),
        "unique_unordered_text_pairs": len({r["unordered_text_digest"] for r in reviewed}),
        "by_assessment": aggregate(reviewed, "assessment"),
        "by_mechanism": aggregate(reviewed, "mechanism"),
        "clear_errors_by_mechanism": aggregate(
            [r for r in reviewed if r["assessment"] == "CLEAR_MODEL_ERROR"], "mechanism"
        ),
        "by_stage": {
            s: aggregate([r for r in reviewed if r["observed_stage_pattern"] == s], "assessment")
            for s in sorted({r["observed_stage_pattern"] for r in reviewed})
        },
        "all121_share_of_full_new_fp_weight": sum(r["weight"] for r in reviewed) / weighted["false_positive"],
        "fp_weight_outside_this_review": weighted["false_positive"] - sum(r["weight"] for r in reviewed),
        "recall_attribution": "This cohort contains only reference negatives; no FN/recall attribution is established here.",
        "precision_warning": "Weights partition observed incremental FP burden under the unchanged old reference. Not a causal precision delta, relabelled metric, guaranteed recoverable gain or all-error audit.",
        "supported_prior_repair_ids": [r["review_id"] for r in inventory if r["include_in_development_regression"]],
        "old_panel_coverage": {
            "scope": "Only newly reviewed members of the 121-case regression cohort; other parent controls were not re-adjudicated in this review.",
            "reviewed_members": aggregate(old_reviewed, "assessment"),
            "clear_mechanisms_present": sorted(
                {r["mechanism"] for r in old_reviewed if r["assessment"] == "CLEAR_MODEL_ERROR"}
            ),
            "clear_mechanisms_missing": sorted(
                {r["mechanism"] for r in reviewed if r["assessment"] == "CLEAR_MODEL_ERROR"}
                - {r["mechanism"] for r in old_reviewed if r["assessment"] == "CLEAR_MODEL_ERROR"}
            ),
        },
        "panel_v2": {
            "protocol": panel_spec,
            "pair_count": len(selected),
            "routing_counts": dict(Counter(r["routing"] for r in selected)),
            "parent_contract_digest": previous["probe"]["contract_digest"],
            "independent_gold_available": False,
            "execution_status": "BLOCKED_PENDING_INDEPENDENT_ANNOTATIONS_AND_EXECUTABLE_PREFLIGHT",
        },
    }
    source_files = [
        Path(__file__),
        review_path,
        panel_path,
        prompt_path,
        schema_path,
        HERE / "independent_x_probe_v1.json",
        ROOT / "summary.json",
        ROOT / "probe/manifest.json",
    ]
    summary["sources"] = {str(p.resolve()): sha256_file(p) for p in source_files}
    summary["sources"].update(previous["sources"])
    summary["artifacts"] = {
        str(p): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "summary.json"
    }
    summary["contract_digest"] = sha256_json(summary)
    write_json_atomic(output / "summary.json", summary)
    return summary


def full_text_recipe(payload: dict) -> str:
    """Lossless shorter-side text plus all edits; never shorten either original."""
    a, b = (payload[f"document_{s}"]["text"] for s in ("a", "b"))
    require(isinstance(a, str) and isinstance(b, str), "REGRESSION_FULL_TEXT", "original full strings required")
    base_side, target_side, base, target = ("A", "B", a, b) if len(a) <= len(b) else ("B", "A", b, a)
    base_lines, target_lines = base.splitlines(keepends=True), target.splitlines(keepends=True)
    opcodes = difflib.SequenceMatcher(None, base_lines, target_lines, autojunk=False).get_opcodes()
    rebuilt = "".join(
        "".join(base_lines[i:j] if tag == "equal" else target_lines[k:stop]) for tag, i, j, k, stop in opcodes
    )
    require(rebuilt == target, "REGRESSION_RECONSTRUCTION", "recipe must retain every original character")
    pieces = [
        f"{base_side} FULL ({len(base)} chars):\n{base}",
        f"{target_side} ({len(target)} chars) = same base with these ordered edits; all unlisted characters copied:",
    ]
    for tag, i, j, k, stop in opcodes:
        if tag != "equal":
            pieces.append(
                f"{tag} {base_side} lines[{i}:{j}] -> {target_side} lines[{k}:{stop}]: {''.join(target_lines[k:stop])}"
            )
    return "\n".join(pieces)


def display(start: int, count: int) -> None:
    rows, packets, outputs, raw = sources()
    for row in rows[start : start + count]:
        pid = row["canonical_pair_id"]
        result = outputs[pid]["coverage_response"]
        print(
            f"\n===== {row['review_id']} {row['reference_reason']} {row['observed_stage_pattern']} new={row['primary']['new_final']} ====="
        )
        print(full_text_recipe(packets[pid]))
        print("NEW basis:", result["basis_explanation"])
        for side in ("a", "b"):
            direction = result[f"{side}_meaning_in_{'b' if side == 'a' else 'a'}"]
            print(side, direction["status"], direction["coverage_explanation"])
        if row["observed_stage_pattern"] == "PRIOR_POSTPROCESSING_REPAIR_NOT_RETAINED":
            print("OLD CRITIC:", json.dumps(raw[pid][1], ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, help="Export complete immutable review and revised panel; no online calls"
    )
    args = parser.parse_args()
    if args.output:
        report = export(args.output)
        print(
            json.dumps(
                {
                    "reviewed_pairs": report["reviewed_pairs"],
                    "by_assessment": report["by_assessment"],
                    "supported_prior_repair_ids": report["supported_prior_repair_ids"],
                    "panel_count": report["panel_v2"]["pair_count"],
                    "external_model_calls": 0,
                },
                ensure_ascii=False,
            )
        )
    else:
        display(args.start, args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
