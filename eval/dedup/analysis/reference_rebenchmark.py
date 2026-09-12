# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Same-output, same-population historical comparison under an explicit reference draft."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import bottleneck_audit as historical
from eval.dedup.analysis import paced_result_audit as paced
from eval.dedup.analysis import policy_only_review as policy
from eval.dedup.analysis import regression_review as audit
from eval.dedup.analysis.development_diagnostic import _index, classify_primary_error
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight, evaluate_predictions
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

HERE = Path(__file__).resolve().parent
POLICY_ROOT = Path("/raid/hfang/ihb/runs/policy-only-application-v1")
DRAFT_STATUS = "PARTIAL_POLICY_V2_AI_DEVELOPMENT_SENSITIVITY_NOT_INDEPENDENT_GOLD"
RATES = ("duplicate_precision", "duplicate_recall", "primary_decision_exact")


def scan_payloads(payloads: dict) -> list[dict]:
    """Broad text-only candidate screen, not a semantic classifier or gold filter."""
    rows = []
    for pid, payload in sorted(payloads.items()):
        a, b = (payload[f"document_{s}"]["text"] for s in ("a", "b"))
        row = {
            "canonical_pair_id": pid,
            "payload_sha256": audit.sha256_json(payload),
            "candidate": False,
            "policy_side_candidate": None,
            "smaller_chars": None,
            "larger_chars": None,
            "shared_char_ratio": None,
        }
        if not isinstance(a, str) or not isinstance(b, str):
            row["screen_status"] = "INCOMPLETE_TEXT_REQUIRES_SEPARATE_REVIEW"
        elif payload.get("long_document_evidence", {}).get("truncated"):
            row["screen_status"] = "TRUNCATED_REQUIRES_SEPARATE_REVIEW"
        else:
            side, small, large = ("A", a, b) if len(a) <= len(b) else ("B", b, a)
            shared = sum(
                len(s.get(f"{side.lower()}_text") or "")
                for s in payload.get("semantic_diff_evidence", {}).get("spans", [])
                if s["kind"] == "SHARED"
            )
            ratio = shared / len(small) if small else 0
            candidate = bool(small and len(large) - len(small) >= 120 and (small in large or ratio >= 0.8))
            row.update(
                policy_side_candidate=side,
                smaller_chars=len(small),
                larger_chars=len(large),
                shared_char_ratio=ratio,
                candidate=candidate,
                screen_status="ADDITIVE_TEXT_CANDIDATE" if candidate else "NOT_SELECTED_BY_ADDITIVE_HEURISTIC",
            )
        rows.append(row)
    return rows


def primary(row: dict, prefix: str = "") -> dict:
    return {k: row[f"{prefix}{k}"] for k in PRIMARY_FIELDS}


def exact_policy_families(supported: list[dict], payloads: dict) -> list[dict]:
    """Screen all inputs for reviewed policy texts without consulting labels or predictions."""
    result = []
    for case in supported:
        small_side = "b" if case["product_side"] == "A" else "a"
        small = payloads[case["canonical_pair_id"]][f"document_{small_side}"]["text"]
        body = small[small.index(case["policy_start"]) :]
        for pid, payload in sorted(payloads.items()):
            texts = [payload[f"document_{s}"]["text"] for s in ("a", "b")]
            if not all(isinstance(t, str) for t in texts):
                continue
            same_small = small in texts
            shared_body = all(body in t for t in texts)
            if same_small or shared_body:
                result.append(
                    {
                        "seed_review_id": case["review_id"],
                        "canonical_pair_id": pid,
                        "same_policy_only_text": same_small,
                        "shared_complete_policy_text": shared_body,
                        "automatic_label_change": False,
                    }
                )
    return result


def revised_reference(
    labels: list[dict], supported: list[dict], deferred: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Apply only declared primary decisions to a separate, explicitly provisional reference."""
    refs = _index(labels, "reference")
    updates = _index(supported, "supported")
    pending = _index(deferred, "deferred")
    require(updates.keys().isdisjoint(pending), "REBENCHMARK_OVERLAP", "supported and deferred must be disjoint")
    require(
        (updates.keys() | pending.keys()) <= refs.keys(), "REBENCHMARK_REFERENCE_MEMBERSHIP", "unknown application"
    )
    result, changes = [], []
    for row in labels:
        pid = row["canonical_pair_id"]
        updated = deepcopy(row)
        case = updates.get(pid)
        if case:
            require(
                case["provenance"] == policy.PROVENANCE and case["independently_adjudicated"] is False,
                "REBENCHMARK_PROVENANCE",
                "an AI application is not independent gold",
            )
            require(
                case["review_id"] == row["review_id"] and case["old_reference"] == primary(row, "human_"),
                "REBENCHMARK_OLD_REFERENCE",
                "application must target the exact historical reference",
            )
            proposal = case["proposed_primary_not_gold"]
            require(
                set(proposal) == set(PRIMARY_FIELDS)
                and proposal["same_duplicate_group"] == "YES"
                and sorted([proposal["a_can_replace_b"], proposal["b_can_replace_a"]]) == ["NO", "YES"],
                "REBENCHMARK_DIRECTION",
                "policy plus product must have exactly one replacement direction",
            )
            updated.update({f"human_{k}": v for k, v in proposal.items()})
            if primary(row, "human_") != proposal:
                changes.append(
                    {
                        "review_id": row["review_id"],
                        "canonical_pair_id": pid,
                        "weight": _weight(row),
                        "before": primary(row, "human_"),
                        "after": proposal,
                        "provenance": policy.PROVENANCE,
                    }
                )
        updated.update(
            reference_status=DRAFT_STATUS,
            reference_application_status="REVIEWED_AI_POLICY_APPLICATION"
            if case
            else "DEFERRED_OLD_PRIMARY_RETAINED"
            if pid in pending
            else "UNCHANGED_NOT_REVALIDATED",
            independent_adjudication=False,
            taxonomy_status="HISTORICAL_TAXONOMY_NOT_REVISED_OR_SCORED",
        )
        require(_weight(updated) == _weight(row), "REBENCHMARK_WEIGHT", "original weights must not change")
        result.append(updated)
    return result, changes


def score(labels: list[dict], predictions: list[dict]) -> dict:
    refs, preds = _index(labels, "reference"), _index(predictions, "predictions")
    require(
        refs.keys() == preds.keys(), "REBENCHMARK_MEMBERSHIP", "score the same full population, never an intersection"
    )
    missing = [pid for pid, r in preds.items() if r.get("metric_only_missing_output")]
    require(
        all(set(primary(preds[pid]).values()) == {"UNRESOLVED"} for pid in missing),
        "REBENCHMARK_SENTINEL",
        "a missing-output placeholder cannot carry a semantic decision",
    )
    metrics = evaluate_predictions(labels, predictions)
    # A scoring placeholder is an engineering failure, even when the reference is unresolved.
    for weighted, key in ((False, "unweighted"), (True, "weighted")):
        block = metrics[key]
        credit = sum(
            (_weight(refs[pid]) if weighted else 1.0)
            for pid in missing
            if primary(refs[pid], "human_") == primary(preds[pid])
        )
        if credit:
            block["primary_decision_exact"] -= credit / block["weight_total"]
        block.pop("taxonomy_exact")
    counts, weights = Counter(), Counter()
    for pid, ref in refs.items():
        error = "ENGINEERING_MISSING_OUTPUT" if pid in missing else classify_primary_error(ref, preds[pid])
        counts[error] += 1
        weights[error] += _weight(ref)
    return {
        "weighted": metrics["weighted"],
        "unweighted": metrics["unweighted"],
        "error_counts": dict(counts),
        "error_weights": dict(weights),
        "missing_output_pairs": len(missing),
        "taxonomy_scored": False,
        "interpretation": "Full original denominator; reference-sensitive agreement, not new model performance.",
    }


def comparison(labels: list[dict], views: dict[str, list[dict]]) -> dict:
    scores = {name: score(labels, rows) for name, rows in views.items()}
    components = {}
    for version in ("v0.6.2.9", "v0.6.2.12", "v0.6.2.32"):
        components[version] = historical.transition_matrix(labels, views[f"{version}/main"], views[f"{version}/final"])
        for k in ("main_metrics", "final_metrics"):
            components[version].pop(k)
    paired = {}
    for a, b in (("v0.6.2.9", "v0.6.2.12"), ("v0.6.2.12", "v0.6.2.32"), ("v0.6.2.9", "v0.6.2.32")):
        data = historical.transition_matrix(labels, views[f"{a}/final"], views[f"{b}/final"])
        data.pop("main_metrics")
        data.pop("final_metrics")
        data["interpretation"] = (
            "Paired historical versions on identical inputs and reference; not an isolated prompt experiment."
        )
        paired[f"{a} -> {b}"] = data
    return {"scores": scores, "main_to_final": components, "paired_versions": paired}


def metric_deltas(before: dict, after: dict) -> dict:
    return {
        name: {
            mode: {
                key: None
                if before[name][mode][key] is None or after[name][mode][key] is None
                else 100 * (after[name][mode][key] - before[name][mode][key])
                for key in RATES
            }
            for mode in ("weighted", "unweighted")
        }
        for name in before
    }


def verify_freeze(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if "contract_digest" in manifest:
        require(
            manifest["contract_digest"] == sha256_json({k: v for k, v in manifest.items() if k != "contract_digest"}),
            "REBENCHMARK_MANIFEST_DIGEST",
            "frozen manifest metadata changed",
        )
    files = manifest["sources"] | manifest["artifacts"]
    require(all(sha256_file(p) == h for p, h in files.items()), "REBENCHMARK_FROZEN_SOURCE", "frozen inputs changed")
    return files | {str(path.resolve()): sha256_file(path)}


def load_views() -> tuple[dict, dict, dict, dict]:
    views, packets, provenance, source_files = {}, {}, {}, {}
    for version, final_policy, main_policy in (("v0.6.2.9", "v3", "v1"), ("v0.6.2.12", "v6", "v6-route")):
        root = paced.HISTORICAL[version]
        main, final, inputs, proof = historical.component_replay(root, final_policy, main_policy=main_policy)
        views[f"{version}/main"], views[f"{version}/final"] = main, final
        packets[version] = {pid: row["payload"] for pid, row in _index(inputs, version).items()}
        provenance[version] = proof
        source_files.update(proof["raw_digests"])
        source_files.update(
            {
                str(p): sha256_file(p)
                for p in (
                    root / "run_manifest.json",
                    root / "run_complete.json",
                    root / "data/judge_results.jsonl",
                    root / "data/judge_payloads.jsonl",
                )
            }
        )
    inputs = [json.loads(line) for line in (audit.NEW / "input_full.jsonl").read_text().splitlines()]
    result = paced.replay(audit.NEW, "full", inputs)
    packets["v0.6.2.32"] = {pid: r["payload"] for pid, r in _index(inputs, "v0.6.2.32").items()}
    for stage, rows in result.pop("scoring").items():
        views[f"v0.6.2.32/{stage}"] = rows
    provenance["v0.6.2.32"] = result
    for p in sorted((audit.NEW / "full").rglob("*")):
        if p.is_file() and (p.suffix in {".json", ".jsonl", ".yaml", ".yml", ".jinja"}):
            source_files[str(p)] = sha256_file(p)
    for p in (audit.NEW / "input_full.jsonl", audit.NEW / "manifest.json"):
        source_files[str(p)] = sha256_file(p)
    reference = packets["v0.6.2.9"]
    require(
        all(p == reference for p in packets.values()),
        "REBENCHMARK_PAYLOAD",
        "all three exact input payloads must agree",
    )
    require(len(reference) == 1000, "REBENCHMARK_POPULATION", "original 1,000 pairs required")
    return views, reference, provenance, source_files


def export(output: Path) -> dict:
    output = output.resolve()
    protected = (HERE, audit.ROOT, audit.NEW, policy.REVIEW_ROOT, POLICY_ROOT, *paced.HISTORICAL.values())
    require(
        all(output != p and p not in output.parents and output not in p.parents for p in protected),
        "REBENCHMARK_OUTPUT",
        "use a separate output root, never a source or parent",
    )
    if (output / "summary.json").exists():
        verify_freeze(output / "summary.json")
        return json.loads((output / "summary.json").read_text())
    require(
        not output.exists() or not any(output.iterdir()),
        "REBENCHMARK_OUTPUT_EXISTS",
        "refuse to overwrite nonempty output",
    )
    source_files = verify_freeze(POLICY_ROOT / "summary.json")
    source_files.update(verify_freeze(audit.ROOT / "summary.json"))
    require(
        sha256_file(historical.REFERENCE) == historical.REFERENCE_SHA256,
        "REBENCHMARK_REFERENCE",
        "historical reference must remain immutable",
    )
    labels = historical._read_csv(historical.REFERENCE)
    views, payloads, provenance, replay_files = load_views()
    source_files.update(replay_files)
    original_predictions_digest = sha256_json(views)
    ledger = audit.read_ledger(audit.ROOT)
    index = {r["canonical_pair_id"]: r for r in ledger}
    require(
        set(index) == set(_index(labels, "reference")) == set(payloads),
        "REBENCHMARK_MEMBERSHIP",
        "exact joins required",
    )
    spec_path = HERE / "policy_only_review_v1.json"
    supported, deferred = policy.apply_cases(json.loads(spec_path.read_text()), ledger, payloads)
    draft, changes = revised_reference(labels, supported, deferred)
    families = exact_policy_families(supported, payloads)
    covered = {r["canonical_pair_id"] for r in supported}
    unmatched_family = sorted({r["canonical_pair_id"] for r in families} - covered)
    old_conflicts = historical.exact_conflicts(
        labels, [{"canonical_pair_id": p, "payload": v} for p, v in payloads.items()]
    )
    draft_conflicts = historical.exact_conflicts(
        draft, [{"canonical_pair_id": p, "payload": v} for p, v in payloads.items()]
    )
    require(
        not any(m["canonical_pair_id"] in covered for c in draft_conflicts for m in c["members"]),
        "REBENCHMARK_NEW_CONFLICT",
        "a revised pair cannot conflict with an exact document-pair mirror",
    )
    reviews = json.loads((policy.REVIEW_ROOT / "review_121_private.json").read_text())
    reviewed = {r["canonical_pair_id"]: r for r in reviews}
    applications = {r["canonical_pair_id"]: r for r in supported + deferred}
    guard_pid = next(r["canonical_pair_id"] for r in ledger if r["review_id"] == "H0316")
    guard = {
        "review_id": "H0316",
        "canonical_pair_id": guard_pid,
        "status": "REVIEWED_ACTUAL_POLICY_OBLIGOR_CONFLICT_OLD_PRIMARY_RETAINED_NOT_GOLD",
        "note": "完整复核较短政策及全部长侧差异: 退款义务主体及联系对象分别为 Brick Takeover GmbH 和 hapotec vertriebs limited; 不是完整相同政策。不能仅凭新增商品或高共享比例推定包含。",
        "provenance": policy.PROVENANCE,
        "independently_adjudicated": False,
        "evidence": [
            audit.exact_evidence(payloads[guard_pid]["document_a"]["text"], "Brick Takeover GmbH", "A"),
            audit.exact_evidence(payloads[guard_pid]["document_b"]["text"], "hapotec vertriebs limited", "B"),
        ],
    }
    screens = scan_payloads(payloads)
    for r in screens:
        pid = r["canonical_pair_id"]
        r["review_id"] = index[pid]["review_id"]
        r["historical_reference_group"] = index[pid]["reference"]["same_duplicate_group"]
        r["individual_review_status"] = (
            applications[pid]["status"]
            if pid in applications
            else "PRIOR_AI_REVIEW121_NOT_REVALIDATED"
            if pid in reviewed
            else "UNREVIEWED_CANDIDATE"
            if r["candidate"]
            else "NOT_SELECTED_NOT_ADJUDICATED"
        )
        if pid == guard_pid:
            r["individual_review_status"] = guard["status"]
    pending_screen = [r for r in screens if r["individual_review_status"] == "UNREVIEWED_CANDIDATE"]
    old_result, new_result = comparison(labels, views), comparison(draft, views)
    errors = []
    view_indices = {k: _index(v, k) for k, v in views.items()}
    refs, revised = _index(labels, "reference"), _index(draft, "draft")
    for pid in sorted(refs, key=lambda p: refs[p]["review_id"]):
        row = index[pid]
        errors.append(
            {
                "review_id": row["review_id"],
                "canonical_pair_id": pid,
                "weight": _weight(refs[pid]),
                "old_reference": primary(refs[pid], "human_"),
                "draft_reference": primary(revised[pid], "human_"),
                "reference_status": revised[pid]["reference_application_status"],
                "historical_review_flags": row["reference_review_flags"],
                "review121_assessment_about_v32": reviewed.get(pid, {}).get("assessment"),
                "policy_application_status": applications.get(pid, {}).get("status"),
                "primary": {k: primary(v[pid]) for k, v in view_indices.items()},
                "old_errors": {k: classify_primary_error(refs[pid], v[pid]) for k, v in view_indices.items()},
                "draft_errors": {k: classify_primary_error(revised[pid], v[pid]) for k, v in view_indices.items()},
                "engineering_missing_views": [
                    k for k, v in view_indices.items() if v[pid].get("metric_only_missing_output")
                ],
            }
        )
    require(
        sha256_json(views) == original_predictions_digest,
        "REBENCHMARK_PREDICTION_MUTATION",
        "scoring must not mutate outputs",
    )
    artifacts = {
        "reference_historical.csv": historical.REFERENCE.read_bytes().decode("utf-8"),
        "reference_policy_v2_partial_draft.csv": historical._csv_text(draft),
        "ledger_1000.csv": historical._csv_text(errors),
        "candidate_screen_1000.csv": historical._csv_text(screens),
    }
    for name, value in artifacts.items():
        write_text_atomic(output / name, value)
    for name, value in {
        "historical_scores.json": old_result,
        "partial_draft_scores.json": new_result,
        "relabel_only_delta_pp.json": metric_deltas(old_result["scores"], new_result["scores"]),
        "reference_changes_private.json": changes,
        "policy_family_matches_private.json": families,
        "actual_obligor_conflict_guard_private.json": guard,
        "exact_reference_conflicts_private.json": {"historical": old_conflicts, "partial_draft": draft_conflicts},
        "applications_private.json": {"supported": supported, "deferred": deferred},
        "replay_provenance.json": provenance,
    }.items():
        write_json_atomic(output / name, value)
    pending_ids = (
        {r["canonical_pair_id"] for r in deferred}
        | {r["canonical_pair_id"] for r in pending_screen}
        | set(unmatched_family)
    )
    pending_ids.update(
        r["canonical_pair_id"]
        for r in reviews
        if r["assessment"] != "CLEAR_MODEL_ERROR" and r["canonical_pair_id"] not in covered
    )
    pending_ids.update(m["canonical_pair_id"] for c in draft_conflicts for m in c["members"])
    audit.write_packets(output, "pending_review", [index[p] for p in sorted(pending_ids)], payloads)
    source_paths = (
        Path(__file__),
        historical.REFERENCE,
        spec_path,
        HERE / "composite_containment_policy_v2.md",
        Path(historical.__file__),
        Path(paced.__file__),
        Path(policy.__file__),
    )
    source_files.update({str(p.resolve()): sha256_file(p) for p in source_paths})
    summary = {
        "schema_version": "dedup-reference-policy-v2-rebenchmark-v1",
        "reference_status": DRAFT_STATUS,
        "original_reference_replaced": False,
        "historical_predictions_modified": False,
        "external_model_calls": 0,
        "independently_adjudicated_new_pairs": 0,
        "release_gate_evaluation": "NOT_ELIGIBLE_PARTIAL_REFERENCE_SENSITIVITY",
        "population": len(labels),
        "weight_total": sum(_weight(r) for r in labels),
        "identical_payloads_all_versions": True,
        "prediction_views": sorted(views),
        "predictions_digest": original_predictions_digest,
        "primary_reference_changes": len(changes),
        "changed_review_ids": [r["review_id"] for r in changes],
        "changed_weight": sum(r["weight"] for r in changes),
        "supported_application_pairs": len(supported),
        "known_policy_family_matches": len({r["canonical_pair_id"] for r in families}),
        "unreviewed_known_policy_family_ids": unmatched_family,
        "screen": {
            "method": "Text only, all 1,000. Nonempty shorter text; length increment >=120 chars; exact substring or shared-span chars / shorter chars >=0.8. Separately scan all inputs for six reviewed policy texts.",
            "limitation": "Heuristic recall is unknown. Not selected does not mean semantically unaffected. Candidates are not adjudications; triage snippets never become gold.",
            "counts": dict(Counter(r["screen_status"] for r in screens)),
            "candidate_old_reference_groups": dict(
                Counter(r["historical_reference_group"] for r in screens if r["candidate"])
            ),
            "unreviewed_additive_candidates": len(pending_screen),
        },
        "pending_review_packet_pairs": len(pending_ids),
        "taxonomy_revised_or_scored": False,
        "exact_reference_conflict_groups": {"historical": len(old_conflicts), "partial_draft": len(draft_conflicts)},
        "interpretation": "Only five reviewed primary labels differ; all versions use that same partial draft. Others retain historical labels, including pending cases. This measures label sensitivity, not the fully adjudicated new policy or a model improvement.",
        "sources": source_files,
        "artifacts": {str(p): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()},
    }
    summary["contract_digest"] = sha256_json(summary)
    write_json_atomic(output / "summary.json", summary)
    return summary


def display(start: int, count: int) -> None:
    _, payloads, _, _ = audit.sources()
    ledger = {r["canonical_pair_id"]: r for r in audit.read_ledger(audit.ROOT)}
    previous = json.loads((HERE / "regression_reviews_121_v1.json").read_text())
    reviewed = {r[0] for r in previous["cases"]}
    selected = [
        r
        for r in scan_payloads(payloads)
        if r["candidate"] and ledger[r["canonical_pair_id"]]["review_id"] not in reviewed
    ]
    selected.sort(key=lambda r: ledger[r["canonical_pair_id"]]["review_id"])
    print("Unreviewed additive candidates:", len(selected))
    for row in selected[start : start + count]:
        pid = row["canonical_pair_id"]
        payload = payloads[pid]
        side = row["policy_side_candidate"]
        text = payload[f"document_{side.lower()}"]["text"]
        spans = payload["semantic_diff_evidence"]["spans"]
        unique = " ".join(s["text"] for s in spans if s["kind"] == f"{'B' if side == 'A' else 'A'}_ONLY")
        print(
            ledger[pid]["review_id"],
            side,
            row["smaller_chars"],
            "SMALL TRIAGE",
            re.sub(r"\s+", " ", text[:240]),
            "ADDITION TRIAGE",
            re.sub(r"\s+", " ", unique[:900]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        result = export(args.output)
        print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "artifacts"}}, indent=2))
    else:
        display(args.start, args.count)
