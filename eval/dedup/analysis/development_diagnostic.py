# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reference-frozen development error distributions, without release approval."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight, evaluate_predictions
from eval.dedup.analysis.policy_review import _payload_side_text, _read_csv, _read_jsonl, _snippet, _write_csv
from eval.dedup.contracts import canonical_json_bytes
from eval.dedup.judging.local_ndd import adapt_ndd_judge_output
from eval.dedup.judging.payload import validate_evidence_offsets
from eval.dedup.judging.schema_v3 import JUDGE_FIELDS_V3, validate_judge_output_v3
from eval.dedup.validation import require, sha256_file, write_json_atomic


def classify_primary_error(label: dict[str, Any], prediction: dict[str, Any]) -> str:
    """Make abstention disjoint from resolved grouping and direction errors."""
    if all(label[f"human_{key}"] == prediction.get(key) for key in PRIMARY_FIELDS):
        return "CORRECT"
    if label["human_same_duplicate_group"] == "UNRESOLVED":
        return "REFERENCE_UNRESOLVED"
    if any(prediction.get(key) in {None, "UNRESOLVED"} for key in PRIMARY_FIELDS):
        return "UNRESOLVED"
    if label["human_same_duplicate_group"] == "NO" and prediction["same_duplicate_group"] == "YES":
        return "OVER_GROUP"
    if label["human_same_duplicate_group"] == "YES" and prediction["same_duplicate_group"] == "NO":
        return "UNDER_GROUP_RESOLVED"
    return "DIRECTION_ONLY"


def _index(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("canonical_pair_id", "")): row for row in rows}
    require(
        "" not in indexed and len(indexed) == len(rows),
        "DIAGNOSTIC_DUPLICATE_ID",
        "expected unique nonempty pair IDs",
        source=source,
    )
    return indexed


def _reason(prediction: dict[str, Any], prefix: str) -> str:
    values = [code.split(":", 1)[1] for code in prediction.get("reason_codes", []) if code.startswith(prefix + ":")]
    require(len(values) <= 1, "DIAGNOSTIC_DUPLICATE_REASON", "expected one reason per axis", prefix=prefix)
    return values[0] if values else "MISSING"


def evaluate_primary_guards(guards: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """A missing or unresolved protected pair is a regression, never a smaller denominator."""
    expected, actual = _index(guards, "guards"), _index(predictions, "predictions")
    require(bool(expected), "DIAGNOSTIC_EMPTY_GUARDS", "a protection suite must not be empty")
    require(
        all(isinstance(g.get("final"), dict) and all(k in g["final"] for k in PRIMARY_FIELDS) for g in guards),
        "DIAGNOSTIC_INVALID_GUARDS",
        "each guard requires the complete expected primary tuple",
    )
    violations = [
        g["review_id"]
        for pair_id, g in expected.items()
        if pair_id not in actual or any(actual[pair_id].get(k) != g["final"][k] for k in PRIMARY_FIELDS)
    ]
    return {
        "expected": len(guards),
        "preserved": len(guards) - len(violations),
        "violations": violations,
        "passed": not violations,
    }


def _distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row["error_type"] != "CORRECT"]
    total_weight = sum(row["sample_weight"] for row in rows)
    error_weight = sum(row["sample_weight"] for row in errors)
    return {
        "rows": len(rows),
        "primary_errors": len(errors),
        "weighted_rows": total_weight,
        "weighted_primary_errors": error_weight,
        "primary_error_rate": len(errors) / len(rows) if rows else None,
        "weighted_primary_error_rate": error_weight / total_weight if total_weight else None,
        "error_types": dict(sorted(Counter(row["error_type"] for row in rows).items())),
        "error_ids": [row["review_id"] for row in errors],
    }


def build_diagnostic(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    payload_rows: list[dict[str, Any]],
    residual_ids: set[str],
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join exact memberships and expose both denominators and disjoint error counts."""
    references, predicted, payloads, metadata = (
        _index(rows, source)
        for rows, source in (
            (labels, "labels"),
            (predictions, "predictions"),
            (payload_rows, "payloads"),
            (metadata_rows, "metadata"),
        )
    )
    require(
        bool(references) and set(references) == set(predicted) == set(payloads) == set(metadata),
        "DIAGNOSTIC_MEMBERSHIP",
        "labels, predictions, payloads and metadata must have identical memberships",
    )
    require(residual_ids <= references.keys(), "DIAGNOSTIC_RESIDUAL_MEMBERSHIP", "residual IDs outside reference")
    require(
        len({row["review_id"] for row in labels}) == len(labels),
        "DIAGNOSTIC_DUPLICATE_REVIEW_ID",
        "review IDs must be unique",
    )
    rows = []
    for pair_id, label in references.items():
        prediction, payload, meta = predicted[pair_id], payloads[pair_id]["payload"], metadata[pair_id]
        token_count = max(int(meta.get("token_count_low") or 0), int(meta.get("token_count_high") or 0))
        ratio = meta.get("token_length_ratio")
        rows.append(
            {
                "review_id": label["review_id"],
                "canonical_pair_id": pair_id,
                "sample_weight": _weight(label),
                "partition": "original_residual_127" if pair_id in residual_ids else "development_complement_873",
                "error_type": classify_primary_error(label, prediction),
                "human_reason_code": label.get("human_reason_code") or "UNSPECIFIED",
                **{
                    f"human_{key}": label[f"human_{key}"]
                    for key in (*PRIMARY_FIELDS, "relation_type", "material_difference")
                },
                **{
                    f"judge_{key}": prediction.get(key)
                    for key in (*PRIMARY_FIELDS, "relation_type", "material_difference")
                },
                "confidence_tier": prediction.get("confidence_tier"),
                "span_ledger_rule": _reason(prediction, "SPAN_LEDGER_RULE"),
                "record_binding_critic": _reason(prediction, "SPAN_RECORD_BINDING_CRITIC"),
                "record_scope": _reason(prediction, "RECORD_SCOPE"),
                "retained_conflict": _reason(prediction, "RETAINED_CONFLICT"),
                "content_profile_pair": "/".join(
                    _reason(prediction, f"SPAN_CONTENT_PROFILE_{side}") for side in ("A", "B")
                ),
                "translation_status": _reason(prediction, "SPAN_TRANSLATION_STATUS"),
                "hard_conflict": _reason(prediction, "SPAN_HARD_CONFLICT"),
                "primary_risk_factor": prediction.get("primary_risk_factor"),
                "overlap_source": prediction.get("dominant_overlap_source"),
                "truncated": bool(payload.get("long_document_evidence", {}).get("truncated", False)),
                "diff_packet_status": payload.get("semantic_diff_evidence", {}).get("status", "MISSING"),
                "track": "BOTH"
                if meta.get("has_track_5a") and meta.get("has_track_5b")
                else "5a"
                if meta.get("has_track_5a")
                else "5b"
                if meta.get("has_track_5b")
                else "MISSING",
                "language_pair": "/".join(
                    sorted({str(meta.get("language_low") or "MISSING"), str(meta.get("language_high") or "MISSING")})
                ),
                "same_hostname": meta.get("same_hostname"),
                "length_bucket": "MISSING"
                if not token_count
                else "LE_256"
                if token_count <= 256
                else "257_1024"
                if token_count <= 1024
                else "1025_4096"
                if token_count <= 4096
                else "GT_4096",
                "length_ratio_bucket": "MISSING"
                if ratio is None
                else "LT_0.25"
                if ratio < 0.25
                else "0.25_0.5"
                if ratio < 0.5
                else "0.5_0.8"
                if ratio < 0.8
                else "GE_0.8",
                "taxonomy_exact": all(
                    label[f"human_{key}"] == prediction.get(key) for key in ("relation_type", "material_difference")
                ),
                "reason_codes": json.dumps(prediction.get("reason_codes", []), ensure_ascii=False),
                "evidence": json.dumps(prediction.get("evidence", []), ensure_ascii=False),
                "document_a": _snippet(_payload_side_text(payload, "A"), limit=2400),
                "document_b": _snippet(_payload_side_text(payload, "B"), limit=2400),
            }
        )
    rows.sort(key=lambda row: (-row["sample_weight"], row["review_id"]))
    axes = (
        "partition",
        "human_reason_code",
        "human_relation_type",
        "judge_relation_type",
        "span_ledger_rule",
        "record_binding_critic",
        "record_scope",
        "retained_conflict",
        "content_profile_pair",
        "translation_status",
        "hard_conflict",
        "overlap_source",
        "primary_risk_factor",
        "truncated",
        "diff_packet_status",
        "track",
        "language_pair",
        "same_hostname",
        "length_bucket",
        "length_ratio_bucket",
    )
    tables = {}
    for axis in axes:
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[axis])].append(row)
        tables[axis] = {key: _distribution(group) for key, group in sorted(groups.items())}
    partitions = {}
    for name in ("original_residual_127", "development_complement_873"):
        selected = [
            label
            for label in labels
            if (label["canonical_pair_id"] in residual_ids) == (name == "original_residual_127")
        ]
        if selected:
            partitions[name] = evaluate_predictions(selected, predictions)
    summary = {
        "schema_version": "dedup-full-development-diagnostic-v1",
        "eligible_for_promotion": False,
        "interpretation": "Frozen development reference including earlier prediction-aware AI reviews, not independent human holdout. Group recall counts unresolved reference duplicates as misses; the error partition counts each primary mismatch once.",
        "metrics": evaluate_predictions(labels, predictions),
        "primary_error_distribution": _distribution(rows),
        "taxonomy_only_ids": [
            row["review_id"] for row in rows if row["error_type"] == "CORRECT" and not row["taxonomy_exact"]
        ],
        "partitions": partitions,
        "cross_tabs": tables,
    }
    return summary, rows


def load_run(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject changed artifacts or incomplete accounting before reporting quality."""
    manifest = json.loads((root / "run_manifest.json").read_text())
    complete = json.loads((root / "run_complete.json").read_text())
    results_path, payload_path = root / "data/judge_results.jsonl", root / "data/judge_payloads.jsonl"
    require(
        sha256_file(results_path) == complete["results_sha256"]
        and sha256_file(payload_path) == manifest["judge_payloads_sha256"],
        "DIAGNOSTIC_ARTIFACT_CHANGED",
        "result or payload hash differs from its frozen manifest",
    )
    predictions, payloads = _read_jsonl(results_path), _read_jsonl(payload_path)
    require(
        int(complete["requested"])
        == int(complete["valid"])
        == len(predictions)
        == len(payloads)
        == int(manifest["source_pair_count"])
        and int(complete["errors"]) == 0,
        "DIAGNOSTIC_INCOMPLETE_RUN",
        "this diagnostic requires a complete run with no terminal errors",
    )
    require(
        all(row.get("judge_contract_digest") == manifest["judge_contract_digest"] for row in predictions),
        "DIAGNOSTIC_CONTRACT_MISMATCH",
        "prediction belongs to another execution contract",
    )
    payload_by_pair = _index(payloads, "payloads")
    require(
        set(_index(predictions, "predictions")) == set(payload_by_pair),
        "DIAGNOSTIC_MEMBERSHIP",
        "prediction and payload memberships differ",
    )
    for row in predictions:
        public = validate_judge_output_v3({key: row.get(key) for key in JUDGE_FIELDS_V3})
        payload = payload_by_pair[row["canonical_pair_id"]]["payload"]
        validate_evidence_offsets(public, payload)
        require(
            not payload.get("long_document_evidence", {}).get("truncated") or public["confidence_tier"] != "HIGH",
            "DIAGNOSTIC_TRUNCATION_CONFIDENCE",
            "truncated evidence cannot support HIGH confidence",
        )
    return manifest, complete, predictions, payloads


def matched_raw_outputs(
    root: Path,
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], dict[str, str]]:
    """Select the final provider response by digest, not by attempt/file ordering."""
    raw_paths = sorted(root.glob("runtime/**/attempt_*/output/*.jsonl"))
    raw_by_digest = {}
    for path in raw_paths:
        for row in _read_jsonl(path):
            main, critic = row.get("qwen_dedup_semantic_judge"), row.get("qwen_dedup_record_binding_critic")
            if main is None or critic is None:
                continue
            digest = hashlib.sha256(
                canonical_json_bytes({"semantic_judge": main, "record_binding_critic": critic})
            ).hexdigest()
            raw_by_digest[(row["canonical_pair_id"], digest)] = (main, critic)
    matched = {}
    for prediction in predictions:
        pair_id = prediction["canonical_pair_id"]
        key = pair_id, prediction["provider_response_sha256"]
        require(
            key in raw_by_digest,
            "DIAGNOSTIC_RAW_MISSING",
            "no saved response matches the published digest",
            pair_id=pair_id,
        )
        matched[pair_id] = raw_by_digest[key]
    return matched, {str(path): sha256_file(path) for path in raw_paths}


def replay_v0629_arbitration(
    root: Path,
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attribute changes to the critic using identical, digest-matched main-model outputs."""
    references, payload_by_pair = _index(labels, "labels"), _index(payloads, "payloads")
    raw_by_pair, raw_digests = matched_raw_outputs(root, predictions)
    main_predictions, changes = [], []
    effects = Counter()
    weighted_effects = Counter()
    for prediction in predictions:
        pair_id = prediction["canonical_pair_id"]
        main, critic = raw_by_pair[pair_id]
        payload = payload_by_pair[pair_id]["payload"]
        final_replay = adapt_ndd_judge_output(
            main, "dedup-judge-output-v3", payload=payload, record_binding_critic=critic, record_binding_policy="v3"
        )
        require(
            all(prediction[key] == value for key, value in final_replay.items()),
            "DIAGNOSTIC_REPLAY_CHANGED",
            "current V0.6.2.9 replay differs from the published result",
            pair_id=pair_id,
        )
        main_only = adapt_ndd_judge_output(main, "dedup-judge-output-v3", payload=payload)
        main_predictions.append({"canonical_pair_id": pair_id, **main_only})
        label = references[pair_id]
        main_error, final_error = classify_primary_error(label, main_only), classify_primary_error(label, prediction)
        changed = any(main_only[key] != prediction[key] for key in PRIMARY_FIELDS)
        effect = (
            "MAIN_CORRECT_CRITIC_REGRESSED"
            if main_error == "CORRECT" and final_error != "CORRECT"
            else "MAIN_WRONG_CRITIC_REPAIRED"
            if main_error != "CORRECT" and final_error == "CORRECT"
            else "BOTH_CORRECT"
            if main_error == "CORRECT"
            else "BOTH_WRONG_CHANGED"
            if changed
            else "BOTH_WRONG_UNCHANGED"
        )
        effects[effect] += 1
        weighted_effects[effect] += _weight(label)
        if changed:
            changes.append(
                {
                    "review_id": label["review_id"],
                    "canonical_pair_id": pair_id,
                    "effect": effect,
                    "sample_weight": _weight(label),
                    "human_reason_code": label.get("human_reason_code"),
                    "main_error": main_error,
                    "final_error": final_error,
                    "main": {key: main_only[key] for key in PRIMARY_FIELDS},
                    "final": {key: prediction[key] for key in PRIMARY_FIELDS},
                }
            )
    return {
        "interpretation": "Same saved main outputs with and without V0.6.2.9 critic arbitration; no new model calls. Final replay must match all published fields. This isolates arbitration effects, not the causal benefit of a future prompt.",
        "main_only_metrics": evaluate_predictions(labels, main_predictions),
        "effects": dict(effects),
        "weighted_effects": dict(weighted_effects),
        "changed_primary_rows": changes,
        "raw_source_digests": raw_digests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--residual-subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--historical-run", type=Path, action="append", default=[])
    parser.add_argument("--original-labels", type=Path)
    parser.add_argument("--replay-v0629", action="store_true")
    parser.add_argument(
        "--guard-replay", type=Path, help="Frozen translation replay containing the 38 critic-repaired negative guards"
    )
    args = parser.parse_args()
    require(
        not any(path.exists() for path in (args.output, args.rows, args.errors)),
        "DIAGNOSTIC_OUTPUT_EXISTS",
        "use new diagnostic output paths",
    )
    manifest, complete, predictions, payloads = load_run(args.run_root)
    require(
        sha256_file(args.labels) == manifest["pair_selection"]["sha256"],
        "DIAGNOSTIC_REFERENCE_CHANGED",
        "reference labels must match the frozen pair-selection file",
    )
    import pyarrow.parquet as pq

    metadata_path = Path(manifest["source_artifacts"]["pair_comparisons"])
    require(
        sha256_file(metadata_path) == manifest["source_digests"]["pair_comparisons"],
        "DIAGNOSTIC_METADATA_CHANGED",
        "metadata source changed",
    )
    labels, subset = _read_csv(args.labels), _read_csv(args.residual_subset)
    pair_ids = {row["canonical_pair_id"] for row in labels}
    residual_ids = set(_index(subset, "residual_subset"))
    require(
        len(labels) == 1000 and len(residual_ids) == 127,
        "DIAGNOSTIC_EXPECTED_MEMBERSHIP",
        "expected 1,000 development and 127 residual pairs",
    )
    metadata = [row for row in pq.read_table(metadata_path).to_pylist() if row["canonical_pair_id"] in pair_ids]
    summary, rows = build_diagnostic(labels, predictions, payloads, residual_ids, metadata)
    if args.guard_replay is not None:
        guarded = json.loads(args.guard_replay.read_text())
        require(
            guarded.get("schema_version") == "dedup-v06212-translation-route-replay-v1"
            and len(guarded.get("critic_repaired_negative_guards", [])) == 38,
            "DIAGNOSTIC_GUARD_SOURCE",
            "expected the frozen full-development 38-case protection suite",
        )
        summary["critic_repaired_negative_guards"] = {
            **evaluate_primary_guards(guarded["critic_repaired_negative_guards"], predictions),
            "source": str(args.guard_replay),
            "sha256": sha256_file(args.guard_replay),
        }
    if args.original_labels is not None:
        original = _index(_read_csv(args.original_labels), "original_labels")
        require(
            set(original) == pair_ids, "DIAGNOSTIC_MEMBERSHIP", "original and current labels must cover the same pairs"
        )
        revised_ids = {
            label["canonical_pair_id"]
            for label in labels
            if any(
                label[f"human_{key}"] != original[label["canonical_pair_id"]][f"human_{key}"] for key in PRIMARY_FIELDS
            )
        }
        for row in rows:
            row["reference_primary_revised"] = row["canonical_pair_id"] in revised_ids
        summary["reference_revision_slices"] = {
            name: evaluate_predictions(
                [label for label in labels if (label["canonical_pair_id"] in revised_ids) == revised], predictions
            )
            for name, revised in (("primary_revised", True), ("primary_unchanged", False))
        }
        summary["original_reference_sha256"] = sha256_file(args.original_labels)
    summary["operations"] = {**complete, "pair_retry_rate": complete["retried"] / complete["requested"]}
    provider_groups = defaultdict(list)
    for path in sorted(args.run_root.glob("**/events/*.jsonl")):
        name = "preflight" if "preflight" in path.relative_to(args.run_root).parts else "runtime"
        provider_groups[name].extend(_read_jsonl(path))
    summary["operations"]["provider_request_events"] = {
        name: {
            "completed_requests": len(events),
            "http_status_counts": dict(Counter(str(event.get("http_status")) for event in events)),
        }
        for name, events in provider_groups.items()
    }
    summary["provenance"] = {
        "run_root": str(args.run_root.resolve()),
        "prompt_version": manifest["settings"]["prompt_version"],
        "judge_contract_digest": manifest["judge_contract_digest"],
        "implementation_sha256": manifest["implementation_sha256"],
        "source_digests": {
            str(p.resolve()): sha256_file(p)
            for p in (
                args.labels,
                args.residual_subset,
                metadata_path,
                args.run_root / "run_manifest.json",
                Path(__file__),
                Path(__file__).with_name("judge_calibration.py"),
            )
        },
    }
    summary["historical_context"] = {}
    for root in args.historical_run:
        old_manifest = json.loads((root / "run_manifest.json").read_text())
        old_complete = json.loads((root / "run_complete.json").read_text())
        old_path = root / "data/judge_results.jsonl"
        require(
            sha256_file(old_path) == old_complete["results_sha256"],
            "DIAGNOSTIC_ARTIFACT_CHANGED",
            "historical results changed",
        )
        old_predictions = _read_jsonl(old_path)
        summary["historical_context"][root.name] = {
            "metrics_on_current_labels": evaluate_predictions(labels, old_predictions),
            "run_root": str(root.resolve()),
            "results_sha256": sha256_file(old_path),
            "prompt_version": old_manifest["settings"]["prompt_version"],
            "interpretation": "Historical outputs rescored on the same current development labels. Payload representations, prompts and runtime contracts differ; not a controlled equal-payload ablation.",
        }
    if args.replay_v0629:
        require(
            manifest["settings"]["prompt_version"] == "dedup-judge-hs-v0.6.2.9",
            "DIAGNOSTIC_REPLAY_POLICY",
            "arbitration replay requires V0.6.2.9",
        )
        summary["arbitration_replay"] = replay_v0629_arbitration(args.run_root, labels, predictions, payloads)
    _write_csv(args.rows, rows)
    errors = [row for row in rows if row["error_type"] != "CORRECT"]
    if errors:
        _write_csv(args.errors, errors)
    write_json_atomic(args.output, summary)
    print(
        json.dumps({"metrics": summary["metrics"]["weighted"], "distribution": summary["primary_error_distribution"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
