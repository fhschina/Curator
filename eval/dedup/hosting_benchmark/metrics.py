# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Deterministic accounting and reporting for hosting benchmark results."""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

from .artifacts import load_run, read_json, read_jsonl
from .relay import audit_paired_request_events

CORE_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _attempt_root(run_root: Path, marker: dict[str, Any]) -> Path:
    return run_root / marker["attempt_root"]


def _terminal_rows(run_root: Path, marker: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(_attempt_root(run_root, marker) / "terminal.jsonl")


def _event_rows(run_root: Path, marker: dict[str, Any]) -> list[dict[str, Any]]:
    root = _attempt_root(run_root, marker) / "events"
    return [row for path in sorted(root.glob("*.jsonl")) for row in read_jsonl(path)]


def _verify_marker_artifacts(run_root: Path, marker: dict[str, Any]) -> None:
    require(marker["status"] == "complete", "HOSTING_RUN_INCOMPLETE", "block did not complete cleanly")
    attempt = _attempt_root(run_root, marker)
    terminal_path = attempt / "terminal.jsonl"
    event_paths = sorted((attempt / "events").glob("*.jsonl"))
    require(
        sha256_file(terminal_path) == marker["terminal_sha256"],
        "HOSTING_ARTIFACT_CHANGED",
        "terminal artifact changed after block completion",
    )
    require(
        sha256_json([sha256_file(path) for path in event_paths]) == marker["event_sha256"],
        "HOSTING_ARTIFACT_CHANGED",
        "request events changed after block completion",
    )
    events = _event_rows(run_root, marker)
    require(
        all(event.get("generation_parameters_valid") is True for event in events)
        and not any(event.get("thinking_content_observed", False) for event in events),
        "HOSTING_GENERATION_CONTRACT_MISMATCH",
        "completed block violates the frozen generation contract",
    )
    require(
        not any(int(event["http_status"]) == 429 for event in events),
        "HOSTING_HUB_QUOTA_LIMITED",
        "completed block contains a 429 response",
    )
    require(
        not any(event.get("error_type") == "context_overflow" for event in events),
        "HOSTING_CONTEXT_OVERFLOW",
        "completed block contains a context overflow",
    )


def _initial_request_events(run_root: Path, marker: dict[str, Any]) -> list[dict[str, Any]]:
    events = [event for event in _event_rows(run_root, marker) if int(event["outer_attempt"]) == 1]
    require(events, "HOSTING_REQUEST_EVENTS_MISSING", "block has no initial request events")
    message_count = min(int(event["message_count"]) for event in events if event["message_count"] is not None)
    initial = [event for event in events if int(event["message_count"]) == message_count]
    require(
        len(initial) == int(marker["pairs"]),
        "HOSTING_REQUEST_ACCOUNTING_MISMATCH",
        "initial request count differs from the immutable block",
    )
    return initial


def _verify_paired_requests(run_root: Path, local: dict[str, Any], hub: dict[str, Any]) -> dict[str, Any]:
    return audit_paired_request_events(
        "local", _initial_request_events(run_root, local), "hub", _initial_request_events(run_root, hub)
    )


def _endpoint_concurrency_summary(
    run_root: Path, endpoint: str, concurrency: int, markers: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = [
        marker for marker in markers if marker["endpoint"] == endpoint and int(marker["concurrency"]) == concurrency
    ]
    require(len(selected) == 3, "HOSTING_ACCOUNTING_MISMATCH", "expected three block repetitions")
    terminals = [row for marker in selected for row in _terminal_rows(run_root, marker)]
    events = [row for marker in selected for row in _event_rows(run_root, marker)]
    durations = [float(marker["duration_seconds"]) for marker in selected]
    goodputs = [float(marker["goodput_pairs_per_second"]) for marker in selected]
    valid = sum(row["record_type"] == "result" for row in terminals)
    pair_ids = [str(row["canonical_pair_id"]) for row in terminals]
    require(
        len(set(pair_ids)) == len(pair_ids),
        "HOSTING_ACCOUNTING_MISMATCH",
        "endpoint/concurrency records contain duplicate pair IDs",
    )
    for event in events:
        require(
            isinstance(event.get("prompt_tokens_local"), int)
            and not isinstance(event.get("prompt_tokens_local"), bool),
            "HOSTING_CANONICAL_PROMPT_USAGE_MISSING",
            "relay omitted the pinned-tokenizer prompt count",
            endpoint=endpoint,
        )
    usage_rows = [row["usage"] for row in events if isinstance(row.get("usage"), dict)]
    prompt_tokens = sum(int(row["prompt_tokens_local"]) for row in events)
    provider_prompt_tokens = sum(int(row.get("prompt_tokens", 0)) for row in usage_rows)
    completion_tokens = sum(int(row.get("completion_tokens", 0)) for row in usage_rows)
    completion_lengths = [float(row["completion_tokens"]) for row in usage_rows if "completion_tokens" in row]
    total_wall_seconds = sum(durations)
    attempts = sum(int(row["attempts"]) for row in terminals)
    return {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "blocks": len(selected),
        "pairs": len(terminals),
        "unique_pairs": len(set(pair_ids)),
        "valid": valid,
        "failed": len(terminals) - valid,
        "valid_rate": valid / len(terminals),
        "first_attempt_valid_rate": sum(
            row["record_type"] == "result" and int(row["attempts"]) == 1 for row in terminals
        )
        / len(terminals),
        "total_wall_seconds": total_wall_seconds,
        "aggregate_goodput_pairs_per_second": valid / total_wall_seconds,
        "block_goodput_median": statistics.median(goodputs),
        "block_goodput_min": min(goodputs),
        "block_goodput_max": max(goodputs),
        "request_latency_p50_seconds": percentile([float(row["duration_seconds"]) for row in events], 0.50),
        "request_latency_p95_seconds": percentile([float(row["duration_seconds"]) for row in events], 0.95),
        "request_latency_p99_seconds": percentile([float(row["duration_seconds"]) for row in events], 0.99),
        "http_attempts": len(events),
        "raw_requests_per_wall_second": len(events) / total_wall_seconds,
        "pair_attempts": attempts,
        "retried_pairs": sum(bool(row.get("retried")) for row in terminals),
        "terminal_errors": len(terminals) - valid,
        "max_observed_outstanding": max(int(row.get("outstanding_at_submit", 0)) for row in events),
        "prompt_tokens": prompt_tokens,
        "prompt_token_count_source": "pinned_client_tokenizer",
        "provider_reported_prompt_tokens": provider_prompt_tokens,
        "provider_reported_prompt_token_delta": provider_prompt_tokens - prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_per_wall_second": prompt_tokens / total_wall_seconds,
        "completion_tokens_per_wall_second": completion_tokens / total_wall_seconds,
        "total_tokens_per_wall_second": (prompt_tokens + completion_tokens) / total_wall_seconds,
        "completion_tokens_p50": percentile(completion_lengths, 0.50),
        "completion_tokens_p95": percentile(completion_lengths, 0.95),
        "completion_tokens_p99": percentile(completion_lengths, 0.99),
        "http_429": sum(int(row["http_status"]) == 429 for row in events),
        "http_5xx": sum(500 <= int(row["http_status"]) <= 599 for row in events),
        "http_status_counts": {
            str(status): count for status, count in sorted(Counter(int(row["http_status"]) for row in events).items())
        },
        "timeouts": sum(row.get("error_type") == "timeout" for row in events),
        "context_overflow": sum(row.get("error_type") == "context_overflow" for row in events),
    }


def _gpu_summary(path: Path, markers: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0}
    intervals = [
        (datetime.fromisoformat(marker["started_at_utc"]), datetime.fromisoformat(marker["completed_at_utc"]))
        for marker in markers
        if marker["endpoint"] == "local"
    ]
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            timestamp = datetime.fromisoformat(parts[0])
            if not any(start <= timestamp <= end for start, end in intervals):
                continue
            samples.append(
                {
                    "utilization_gpu_percent": float(parts[3]),
                    "memory_used_mib": float(parts[4]),
                    "power_draw_watts": float(parts[5]),
                    "sm_clock_mhz": float(parts[6]),
                }
            )
        except ValueError:
            continue
    result: dict[str, Any] = {"samples": len(samples)}
    for field in ("utilization_gpu_percent", "memory_used_mib", "power_draw_watts", "sm_clock_mhz"):
        values = [row[field] for row in samples]
        result[field] = {
            "mean": statistics.fmean(values) if values else None,
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return result


def _agreement(run_root: Path, markers: list[dict[str, Any]]) -> dict[str, Any]:
    by_endpoint: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for marker in markers:
        for row in _terminal_rows(run_root, marker):
            if row["record_type"] == "result":
                by_endpoint[marker["endpoint"]][row["canonical_pair_id"]] = row
    common = sorted(set(by_endpoint["local"]) & set(by_endpoint["hub"]))
    return {
        "common_valid_pairs": len(common),
        "all_core_fields": sum(
            all(by_endpoint["local"][pair_id][field] == by_endpoint["hub"][pair_id][field] for field in CORE_FIELDS)
            for pair_id in common
        )
        / len(common)
        if common
        else None,
        "by_field": {
            field: sum(
                by_endpoint["local"][pair_id][field] == by_endpoint["hub"][pair_id][field] for pair_id in common
            )
            / len(common)
            if common
            else None
            for field in CORE_FIELDS
        },
    }


def summarize(run_root: str | Path) -> dict[str, Any]:
    root, manifest, config = load_run(run_root)
    completion = read_json(root / "run_complete.json")
    require(completion["status"] == "complete", "HOSTING_RUN_INCOMPLETE", "benchmark run is incomplete")
    markers = completion["measured"]
    for marker in [*completion["warmup"].values(), *markers]:
        require(
            marker["contract_digest"] == completion["contract_digest"],
            "HOSTING_CONTRACT_CHANGED",
            "block contract differs from the completed run contract",
        )
        _verify_marker_artifacts(root, marker)
    warmup_audit = _verify_paired_requests(root, completion["warmup"]["local"], completion["warmup"]["hub"])
    paired_markers: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for marker in markers:
        paired_markers[marker["block_id"]][marker["endpoint"]] = marker
    measured_audits = []
    for pair in paired_markers.values():
        require(set(pair) == {"local", "hub"}, "HOSTING_ACCOUNTING_MISMATCH", "paired block is incomplete")
        measured_audits.append(_verify_paired_requests(root, pair["local"], pair["hub"]))
    measured_provider_drift = Counter()
    for audit in measured_audits:
        measured_provider_drift.update(
            {int(delta): int(count) for delta, count in audit["provider_prompt_token_usage_delta_counts"].items()}
        )
    rows = [
        _endpoint_concurrency_summary(root, endpoint, concurrency, markers)
        for concurrency in config.workload.concurrencies
        for endpoint in ("local", "hub")
    ]
    endpoint_pair_ids = {
        endpoint: [
            str(row["canonical_pair_id"])
            for marker in markers
            if marker["endpoint"] == endpoint
            for row in _terminal_rows(root, marker)
        ]
        for endpoint in ("local", "hub")
    }
    endpoint_totals = {
        endpoint: {
            "pairs": sum(row["pairs"] for row in rows if row["endpoint"] == endpoint),
            "unique_pairs": len(set(endpoint_pair_ids[endpoint])),
            "valid": sum(row["valid"] for row in rows if row["endpoint"] == endpoint),
            "failed": sum(row["failed"] for row in rows if row["endpoint"] == endpoint),
            "http_429": sum(row["http_429"] for row in rows if row["endpoint"] == endpoint),
            "context_overflow": sum(row["context_overflow"] for row in rows if row["endpoint"] == endpoint),
        }
        for endpoint in ("local", "hub")
    }
    expected = int(manifest["workload"]["measured_pairs_per_endpoint"])
    for totals in endpoint_totals.values():
        require(totals["pairs"] == expected, "HOSTING_ACCOUNTING_MISMATCH", "endpoint pair count differs")
        require(totals["unique_pairs"] == expected, "HOSTING_ACCOUNTING_MISMATCH", "endpoint pair IDs are not unique")
        require(totals["http_429"] == 0, "HOSTING_HUB_QUOTA_LIMITED", "formal run contains 429")
        require(totals["context_overflow"] == 0, "HOSTING_CONTEXT_OVERFLOW", "formal run overflowed context")
        require(totals["valid"] / totals["pairs"] >= 0.99, "HOSTING_VALID_RATE_TOO_LOW", "valid rate below 99%")
    c8 = {row["endpoint"]: row for row in rows if row["concurrency"] == 8}
    paired_c8 = []
    by_block: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for marker in markers:
        if int(marker["concurrency"]) == 8:
            by_block[marker["block_id"]][marker["endpoint"]] = marker
    for block_id in sorted(by_block):
        pair = by_block[block_id]
        paired_c8.append(
            float(pair["hub"]["goodput_pairs_per_second"]) / float(pair["local"]["goodput_pairs_per_second"])
        )
    summary = {
        "schema_version": "hosting-benchmark-summary-v2",
        "run_id": manifest["run_id"],
        "status": "pass",
        "comparison_scope": "same canonical chat request, advertised model/precision matched black-box comparison",
        "cold_start_seconds": completion["cold_start_seconds"],
        "endpoint_totals": endpoint_totals,
        "by_concurrency": rows,
        "prompt_accounting": {
            "basis": "pinned_client_tokenizer",
            "canonical_request_and_prompt_token_equality": True,
            "warmup": warmup_audit,
            "measured_paired_requests": sum(int(audit["request_count"]) for audit in measured_audits),
            "measured_provider_prompt_token_usage_equality": not measured_provider_drift,
            "measured_provider_prompt_token_usage_mismatched_requests": sum(measured_provider_drift.values()),
            "measured_provider_prompt_token_usage_delta_counts": {
                str(delta): count for delta, count in sorted(measured_provider_drift.items())
            },
        },
        "agreement": _agreement(root, markers),
        "local_gpu": _gpu_summary(root / "gpu_samples.csv", markers),
        "blocks": [
            {
                key: marker[key]
                for key in (
                    "endpoint",
                    "block_id",
                    "concurrency",
                    "pairs",
                    "valid",
                    "failed",
                    "attempts",
                    "started_at_utc",
                    "completed_at_utc",
                    "duration_seconds",
                    "goodput_pairs_per_second",
                    "http_status_counts",
                    "status",
                )
            }
            for marker in markers
        ],
        "headline": {
            "concurrency": 8,
            "local_valid_pairs_per_second": c8["local"]["aggregate_goodput_pairs_per_second"],
            "hub_valid_pairs_per_second": c8["hub"]["aggregate_goodput_pairs_per_second"],
            "hub_over_local_goodput_ratio_median": statistics.median(paired_c8),
            "hub_over_local_goodput_ratio_min": min(paired_c8),
            "hub_over_local_goodput_ratio_max": max(paired_c8),
            "local_latency_p50_seconds": c8["local"]["request_latency_p50_seconds"],
            "local_latency_p95_seconds": c8["local"]["request_latency_p95_seconds"],
            "hub_latency_p50_seconds": c8["hub"]["request_latency_p50_seconds"],
            "hub_latency_p95_seconds": c8["hub"]["request_latency_p95_seconds"],
        },
    }
    write_json_atomic(root / "summary.json", summary)
    csv_path = root / "summary_by_concurrency.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    block_csv_path = root / "summary_by_block.csv"
    with block_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary["blocks"][0]))
        writer.writeheader()
        writer.writerows(summary["blocks"])
    headline = summary["headline"]
    report = f"""# Qwen3.8-27B-FP8 Hosting Benchmark

Status: **PASS**

This is a same-canonical-chat-request, advertised model/precision matched black-box hosting comparison. The pinned client tokenizer is the common prompt-token accounting basis. NVIDIA Inference Hub does not disclose the remote checkpoint revision, tokenizer revision, or serving hardware.

At concurrency 8, Hub/Local achieved {headline["hub_valid_pairs_per_second"]:.4f}/{headline["local_valid_pairs_per_second"]:.4f} schema-valid pairs/s, a median {headline["hub_over_local_goodput_ratio_median"]:.4f}x Hub/Local throughput ratio across three paired blocks (range {headline["hub_over_local_goodput_ratio_min"]:.4f}-{headline["hub_over_local_goodput_ratio_max"]:.4f}). Local p50/p95 request latency was {headline["local_latency_p50_seconds"]:.4f}/{headline["local_latency_p95_seconds"]:.4f}s; Hub p50/p95 was {headline["hub_latency_p50_seconds"]:.4f}/{headline["hub_latency_p95_seconds"]:.4f}s.

All {expected:,} measured unique pairs per endpoint passed canonical-request, pinned-tokenizer, quota, context, accounting, and >=99% schema-valid completion gates. Provider-reported prompt usage differed on {summary["prompt_accounting"]["measured_provider_prompt_token_usage_mismatched_requests"]:,} paired initial requests; this drift is reported as black-box service telemetry and is not used for comparable prompt-token throughput.
"""
    write_text_atomic(root / "RESULTS.md", report)
    return summary
