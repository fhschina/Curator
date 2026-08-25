# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static Pair Explorer data preparation and HTML rendering."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from eval.dedup.judging.schema import flatten_reason_codes
from eval.dedup.validation import require

PAIR_EXPLORER_VERSION = "dedup-pair-explorer-v3"
GROUP_MEMBER_SAMPLE_LIMIT = 12


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def pair_explorer_destination(report_destination: Path) -> Path:
    """Return the run-scoped dashboard path paired with an automated report path."""

    stem = report_destination.stem
    if stem == "final_report":
        label = ""
    elif stem.startswith("final_report."):
        label = stem.removeprefix("final_report")
    else:
        label = f".{stem}"
    return report_destination.with_name(f"pair_explorer{label}.html")


def _clip_visible_text(text: str, *, center: int | None, limit: int = 640) -> str:
    if len(text) <= limit:
        return text.strip()
    start = 0 if center is None else max(0, min(center - limit // 3, len(text) - limit))
    end = min(len(text), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _visible_excerpt(
    payload: dict[str, Any],
    *,
    side: str,
    evidence: list[dict[str, Any]],
) -> str:
    document = payload["document_a" if side == "A" else "document_b"]
    side_evidence = [item for item in evidence if item.get("side") == side]
    text = document.get("text")
    if isinstance(text, str):
        center = int(side_evidence[0]["start_char"]) if side_evidence else None
        return _clip_visible_text(text, center=center)
    windows = [item for item in payload["long_document_evidence"]["windows"] if item.get("side") == side]
    if not windows:
        return "No judge-visible text was available."
    if side_evidence:
        start_char = int(side_evidence[0]["start_char"])
        selected = next(
            (item for item in windows if int(item["start_char"]) <= start_char <= int(item["end_char"])),
            windows[0],
        )
        center = start_char - int(selected["start_char"])
    else:
        selected = windows[0]
        center = None
    return _clip_visible_text(str(selected["text"]), center=center)


def _load_endpoint_outcomes(path: Path, doc_ids: set[int]) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for Pair Explorer endpoint context"
        raise RuntimeError(msg) from exc
    columns = [
        "doc_id",
        "predicted_group_id",
        "predicted_cluster_key",
        "predicted_group_size",
        "action",
        "final_keeper_id",
    ]
    table = pq.read_table(path, columns=columns, filters=[("doc_id", "in", sorted(doc_ids))])
    outcomes = {int(row["doc_id"]): row for row in table.to_pylist()}
    require(
        len(outcomes) == len(doc_ids),
        "PAIR_EXPLORER_ENDPOINT_JOIN_INCOMPLETE",
        "Pair Explorer requires one SUT outcome per endpoint",
        expected=len(doc_ids),
        actual=len(outcomes),
    )
    return outcomes


def _endpoint_metadata(
    row: dict[str, Any],
    doc_id: int,
    endpoint_outcomes: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    low = int(row["doc_id_low"])
    suffix = "low" if doc_id == low else "high"
    outcome = endpoint_outcomes[doc_id]
    return {
        "doc_id": str(doc_id),
        "url": row.get(f"url_{suffix}"),
        "hostname": row.get(f"hostname_{suffix}"),
        "language": row.get(f"language_{suffix}"),
        "length_bucket": row.get(f"length_bucket_{suffix}"),
        "token_count": int(row.get(f"token_count_{suffix}") or 0),
        "excerpt": "",
        "sut_action": str(outcome["action"]),
        "predicted_group_id": int(outcome["predicted_group_id"]),
        "predicted_cluster_key": str(outcome["predicted_cluster_key"]),
        "predicted_group_size": int(outcome["predicted_group_size"]),
        "final_keeper_id": str(outcome["final_keeper_id"]),
    }


def _best_numeric(events: list[dict[str, Any]], field: str, *, minimum: bool) -> int | float | None:
    values = [item[field] for item in events if item.get(field) is not None]
    if not values:
        return None
    return min(values) if minimum else max(values)


def _evaluation_provenance(events: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [item for item in events if item.get("track") == "5b"]
    removal = [item for item in events if item.get("track") == "5a"]
    return {
        "retrieval": {
            "sources": sorted({str(item["retriever_bitmask"]) for item in retrieval if item.get("retriever_bitmask")}),
            "cosine": _best_numeric(retrieval, "cosine", minimum=False),
            "jaccard": _best_numeric(retrieval, "jaccard", minimum=False),
            "containment": _best_numeric(retrieval, "containment", minimum=False),
            "lexical_rank": _best_numeric(retrieval, "lexical_rank", minimum=True),
            "semantic_rank": _best_numeric(retrieval, "semantic_rank", minimum=True),
            "selection_rules": sorted(
                {str(item["selection_rule"]) for item in retrieval if item.get("selection_rule")}
            ),
            "semantic_high_lexical_low": any(bool(item.get("semantic_high_lexical_low")) for item in retrieval),
        },
        "removal_sampling": {
            "selection_rules": sorted({str(item["selection_rule"]) for item in removal if item.get("selection_rule")}),
            "frame_size": _best_numeric(removal, "frame_size", minimum=False),
            "selection_probability": _best_numeric(removal, "selection_probability", minimum=False),
            "pair_seed": _best_numeric(removal, "pair_seed", minimum=True),
        },
        "event_count": len(events),
    }


def _evidence_status(result: dict[str, Any], *, judge_status: str) -> dict[str, Any]:
    if judge_status != "valid":
        return {
            "returned": 0,
            "retained": 0,
            "realigned": 0,
            "dropped": 0,
            "coverage": "UNAVAILABLE",
        }
    retained = len(result.get("evidence", []))
    repair_events = result.get("deterministic_repair_events", [])
    dropped = sum(item.get("action") == "drop_unalignable" for item in repair_events)
    realigned = sum(item.get("action") == "realign_offsets" for item in repair_events)
    returned = retained + dropped
    coverage = "NONE" if retained == 0 else "COMPLETE" if retained == returned else "PARTIAL"
    return {
        "returned": returned,
        "retained": retained,
        "realigned": realigned,
        "dropped": dropped,
        "coverage": coverage,
    }


def _directional_answer(result: dict[str, Any], *, source_id: int, target_id: int, a_id: int, b_id: int) -> Any:
    if not result:
        return None
    if source_id == a_id and target_id == b_id:
        return result.get("a_can_replace_b")
    if source_id == b_id and target_id == a_id:
        return result.get("b_can_replace_a")
    msg = "directional replacement endpoint does not match Judge presentation"
    raise ValueError(msg)


def _friendly_outcome(outcomes: list[str], judge_status: str) -> str:
    labels = {
        "wrong_removal": "WRONG_REMOVAL",
        "safe_removal": "SAFE_REMOVAL",
        "discovered_candidate_fn": "MISSED_DUPLICATE",
        "hard_negative": "CONFIRMED_DIFFERENT",
        "unresolved": "UNRESOLVED",
        "judge_error": "JUDGE_ERROR",
    }
    resolved = [labels[item] for item in outcomes if item in labels]
    return " + ".join(resolved) if resolved else judge_status.upper()


def _sut_provenance_status(sut_manifest: dict[str, Any], *, has_5a: bool) -> dict[str, str]:
    available = sut_manifest.get("provenance_availability", {})
    if has_5a:
        edge_status = "AVAILABLE_UPSTREAM" if available.get("candidate_edges") else "NOT_PRESERVED"
        lineage_status = edge_status
    else:
        edge_status = "NOT_APPLICABLE"
        lineage_status = "NOT_APPLICABLE"
    return {
        "group_membership": "AVAILABLE",
        "final_keeper_mapping": "AVAILABLE",
        "direct_candidate_edge": edge_status,
        "edge_similarity_score": edge_status,
        "removal_lineage": lineage_status,
        "keeper_selection_reason": "NOT_PRESERVED",
        "resolved_sut_config": "AVAILABLE" if available.get("resolved_config") else "NOT_PRESERVED",
        "minhash_cache": "AVAILABLE" if available.get("minhash_cache") else "NOT_PRESERVED",
        "lsh_cache": "AVAILABLE" if available.get("lsh_cache") else "NOT_PRESERVED",
    }


def build_pair_explorer_records(run_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join Judge, SUT endpoint, sampling, and visible-text context for the dashboard."""

    results = {row["canonical_pair_id"]: row for row in _read_jsonl(run_root / "data" / "judge_results.jsonl")}
    errors = {row["canonical_pair_id"]: row for row in _read_jsonl(run_root / "logs" / "judge_errors.jsonl")}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for Pair Explorer provenance"
        raise RuntimeError(msg) from exc
    provenance_by_pair: dict[str, list[dict[str, Any]]] = {}
    for item in pq.read_table(run_root / "data" / "pair_provenance.parquet").to_pylist():
        provenance_by_pair.setdefault(str(item["canonical_pair_id"]), []).append(item)
    endpoints = {int(row["doc_id_low"]) for row in rows} | {int(row["doc_id_high"]) for row in rows}
    endpoint_outcomes = _load_endpoint_outcomes(run_root / "data" / "document_outcomes.parquet", endpoints)
    sut_manifest = json.loads((run_root / "manifests" / "sut_run_manifest.json").read_text(encoding="utf-8"))
    records = []
    for row in sorted(rows, key=lambda item: str(item["canonical_pair_id"])):
        pair_id = str(row["canonical_pair_id"])
        result = results.get(pair_id, {})
        has_5a = bool(row["has_track_5a"])
        has_5b = bool(row["has_track_5b"])
        low = int(row["doc_id_low"])
        high = int(row["doc_id_high"])
        if has_5a:
            left_id = int(row["keeper_doc_id"])
            right_id = int(row["removed_doc_id"])
            left_role, right_role = "Keeper", "Removed"
        else:
            left_id, right_id = low, high
            left_role, right_role = "Document low", "Document high"
        outcomes = []
        if has_5a and row.get("removal_outcome"):
            outcomes.append(str(row["removal_outcome"]))
        if has_5b and row.get("cross_group_outcome"):
            outcomes.append(str(row["cross_group_outcome"]))
        error_types = sorted(
            {str(item.get("error_type", "unknown")) for item in errors.get(pair_id, {}).get("errors", [])}
        )
        judge_status = str(row["judge_status"])
        a_id = int(row["presented_doc_a"])
        b_id = int(row["presented_doc_b"])
        same_group = result.get("same_duplicate_group") or row.get("same_duplicate_group")
        record = {
            "pair_id": pair_id,
            "track": "5a + 5b" if has_5a and has_5b else "5a" if has_5a else "5b",
            "has_5a": has_5a,
            "has_5b": has_5b,
            "outcomes": outcomes,
            "evaluation_outcome": _friendly_outcome(outcomes, judge_status),
            "judge_status": judge_status,
            "relation_type": result.get("relation_type") or row.get("relation_type"),
            "material_difference": result.get("material_difference") or row.get("material_difference"),
            "fuzzy_scope": result.get("fuzzy_scope") or row.get("fuzzy_scope"),
            "same_duplicate_group": same_group,
            "judge_group_verdict": {
                "YES": "DUPLICATE",
                "NO": "NOT_DUPLICATE",
                "UNRESOLVED": "UNRESOLVED",
            }.get(same_group, "UNAVAILABLE"),
            "keeper_can_replace_removed": row.get("keeper_can_replace_removed"),
            "left_can_replace_right": _directional_answer(
                result, source_id=left_id, target_id=right_id, a_id=a_id, b_id=b_id
            ),
            "right_can_replace_left": _directional_answer(
                result, source_id=right_id, target_id=left_id, a_id=a_id, b_id=b_id
            ),
            "judge_removal_verdict": (
                {"YES": "SAFE", "NO": "UNSAFE", "UNRESOLVED": "UNRESOLVED"}.get(
                    row.get("keeper_can_replace_removed"), "UNAVAILABLE"
                )
                if has_5a
                else "NOT_APPLICABLE"
            ),
            "confidence": result.get("confidence") if result else row.get("confidence"),
            "reason_codes": flatten_reason_codes(result.get("reason_codes", [])),
            "evidence": [],
            "judge_evidence_status": _evidence_status(result, judge_status=judge_status),
            "error_types": error_types,
            "group_size_bucket": row["report_group_size_bucket"],
            "predicted_group_size": int(row["predicted_group_size_low"]),
            "predicted_group_size_low": int(row["predicted_group_size_low"]),
            "predicted_group_size_high": int(row["predicted_group_size_high"]),
            "sut_grouping_result": "SAME_GROUP" if bool(row["predicted_same_group"]) else "DIFFERENT_GROUPS",
            "token_length_ratio": round(float(row["token_length_ratio"]), 6),
            "token_length_ratio_bucket": row["report_ratio_bucket"],
            "same_hostname": bool(row["same_hostname"]),
            "retriever_category": row.get("retriever_category"),
            "left_role": left_role,
            "right_role": right_role,
            "left": _endpoint_metadata(row, left_id, endpoint_outcomes),
            "right": _endpoint_metadata(row, right_id, endpoint_outcomes),
            "evaluation_provenance": _evaluation_provenance(provenance_by_pair.get(pair_id, [])),
            "sut_provenance": _sut_provenance_status(sut_manifest, has_5a=has_5a),
            "group_context_ids": [],
            "risk_indicators": [],
            "presented_doc_a": str(a_id),
            "presented_doc_b": str(b_id),
            "raw_evidence": list(result.get("evidence", [])),
        }
        records.append(record)
    by_pair = {row["pair_id"]: row for row in records}
    matched_payloads = 0
    with (run_root / "data" / "judge_payloads.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload_row = json.loads(line)
            record = by_pair.get(str(payload_row["canonical_pair_id"]))
            if record is None:
                continue
            matched_payloads += 1
            payload = payload_row["payload"]
            evidence = record.pop("raw_evidence")
            side_by_doc = {
                record["presented_doc_a"]: "A",
                record["presented_doc_b"]: "B",
            }
            for key in ("left", "right"):
                side = side_by_doc[record[key]["doc_id"]]
                record[key]["excerpt"] = _visible_excerpt(payload, side=side, evidence=evidence)
            for item in evidence:
                presented_id = record["presented_doc_a"] if item["side"] == "A" else record["presented_doc_b"]
                role = record["left_role"] if presented_id == record["left"]["doc_id"] else record["right_role"]
                record["evidence"].append(
                    {
                        "role": role,
                        "quote": str(item["quote"]),
                        "side": str(item["side"]),
                        "start_char": int(item["start_char"]),
                        "end_char": int(item["end_char"]),
                    }
                )
    require(
        matched_payloads == len(records),
        "PAIR_EXPLORER_PAYLOAD_JOIN_INCOMPLETE",
        "Pair Explorer requires one Judge-visible payload per comparison row",
        comparisons=len(records),
        payloads=matched_payloads,
    )
    for record in records:
        record.pop("presented_doc_a")
        record.pop("presented_doc_b")
        record.pop("raw_evidence", None)
    return records


def pair_explorer_review_queue(
    records: list[dict[str, Any]], examples: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Keep actionable pairs and report controls in the static dashboard."""

    example_ids = {row["pair_id"] for group in examples.values() for row in group}
    return [
        row
        for row in records
        if row["pair_id"] in example_ids
        or "wrong_removal" in row["outcomes"]
        or "discovered_candidate_fn" in row["outcomes"]
        or "unresolved" in row["outcomes"]
        or row["judge_status"] != "valid"
    ]


def _member_sample(rows: list[dict[str, Any]], required_doc_ids: set[int]) -> list[dict[str, Any]]:
    by_id = {int(row["doc_id"]): row for row in rows}
    selected: list[int] = []

    def add(doc_id: int) -> None:
        if doc_id in by_id and doc_id not in selected and len(selected) < GROUP_MEMBER_SAMPLE_LIMIT:
            selected.append(doc_id)

    for doc_id in sorted(required_doc_ids):
        add(doc_id)
    keepers = sorted(int(row["doc_id"]) for row in rows if str(row["action"]) == "KEEP")
    for doc_id in keepers:
        add(doc_id)
    first_by_host: dict[str, int] = {}
    for row in sorted(rows, key=lambda item: int(item["doc_id"])):
        first_by_host.setdefault(str(row.get("hostname") or "(missing)"), int(row["doc_id"]))
    for doc_id in first_by_host.values():
        add(doc_id)
    for doc_id in sorted(by_id):
        add(doc_id)
    return [
        {
            "doc_id": str(row["doc_id"]),
            "action": str(row["action"]),
            "final_keeper_id": str(row["final_keeper_id"]),
            "hostname": row.get("hostname"),
            "language": row.get("language"),
            "token_count": int(row["token_count"]),
            "url": row.get("url"),
        }
        for row in (by_id[doc_id] for doc_id in selected)
    ]


def _group_context_summary(
    group_id: int,
    rows: list[dict[str, Any]],
    required_doc_ids: set[int],
) -> dict[str, Any]:
    hostnames = Counter(str(row.get("hostname") or "(missing)") for row in rows)
    languages = Counter(str(row.get("language") or "(missing)") for row in rows)
    token_counts = sorted(int(row["token_count"]) for row in rows)
    first = rows[0]
    return {
        "group_id": group_id,
        "cluster_key": str(first["predicted_cluster_key"]),
        "group_size": len(rows),
        "hostname_count": len(hostnames),
        "top_hostnames": [{"value": value, "count": count} for value, count in hostnames.most_common(8)],
        "languages": [{"value": value, "count": count} for value, count in languages.most_common()],
        "token_count_min": token_counts[0],
        "token_count_median": median(token_counts),
        "token_count_max": token_counts[-1],
        "members": _member_sample(rows, required_doc_ids),
        "member_sample_limit": GROUP_MEMBER_SAMPLE_LIMIT,
    }


def attach_group_context(
    run_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach bounded group references and return de-duplicated group summaries."""

    required_by_group: dict[int, set[int]] = {}
    for record in records:
        for endpoint in (record["left"], record["right"]):
            group_id = int(endpoint["predicted_group_id"])
            if group_id >= 0 and int(endpoint["predicted_group_size"]) > 1:
                required_by_group.setdefault(group_id, set()).add(int(endpoint["doc_id"]))
    if not required_by_group:
        return {}
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for Pair Explorer group context"
        raise RuntimeError(msg) from exc
    columns = [
        "doc_id",
        "predicted_group_id",
        "predicted_cluster_key",
        "action",
        "final_keeper_id",
        "token_count",
        "hostname",
        "language",
        "url",
    ]
    table = pq.read_table(
        run_root / "data" / "document_outcomes.parquet",
        columns=columns,
        filters=[("predicted_group_id", "in", sorted(required_by_group))],
    )
    by_group: dict[int, list[dict[str, Any]]] = {}
    for row in table.to_pylist():
        by_group.setdefault(int(row["predicted_group_id"]), []).append(row)
    require(
        set(by_group) == set(required_by_group),
        "PAIR_EXPLORER_GROUP_JOIN_INCOMPLETE",
        "Pair Explorer group context is missing requested groups",
        expected=len(required_by_group),
        actual=len(by_group),
    )
    contexts = {
        str(group_id): _group_context_summary(group_id, rows, required_by_group[group_id])
        for group_id, rows in sorted(by_group.items())
    }
    for record in records:
        group_ids = []
        for endpoint in (record["left"], record["right"]):
            group_id = int(endpoint["predicted_group_id"])
            if str(group_id) in contexts and group_id not in group_ids:
                group_ids.append(group_id)
        record["group_context_ids"] = [str(group_id) for group_id in group_ids]
        maximum_group_size = max(
            int(record["left"]["predicted_group_size"]), int(record["right"]["predicted_group_size"])
        )
        evidence_coverage = record["judge_evidence_status"]["coverage"]
        record["risk_indicators"] = [
            {
                "label": "Large group",
                "value": "YES" if maximum_group_size >= 21 else "NO",
                "note": "At least one endpoint belongs to a group with 21+ documents.",
            },
            {
                "label": "Cross-host pair",
                "value": "YES" if not record["same_hostname"] else "NO",
                "note": "Cross-host overlap can be legitimate duplication or shared boilerplate.",
            },
            {
                "label": "Judge evidence coverage",
                "value": evidence_coverage,
                "note": "Quote coverage describes auditability, not verdict correctness.",
            },
            {
                "label": "Possible template-driven grouping",
                "value": (
                    "POSSIBLE"
                    if maximum_group_size >= 21
                    and (
                        not record["same_hostname"]
                        or any(
                            code == "BOILERPLATE"
                            or code == "PRIMARY_RISK:BOILERPLATE_DOMINATED_SIMILARITY"
                            or code == "PRIMARY_RISK:TEMPLATE_SLOT_COLLISION"
                            for code in record["reason_codes"]
                        )
                    )
                    else "NO_SIGNAL"
                ),
                "note": "Heuristic only; SUT candidate edges were not preserved.",
            },
        ]
    return contexts


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def pair_explorer_html(
    *,
    evaluation_run_id: str,
    records: list[dict[str, Any]],
    group_contexts: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the self-contained static Pair Explorer."""

    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';script-src 'unsafe-inline';base-uri 'none';form-action 'none'">
<title>Dedup Pair Explorer</title>
<style>
:root{color-scheme:light dark;--bg:#f4f6f9;--panel:#fff;--panel2:#f8fafc;--text:#172033;--muted:#647087;--line:#d8dee9;--accent:#2457d6;--danger:#b42318;--danger-bg:#fff1f0;--good:#18794e;--good-bg:#edf9f3;--warn:#946200;--warn-bg:#fff8e6;--neutral:#475467;--neutral-bg:#f2f4f7}
@media(prefers-color-scheme:dark){:root{--bg:#10141c;--panel:#171d28;--panel2:#1c2431;--text:#eef2f8;--muted:#aab4c4;--line:#303949;--accent:#8aa9ff;--danger:#ff8a80;--danger-bg:#351b1b;--good:#66d19e;--good-bg:#173126;--warn:#f2c14e;--warn-bg:#332b17;--neutral:#c3cad5;--neutral-bg:#252d39}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}header{padding:20px 28px 14px;border-bottom:1px solid var(--line);background:var(--panel)}h1{margin:0 0 4px;font-size:24px}h2{margin:0;font-size:18px}h3{margin:0;font-size:15px}.muted,.metadata{color:var(--muted)}.toolbar,.panel-head,.section-head,.pagination,.review-actions{display:flex;align-items:center;gap:10px}.toolbar{flex-wrap:wrap;margin-top:12px}.panel-head,.section-head{justify-content:space-between}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:16px 28px 0}.stat,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.stat{padding:10px 12px}.stat strong{display:block;font-size:20px}.controls{display:flex;flex-wrap:wrap;gap:8px;padding:14px 28px}input,select,button,textarea{border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--text);padding:7px 9px;font:inherit}input,select,button{min-height:36px}button{cursor:pointer}textarea{width:100%;min-height:92px;resize:vertical}#search{min-width:270px;flex:1}.workspace{display:grid;grid-template-columns:minmax(500px,.9fr) minmax(560px,1.1fr);gap:14px;padding:0 28px 28px}.panel{min-width:0;overflow:hidden}.panel-head{padding:12px 14px;border-bottom:1px solid var(--line)}.table-wrap{overflow:auto;max-height:calc(100vh - 300px)}table{width:100%;border-collapse:collapse}th,td{padding:9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-size:12px}tr.pair-row{cursor:pointer}tr.pair-row:hover,tr.selected{background:color-mix(in srgb,var(--accent) 10%,transparent)}.pair-id{max-width:175px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:monospace}.detail{padding:16px;max-height:calc(100vh - 230px);overflow:auto}.pair-heading{overflow-wrap:anywhere;font-family:monospace}.decision-flow{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:15px 0}.decision-card{border:1px solid var(--line);border-top:4px solid var(--neutral);border-radius:8px;padding:11px;background:var(--panel2)}.decision-card span,.fact span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}.decision-card strong{display:block;margin:5px 0;font-size:17px}.tone-danger{border-top-color:var(--danger);background:var(--danger-bg)}.tone-good{border-top-color:var(--good);background:var(--good-bg)}.tone-warn{border-top-color:var(--warn);background:var(--warn-bg)}.detail-section{margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}.section-head{margin-bottom:9px}.section-note,.metadata{font-size:12px}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 14px}.fact strong{display:block;overflow-wrap:anywhere}.detail-grid,.sut-grid,.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.document,.sut-endpoint,.context-card,.review-card,.notice,.risk{border:1px solid var(--line);border-radius:7px;padding:12px;background:var(--panel2);min-width:0}.document h3,.sut-endpoint h3{margin-bottom:7px}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:10px;border-radius:6px;background:var(--bg);font:12px/1.5 monospace}.quote{margin:8px 0;padding:9px;border-left:3px solid var(--accent);background:var(--panel2);white-space:pre-wrap}.status-list{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px}.status-row{display:flex;justify-content:space-between;border-bottom:1px dotted var(--line)}.pill{border:1px solid currentColor;border-radius:999px;padding:1px 7px;font-size:11px}.pill-AVAILABLE{color:var(--good)}.pill-NOT_PRESERVED{color:var(--danger)}.context-card{margin-top:9px}details summary{cursor:pointer;font-weight:700}.context-table{margin-top:10px;font-size:12px}.context-table th{position:static}.risk strong{display:block}.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.review-grid select{width:100%}.review-actions{justify-content:flex-end;margin-top:8px}.glossary{margin-top:16px}.empty{padding:36px;text-align:center;color:var(--muted)}a{color:var(--accent)}.hidden{display:none}.outcome-wrong_removal{color:var(--danger);font-weight:700}.outcome-safe_removal{color:var(--good);font-weight:700}.outcome-discovered_candidate_fn{color:var(--warn);font-weight:700}
@media(max-width:1200px){.workspace{grid-template-columns:1fr}.table-wrap,.detail{max-height:none}}@media(max-width:760px){header,.stats,.controls,.workspace{padding-left:14px;padding-right:14px}.stats{grid-template-columns:1fr 1fr}.decision-flow,.facts,.detail-grid,.sut-grid,.risk-grid{grid-template-columns:1fr}}
</style><style>
#language{border-color:var(--accent)}
.reason-panel{margin:0 28px 14px}
.reason-head{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px;border-bottom:1px solid var(--line)}
.reason-summary{font-size:12px}
.reason-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px 22px;padding:10px 14px 12px}
.reason-row{display:grid;grid-template-columns:minmax(155px,.55fr) minmax(90px,1fr) auto;align-items:center;gap:9px;width:100%;min-height:0;padding:4px 2px;border:0;background:transparent;text-align:left}
.reason-row:hover .reason-track,.reason-row.active .reason-track{outline:2px solid var(--accent);outline-offset:1px}
.reason-row.active .reason-label{color:var(--accent);font-weight:700}
.reason-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.reason-track{height:8px;overflow:hidden;border-radius:999px;background:var(--neutral-bg)}
.reason-fill{display:block;height:100%;min-width:2px;border-radius:inherit;background:var(--accent)}
.reason-count{min-width:42px;text-align:right;font-variant-numeric:tabular-nums}
.reason-empty{padding:20px}
@media(max-width:760px){.reason-panel{margin-left:14px;margin-right:14px}.reason-grid{grid-template-columns:1fr}.reason-row{grid-template-columns:minmax(145px,.55fr) minmax(70px,1fr) auto}}
</style></head><body>
<header><h1>Dedup Pair Explorer</h1><div class="muted">Run <strong>__RUN_ID__</strong> · __EXPLORER_VERSION__ · Automated Judge labels are not human ground truth.</div>
<div class="toolbar"><button id="export-csv">Export reviews CSV</button><button id="export-json">Export reviews JSON</button><button id="import-reviews">Import reviews</button><input id="review-file" class="hidden" type="file" accept=".csv,.json"><span id="storage-status" class="muted">Reviews are stored in this browser.</span></div></header>
<section class="stats"><div class="stat"><span class="muted">Matching pairs</span><strong id="stat-total">0</strong></div><div class="stat"><span class="muted">Wrong removals</span><strong id="stat-wrong">0</strong></div><div class="stat"><span class="muted">Safe removals</span><strong id="stat-safe">0</strong></div><div class="stat"><span class="muted">Missed duplicates</span><strong id="stat-positive">0</strong></div><div class="stat"><span class="muted">Human reviewed</span><strong id="stat-reviewed">0</strong></div></section>
<section class="controls"><input id="search" type="search" placeholder="Search pair, document, group, hostname, reason…"><select id="track"><option value="">All tracks</option><option>5a</option><option>5b</option></select><select id="language" aria-label="Language, matching either document"><option value="">Any language · either document</option><option value="__same__">Same-language pairs</option><option value="__cross__">Cross-language pairs</option></select><select id="outcome"><option value="">All outcomes</option></select><select id="relation"><option value="">All relations</option></select><select id="material"><option value="">All differences</option></select><select id="scope"><option value="">All scopes</option></select><select id="group"><option value="">All group sizes</option></select><select id="retriever"><option value="">All retrievers</option></select><select id="evidence"><option value="">Any evidence</option><option>COMPLETE</option><option>PARTIAL</option><option>NONE</option><option>UNAVAILABLE</option></select><select id="review"><option value="">Any review</option><option>UNREVIEWED</option><option>REVIEWED</option><option>DISAGREE_WITH_JUDGE</option><option>UNSURE</option></select><select id="hostname"><option value="">Any hostname</option><option value="same">Same</option><option value="different">Different</option></select><button id="reset">Reset</button></section>
<section class="panel reason-panel" aria-labelledby="reason-title"><div class="reason-head"><div><h2 id="reason-title">Reason code distribution</h2><div id="reason-summary" class="reason-summary muted"></div></div><button id="reason-toggle" type="button" aria-expanded="false">Show all</button></div><div id="reason-grid" class="reason-grid"></div><div id="reason-empty" class="reason-empty muted" hidden>No reason codes match the current filters.</div></section>
<main class="workspace"><section class="panel"><div class="panel-head"><h2>Pairs</h2><div class="pagination"><button id="previous">Previous</button><span id="page-label" class="muted"></span><button id="next">Next</button></div></div><div class="table-wrap"><table><thead><tr><th>Track</th><th>Outcome</th><th>Judge</th><th>Evidence</th><th>Review</th><th>Pair ID</th></tr></thead><tbody id="pair-rows"></tbody></table><div id="no-results" class="empty" hidden>No matching pairs.</div></div></section><section class="panel"><div class="panel-head"><h2>Pair detail</h2><button id="copy-id" disabled>Copy pair ID</button></div><div id="detail" class="detail"><div class="empty">Select a pair.</div></div></section></main>
<script>
const PAIRS=__PAIR_DATA__,GROUPS=__GROUP_CONTEXT_DATA__,RUN_ID=__RUN_ID_JSON__,VERSION=__EXPLORER_VERSION_JSON__,PAGE_SIZE=100,REASON_LIMIT=8,STORE=`dedup-pair-reviews:${RUN_ID}:${VERSION}`,LEGACY_STORE=`dedup-pair-reviews:${RUN_ID}:dedup-pair-explorer-v2`,PAIR_IDS=new Set(PAIRS.map(x=>x.pair_id)),VALID_STATUS=new Set(["AGREE_WITH_JUDGE","DISAGREE_WITH_JUDGE","UNSURE"]),VALID_VERDICT=new Set(["","DUPLICATE","NOT_DUPLICATE","UNRESOLVED"]);
const ids=["search","track","language","outcome","relation","material","scope","group","retriever","evidence","review","hostname"],controls=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)])),tbody=document.getElementById("pair-rows"),detail=document.getElementById("detail"),reasonGrid=document.getElementById("reason-grid"),state={page:0,filtered:[],selected:null,reason:"",reasonExpanded:false};let reviews=loadReviews();
const create=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined&&text!==null)n.textContent=String(text);return n},value=v=>v===undefined||v===null||v===""?"—":String(v),number=v=>v===undefined||v===null?"—":Number(v).toFixed(3),raw=r=>r.outcomes.join(" + ")||r.judge_status,priority=r=>r.outcomes.includes("wrong_removal")?0:r.outcomes.includes("discovered_candidate_fn")?1:r.judge_status!=="valid"?2:3;
function fill(id,field){for(const v of[...new Set(PAIRS.map(r=>r[field]).filter(Boolean))].sort()){const o=create("option","",v);o.value=v;controls[id].appendChild(o)}}function fillLanguages(){const counts=new Map();for(const r of PAIRS)for(const v of new Set([r.left.language,r.right.language].filter(Boolean)))counts.set(v,(counts.get(v)||0)+1);for(const[v,count]of[...counts].sort((a,b)=>a[0].localeCompare(b[0]))){const o=create("option","",`${v} (${count.toLocaleString()})`);o.value=v;controls.language.appendChild(o)}}for(const v of[...new Set(PAIRS.flatMap(r=>r.outcomes))].sort()){const o=create("option","",v);o.value=v;controls.outcome.appendChild(o)}fillLanguages();fill("relation","relation_type");fill("material","material_difference");fill("scope","fuzzy_scope");fill("group","group_size_bucket");fill("retriever","retriever_category");
function loadReviews(){try{return JSON.parse(localStorage.getItem(STORE)||localStorage.getItem(LEGACY_STORE)||"{}")}catch(_){return{}}}function persist(){try{localStorage.setItem(STORE,JSON.stringify(reviews));document.getElementById("storage-status").textContent="Reviews saved in this browser."}catch(_){document.getElementById("storage-status").textContent="Storage unavailable; export before closing."}}function review(id){return reviews[id]||{review_status:"",human_verdict:"",notes:"",updated_at:""}}function reviewed(id){return VALID_STATUS.has(review(id).review_status)}
function matches(r,{includeReason=true}={}){const q=controls.search.value.trim().toLowerCase(),lv=controls.language.value,rv=review(r.pair_id),search=[r.pair_id,r.left.doc_id,r.right.doc_id,r.left.predicted_cluster_key,r.right.predicted_cluster_key,r.left.hostname,r.right.hostname,r.left.language,r.right.language,...r.reason_codes,...r.error_types].join(" ").toLowerCase(),rm=!controls.review.value||(controls.review.value==="UNREVIEWED"?!reviewed(r.pair_id):controls.review.value==="REVIEWED"?reviewed(r.pair_id):rv.review_status===controls.review.value),lm=!lv||(lv==="__same__"?r.left.language===r.right.language:lv==="__cross__"?r.left.language!==r.right.language:r.left.language===lv||r.right.language===lv);return(!q||search.includes(q))&&(!controls.track.value||(controls.track.value==="5a"?r.has_5a:r.has_5b))&&lm&&(!controls.outcome.value||r.outcomes.includes(controls.outcome.value))&&(!controls.relation.value||r.relation_type===controls.relation.value)&&(!controls.material.value||r.material_difference===controls.material.value)&&(!controls.scope.value||r.fuzzy_scope===controls.scope.value)&&(!controls.group.value||r.group_size_bucket===controls.group.value)&&(!controls.retriever.value||r.retriever_category===controls.retriever.value)&&(!controls.evidence.value||r.judge_evidence_status.coverage===controls.evidence.value)&&rm&&(!controls.hostname.value||(controls.hostname.value==="same")===r.same_hostname)&&(!includeReason||!state.reason||r.reason_codes.includes(state.reason))}
function stats(rows){for(const[id,count]of[["stat-total",rows.length],["stat-wrong",rows.filter(r=>r.outcomes.includes("wrong_removal")).length],["stat-safe",rows.filter(r=>r.outcomes.includes("safe_removal")).length],["stat-positive",rows.filter(r=>r.outcomes.includes("discovered_candidate_fn")).length],["stat-reviewed",rows.filter(r=>reviewed(r.pair_id)).length]])document.getElementById(id).textContent=count.toLocaleString()}
function renderReasonChart(){const rows=PAIRS.filter(r=>matches(r,{includeReason:false})),counts=new Map();for(const r of rows)for(const code of new Set(r.reason_codes))counts.set(code,(counts.get(code)||0)+1);const sorted=[...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])),maximum=sorted[0]?.[1]||1;let visible=state.reasonExpanded?sorted:sorted.slice(0,REASON_LIMIT);if(state.reason&&!visible.some(([code])=>code===state.reason)){const selected=sorted.find(([code])=>code===state.reason);if(selected)visible=[...visible.slice(0,Math.max(0,REASON_LIMIT-1)),selected]}reasonGrid.replaceChildren();for(const[code,count]of visible){const row=create("button",`reason-row${code===state.reason?" active":""}`),label=create("span","reason-label",code),track=create("span","reason-track"),fill=create("span","reason-fill"),amount=create("span","reason-count",count.toLocaleString());row.type="button";row.title=`Filter pairs by ${code}`;row.setAttribute("aria-pressed",String(code===state.reason));fill.style.width=`${Math.max(2,count/maximum*100)}%`;track.appendChild(fill);row.append(label,track,amount);row.onclick=()=>{state.reason=state.reason===code?"":code;applyFilters()};reasonGrid.appendChild(row)}const empty=document.getElementById("reason-empty"),toggle=document.getElementById("reason-toggle");empty.hidden=sorted.length!==0;reasonGrid.hidden=sorted.length===0;toggle.hidden=sorted.length<=REASON_LIMIT;toggle.textContent=state.reasonExpanded?`Show top ${REASON_LIMIT}`:`Show all (${sorted.length})`;toggle.setAttribute("aria-expanded",String(state.reasonExpanded));document.getElementById("reason-summary").textContent=`${rows.length.toLocaleString()} pairs after other filters · ${sorted.length} reason codes${state.reason?` · filtering by ${state.reason}`:""} · multi-label counts`}
function renderRows(){tbody.replaceChildren();const pages=Math.max(1,Math.ceil(state.filtered.length/PAGE_SIZE));state.page=Math.min(state.page,pages-1);for(const r of state.filtered.slice(state.page*PAGE_SIZE,(state.page+1)*PAGE_SIZE)){const tr=create("tr",`${r.pair_id===state.selected?"selected ":""}pair-row`),rv=review(r.pair_id);tr.append(create("td","",r.track),create("td",`outcome-${r.outcomes[0]||"other"}`,r.evaluation_outcome),create("td","",r.judge_group_verdict),create("td","",r.judge_evidence_status.coverage),create("td","",rv.review_status||"UNREVIEWED"),create("td","pair-id",r.pair_id));tr.onclick=()=>selectPair(r.pair_id);tbody.appendChild(tr)}document.getElementById("no-results").hidden=state.filtered.length!==0;document.getElementById("page-label").textContent=`${state.page+1} / ${pages}`;document.getElementById("previous").disabled=state.page===0;document.getElementById("next").disabled=state.page+1>=pages}
function fact(c,label,v){const d=create("div","fact");d.append(create("span","",label),create("strong","",value(v)));c.appendChild(d)}function section(title,note=""){const s=create("section","detail-section"),h=create("div","section-head");h.append(create("h2","",title),create("div","section-note",note));s.appendChild(h);detail.appendChild(s);return s}function card(c,label,v,note,tone=""){const d=create("div",`decision-card ${tone}`);d.append(create("span","",label),create("strong","",v),create("div","muted",note));c.appendChild(d)}function outcomeTone(r){return r.outcomes.includes("wrong_removal")?"tone-danger":r.outcomes.includes("safe_removal")?"tone-good":r.outcomes.includes("discovered_candidate_fn")?"tone-warn":""}
function summary(r){const c=create("section","decision-flow");card(c,"SUT decision",r.sut_grouping_result,r.has_5a?`Removed ${r.right.doc_id}; kept ${r.left.doc_id}`:"No removal between this pair",r.sut_grouping_result==="DIFFERENT_GROUPS"?"tone-warn":"");card(c,"Judge verdict",r.judge_group_verdict,r.has_5a?`Deletion safety: ${r.judge_removal_verdict}`:`Confidence: ${value(r.confidence)}`);card(c,"Evaluation outcome",r.evaluation_outcome,raw(r),outcomeTone(r));detail.appendChild(c)}
function statuses(c,data){const list=create("div","status-list");for(const[k,v]of Object.entries(data)){const row=create("div","status-row");row.append(create("span","",k.replaceAll("_"," ")),create("span",`pill pill-${v}`,v));list.appendChild(row)}c.appendChild(list)}function endpoint(c,role,e){const d=create("div","sut-endpoint"),f=create("div","facts");d.appendChild(create("h3","",`${role} · doc ${e.doc_id}`));for(const x of[["Action",e.sut_action],["Group size",e.predicted_group_size],["Group ID",e.predicted_group_id===-1?"singleton":e.predicted_group_id],["Cluster key",e.predicted_cluster_key],["Final keeper",e.final_keeper_id]])fact(f,...x);d.appendChild(f);c.appendChild(d)}
function sut(r){const s=section("SUT decision","Observed grouping/action; no hidden SUT reasoning is inferred."),f=create("div","facts");fact(f,"SUT grouping result",r.sut_grouping_result);if(r.has_5a)for(const x of[["SUT action","REMOVE"],["Removed document",r.right.doc_id],["Final keeper",r.left.doc_id],["Predicted cluster key",r.left.predicted_cluster_key],["Predicted group size",r.left.predicted_group_size],["Direct match edge",r.sut_provenance.direct_candidate_edge],["Removal lineage",r.sut_provenance.removal_lineage]])fact(f,...x);s.appendChild(f);if(!r.has_5a){const grid=create("div","sut-grid");endpoint(grid,r.left_role,r.left);endpoint(grid,r.right_role,r.right);s.appendChild(grid)}const d=create("details","glossary");d.appendChild(create("summary","","SUT provenance availability"));statuses(d,r.sut_provenance);s.appendChild(d)}
function judge(r){const s=section("Judge verdict","Group membership and directional replacement are separate decisions."),f=create("div","facts"),items=[["Judge group verdict",r.judge_group_verdict],["Same duplicate group",r.same_duplicate_group],[`${r.left_role} can replace ${r.right_role}`,r.left_can_replace_right],[`${r.right_role} can replace ${r.left_role}`,r.right_can_replace_left],["Relation",r.relation_type],["Material difference",r.material_difference],["Fuzzy scope",r.fuzzy_scope],["Confidence",r.confidence],["Reason codes",r.reason_codes.join(", ")||"None"],["Judge errors",r.error_types.join(", ")||"None"]];if(r.has_5a)items.splice(2,0,["Deletion safety",r.judge_removal_verdict]);for(const x of items)fact(f,...x);s.appendChild(f)}
function safeLink(c,url,text){try{const u=new URL(url);if(["http:","https:"].includes(u.protocol)){const a=create("a","",text||url);a.href=u.href;a.target="_blank";a.rel="noopener noreferrer";c.appendChild(a);return}}catch(_){}c.appendChild(create("span","",url||"—"))}function documentCard(c,role,d){const x=create("div","document");x.append(create("h3","",`${role} · doc ${d.doc_id}`),create("div","metadata",`${d.length_bucket} · ${d.token_count.toLocaleString()} tokens · ${d.language}`));safeLink(x,d.url,d.url);x.appendChild(create("pre","",d.excerpt||"No excerpt available."));c.appendChild(x)}function documents(r){const s=section("Documents","Judge-visible excerpts; not necessarily complete documents."),grid=create("div","detail-grid"),f=create("div","facts");documentCard(grid,r.left_role,r.left);documentCard(grid,r.right_role,r.right);s.appendChild(grid);for(const x of[["Token ratio",r.token_length_ratio.toFixed(3)],["Same hostname",r.same_hostname],[`${r.left_role} group size`,r.left.predicted_group_size],[`${r.right_role} group size`,r.right.predicted_group_size]])fact(f,...x);s.appendChild(f)}
function evidence(r){const s=section("Judge evidence","Judge-selected quotes; not SUT matching evidence or chain-of-thought."),e=r.judge_evidence_status,f=create("div","facts");for(const x of[["Coverage",e.coverage],["Returned by Judge",e.returned],["Retained",e.retained],["Realigned",e.realigned],["Dropped",e.dropped]])fact(f,...x);s.appendChild(f);if(r.evidence.length)for(const item of r.evidence){const q=create("div","quote");q.append(create("strong","",`${item.role}: `),document.createTextNode(item.quote),create("div","metadata",`Judge side ${item.side} · chars ${item.start_char}-${item.end_char}`));s.appendChild(q)}else s.appendChild(create("div","notice muted","No aligned quote spans were retained; the verdict may still be schema-valid."))}
function provenance(r){if(r.has_5b){const s=section("Evaluation candidate discovery","Evaluation retrieval, not original SUT fuzzy scores."),d=r.evaluation_provenance.retrieval,f=create("div","facts");for(const x of[["Retriever",d.sources.join(" + ")||r.retriever_category],["Cosine",number(d.cosine)],["Jaccard",number(d.jaccard)],["Containment",number(d.containment)],["Semantic rank",d.semantic_rank],["Lexical rank",d.lexical_rank],["Selection rule",d.selection_rules.join(", ")]])fact(f,...x);s.appendChild(f)}if(r.has_5a){const s=section("Evaluation removal sampling","Actual removal sample; not necessarily a direct SUT edge."),d=r.evaluation_provenance.removal_sampling,f=create("div","facts");for(const x of[["Selection rule",d.selection_rules.join(", ")],["Frame size",d.frame_size],["Selection probability",d.selection_probability],["Pair seed",d.pair_seed]])fact(f,...x);s.appendChild(f)}}
function groups(r){if(!r.group_context_ids.length)return;const s=section("SUT group context","Bounded deterministic member samples; not an edge graph.");for(const id of r.group_context_ids){const g=GROUPS[id],d=create("details","context-card"),f=create("div","facts");if(!g)continue;d.appendChild(create("summary","",`Group ${g.group_id} · ${g.group_size} docs · ${g.hostname_count} hosts`));for(const x of[["Cluster key",g.cluster_key],["Group size",g.group_size],["Hostname count",g.hostname_count],["Token min/median/max",`${g.token_count_min}/${g.token_count_median}/${g.token_count_max}`],["Top hostnames",g.top_hostnames.map(x=>`${x.value} (${x.count})`).join(", ")],["Languages",g.languages.map(x=>`${x.value} (${x.count})`).join(", ")]])fact(f,...x);d.appendChild(f);const table=create("table","context-table"),head=create("tr");for(const h of["Doc","Action","Keeper","Host","Lang","Tokens","URL"])head.appendChild(create("th","",h));table.appendChild(head);for(const m of g.members){const tr=create("tr");for(const v of[m.doc_id,m.action,m.final_keeper_id,m.hostname,m.language,m.token_count])tr.appendChild(create("td","",value(v)));const link=create("td");safeLink(link,m.url,"open");tr.appendChild(link);table.appendChild(tr)}d.append(table,create("div","metadata",`Showing up to ${g.member_sample_limit} deterministic members.`));s.appendChild(d)}}function risks(r){if(!r.risk_indicators.length)return;const s=section("Group risk signals","Heuristics for triage, not reconstructed SUT reasons."),grid=create("div","risk-grid");for(const x of r.risk_indicators){const d=create("div","risk");d.append(create("span","muted",x.label),create("strong","",x.value),create("div","metadata",x.note));grid.appendChild(d)}s.appendChild(grid)}
function saveReview(id,patch){reviews[id]={...review(id),...patch,updated_at:new Date().toISOString()};if(!reviews[id].review_status&&!reviews[id].human_verdict&&!reviews[id].notes)delete reviews[id];persist();stats(state.filtered);renderRows()}function reviewSection(r){const s=section("Human review","Stored in this browser until exported."),card=create("div","review-card"),grid=create("div","review-grid"),rv=review(r.pair_id),status=create("select"),verdict=create("select");for(const x of[["","Unreviewed"],["AGREE_WITH_JUDGE","Agree"],["DISAGREE_WITH_JUDGE","Disagree"],["UNSURE","Unsure"]]){const o=create("option","",x[1]);o.value=x[0];o.selected=rv.review_status===x[0];status.appendChild(o)}for(const x of[["","Not labeled"],["DUPLICATE","Duplicate"],["NOT_DUPLICATE","Not duplicate"],["UNRESOLVED","Unresolved"]]){const o=create("option","",x[1]);o.value=x[0];o.selected=rv.human_verdict===x[0];verdict.appendChild(o)}status.onchange=()=>saveReview(r.pair_id,{review_status:status.value});verdict.onchange=()=>saveReview(r.pair_id,{human_verdict:verdict.value});const a=create("label"),b=create("label");a.append(create("span","metadata","Agreement with Judge"),status);b.append(create("span","metadata","Human group verdict"),verdict);grid.append(a,b);card.appendChild(grid);const notes=create("textarea");notes.value=rv.notes||"";notes.placeholder="Notes, missing evidence, suspected SUT root cause…";notes.onchange=()=>saveReview(r.pair_id,{notes:notes.value});card.appendChild(notes);const actions=create("div","review-actions"),clear=create("button","","Clear review");clear.onclick=()=>{delete reviews[r.pair_id];persist();selectPair(r.pair_id,false);applyFilters()};actions.append(clear,create("span","muted",rv.updated_at||"Not saved"));card.appendChild(actions);s.appendChild(card)}
function glossary(){const d=create("details","glossary");d.append(create("summary","","How to read this page"),create("p","","SUT decision shows observed grouping/action. Judge verdict shows duplicate-group and directional replacement judgments. Evaluation discovery scores are separate from unavailable SUT fuzzy edges. AVAILABLE, NOT APPLICABLE, and NOT PRESERVED are intentionally distinct."));detail.appendChild(d)}function selectPair(id,hash=true){const r=PAIRS.find(x=>x.pair_id===id);if(!r)return;state.selected=id;detail.replaceChildren();detail.append(create("h2","pair-heading",r.pair_id),create("div","muted",`${r.track} · ${raw(r)}`));summary(r);sut(r);judge(r);documents(r);evidence(r);provenance(r);groups(r);risks(r);reviewSection(r);glossary();document.getElementById("copy-id").disabled=false;if(hash)history.replaceState(null,"",`#pair=${encodeURIComponent(id)}`);renderRows()}function applyFilters(){state.page=0;state.filtered=PAIRS.filter(r=>matches(r)).sort((a,b)=>priority(a)-priority(b)||a.pair_id.localeCompare(b.pair_id));stats(state.filtered);renderReasonChart();renderRows()}
function csvEscape(v){const t=String(v??"");return/[",\\n\\r]/.test(t)?`"${t.replaceAll('"','""')}"`:t}function rowsForExport(){return PAIRS.filter(r=>reviews[r.pair_id]).map(r=>({pair_id:r.pair_id,track:r.track,evaluation_outcome:r.evaluation_outcome,judge_verdict:r.judge_group_verdict,...review(r.pair_id)}))}function download(name,type,text){const u=URL.createObjectURL(new Blob([text],{type})),a=create("a");a.href=u;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),0)}function exportCsv(){const fields=["pair_id","track","evaluation_outcome","judge_verdict","review_status","human_verdict","notes","updated_at"],lines=[fields.join(","),...rowsForExport().map(r=>fields.map(f=>csvEscape(r[f])).join(","))];download(`dedup_reviews_${RUN_ID}.csv`,"text/csv",lines.join("\\n")+"\\n")}function exportJson(){download(`dedup_reviews_${RUN_ID}.json`,"application/json",JSON.stringify({run_id:RUN_ID,dashboard_version:VERSION,reviews:rowsForExport()},null,2))}function parseCsv(text){const lines=text.trim().split(/\\r?\\n/),headers=lines.shift().split(",");return lines.map(line=>{const fields=line.match(/("(?:[^"]|"")*"|[^,]*)(?:,|$)/g).slice(0,-1).map(x=>x.replace(/,$/,"").replace(/^"|"$/g,"").replaceAll('""','"'));return Object.fromEntries(headers.map((h,i)=>[h,fields[i]||""]))})}function importRows(items){let n=0;for(const x of items){if(PAIR_IDS.has(x.pair_id)&&VALID_STATUS.has(x.review_status)&&VALID_VERDICT.has(x.human_verdict||"")){reviews[x.pair_id]={review_status:x.review_status,human_verdict:x.human_verdict||"",notes:x.notes||"",updated_at:x.updated_at||new Date().toISOString()};n++}}persist();applyFilters();document.getElementById("storage-status").textContent=`Imported ${n} reviews.`}async function importFile(file){try{const text=await file.text(),parsed=file.name.endsWith(".json")?JSON.parse(text):parseCsv(text);importRows(Array.isArray(parsed)?parsed:parsed.reviews||[])}catch(e){document.getElementById("storage-status").textContent=`Import failed: ${e.message}`}}
for(const c of Object.values(controls))c.addEventListener(c.id==="search"?"input":"change",applyFilters);document.getElementById("reason-toggle").onclick=()=>{state.reasonExpanded=!state.reasonExpanded;renderReasonChart()};document.getElementById("reset").onclick=()=>{for(const c of Object.values(controls))c.value="";state.reason="";state.reasonExpanded=false;applyFilters()};document.getElementById("previous").onclick=()=>{if(state.page){state.page--;renderRows()}};document.getElementById("next").onclick=()=>{state.page++;renderRows()};document.getElementById("copy-id").onclick=()=>navigator.clipboard.writeText(state.selected);document.getElementById("export-csv").onclick=exportCsv;document.getElementById("export-json").onclick=exportJson;document.getElementById("import-reviews").onclick=()=>document.getElementById("review-file").click();document.getElementById("review-file").onchange=e=>{if(e.target.files[0])importFile(e.target.files[0])};window.onhashchange=()=>{const m=location.hash.match(/^#pair=(.+)$/);if(m)selectPair(decodeURIComponent(m[1]),false)};const params=new URLSearchParams(location.search),track=params.get("track"),language=params.get("language");if(track)controls.track.value=track;if(language&&[...controls.language.options].some(option=>option.value===language))controls.language.value=language;applyFilters();const initial=location.hash.match(/^#pair=(.+)$/);if(initial)selectPair(decodeURIComponent(initial[1]),false);
</script></body></html>"""
    return (
        template.replace("__RUN_ID__", html.escape(evaluation_run_id))
        .replace("__EXPLORER_VERSION__", PAIR_EXPLORER_VERSION)
        .replace("__PAIR_DATA__", _script_json(records))
        .replace("__GROUP_CONTEXT_DATA__", _script_json(group_contexts or {}))
        .replace("__RUN_ID_JSON__", _script_json(evaluation_run_id))
        .replace("__EXPLORER_VERSION_JSON__", _script_json(PAIR_EXPLORER_VERSION))
    )
