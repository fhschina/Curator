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

"""Human-QA exchange and Step 10 reproducibility report."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from eval.dedup.analysis.metrics import wilson_interval
from eval.dedup.config import JudgeConfig, ProfileConfig
from eval.dedup.contracts import (
    REASON_CODES,
    DuplicateAnswer,
    FuzzyScope,
    MaterialDifference,
    RelationType,
    stable_record_id,
)
from eval.dedup.dashboard import (
    PAIR_EXPLORER_VERSION,
    attach_group_context,
    build_pair_explorer_records,
    pair_explorer_destination,
    pair_explorer_html,
    pair_explorer_review_queue,
)
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)

AUTOMATED_REPORT_VERSION = "dedup-automated-report-v4"
HUMAN_QA_REPORT_VERSION = "dedup-human-qa-report-v1"
RECOMMENDATION_PROMPT_VERSION = "dedup-recommendations-v1"
PAIR_EXPLORER_URL = "http://umb-b200-218.cl1u1.colossus.nvidia.com:18743/dedup-dashboard/"
HUMAN_QA_DASHBOARD_URL = f"{PAIR_EXPLORER_URL}human-qa/"
EVALUATION_README_URL = "https://github.com/fhschina/Curator/blob/dedup-eval/eval/dedup/README.md"
APPENDIX_MARKER = "\n## Appendix "

HUMAN_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
)
HUMAN_REQUIRED_FIELDS = HUMAN_FIELDS[:3]
HUMAN_OPTIONAL_FIELDS = HUMAN_FIELDS[3:]
HUMAN_LABEL_FIELDS = ("qa_pair_id", *HUMAN_FIELDS, "reason_codes", "reviewer_status", "notes")
LEGACY_HUMAN_LABEL_FIELDS = ("qa_pair_id", *HUMAN_FIELDS, "reviewer_status", "notes")
HUMAN_DOCUMENT_FIELDS = tuple(
    f"document_{side}_{field}"
    for side in ("a", "b")
    for field in ("url", "crawl_timestamp", "language", "character_count", "text")
)
HUMAN_EXPORT_FIELDS = (*HUMAN_LABEL_FIELDS, *HUMAN_DOCUMENT_FIELDS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_human_qa_export(
    *,
    selected_pair_ids: list[str],
    payloads: dict[str, dict[str, Any]],
    qa_seed: int,
    id_namespace: str,
    packet_destination: Path,
    labels_destination: Path,
) -> int:
    packet_rows = []
    for pair_id in selected_pair_ids:
        require(
            pair_id in payloads,
            "QA_PAYLOAD_MISSING",
            "selected QA pair has no Judge-visible payload",
            canonical_pair_id=pair_id,
        )
        payload = payloads[pair_id]
        packet_rows.append(
            {
                "qa_pair_id": stable_record_id(id_namespace, qa_seed, pair_id),
                "judge_payload_hash": payload["judge_payload_hash"],
                "visible_payload": payload["payload"],
            }
        )
    write_text_atomic(
        packet_destination,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in packet_rows),
    )
    labels_destination.parent.mkdir(parents=True, exist_ok=True)
    with labels_destination.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HUMAN_LABEL_FIELDS)
        writer.writeheader()
        for row in packet_rows:
            writer.writerow({"qa_pair_id": row["qa_pair_id"], "reviewer_status": "PENDING"})
    return len(packet_rows)


def export_human_qa(
    *,
    profile: ProfileConfig,
    qa_seed: int,
    payloads_path: Path,
    provenance_path: Path,
    packet_destination: Path,
    labels_destination: Path,
) -> dict[str, int]:
    """Freeze a blind QA sample without consulting judge outputs."""

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required for QA export"
        raise RuntimeError(msg) from exc
    payloads = {row["canonical_pair_id"]: row for row in _read_jsonl(payloads_path)}
    provenance = pq.read_table(provenance_path).to_pylist()
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in provenance:
        by_pair.setdefault(row["canonical_pair_id"], []).append(row)
    rng = np.random.default_rng(qa_seed)

    def sample(pool: list[str], size: int) -> list[str]:
        pool = sorted(set(pool))
        if not pool or size == 0:
            return []
        return rng.choice(pool, size=min(size, len(pool)), replace=False).tolist()

    if profile.formal_v0:
        removal_target, lexical_target, semantic_target = 100, 50, 50
    else:
        removal_target = profile.qa_pair_budget // 2
        lexical_target = profile.qa_pair_budget // 4
        semantic_target = profile.qa_pair_budget - removal_target - lexical_target
    removal_pool = [pair_id for pair_id, rows in by_pair.items() if any(row["track"] == "5a" for row in rows)]
    lexical_pool = [
        pair_id
        for pair_id, rows in by_pair.items()
        if any(row["track"] == "5b" and row.get("retriever_bitmask") in {"lexical_only", "both"} for row in rows)
    ]
    semantic_pool = [
        pair_id
        for pair_id, rows in by_pair.items()
        if any(row["track"] == "5b" and row.get("retriever_bitmask") in {"semantic_only", "both"} for row in rows)
    ]
    selected = sample(removal_pool, removal_target)
    if profile.formal_v0:
        require(len(selected) == removal_target, "QA_SAMPLE_INCOMPLETE", "full QA removal quota cannot be filled")
    lexical_selected = sample([item for item in lexical_pool if item not in selected], lexical_target)
    selected += lexical_selected
    if profile.formal_v0:
        require(
            len(lexical_selected) == lexical_target,
            "QA_SAMPLE_INCOMPLETE",
            "full QA lexical-focused quota cannot be filled",
        )
    semantic_selected = sample([item for item in semantic_pool if item not in selected], semantic_target)
    selected += semantic_selected
    if profile.formal_v0:
        require(
            len(semantic_selected) == semantic_target,
            "QA_SAMPLE_INCOMPLETE",
            "full QA semantic-focused quota cannot be filled",
        )
    target_size = profile.qa_pair_budget if profile.formal_v0 else min(profile.qa_pair_budget, len(payloads))
    if len(selected) < target_size and not profile.formal_v0:
        selected += sample([item for item in payloads if item not in selected], target_size - len(selected))
    require(
        len(selected) == target_size,
        "QA_SAMPLE_INCOMPLETE",
        "QA sample could not be filled",
    )
    qa_pairs = _write_human_qa_export(
        selected_pair_ids=selected,
        payloads=payloads,
        qa_seed=qa_seed,
        id_namespace="human-qa-v1",
        packet_destination=packet_destination,
        labels_destination=labels_destination,
    )
    return {"qa_pairs": qa_pairs}


def export_human_qa_diagnostic(
    *,
    profile: ProfileConfig,
    qa_seed: int,
    payloads_path: Path,
    comparisons_path: Path,
    blind_packet_path: Path,
    packet_destination: Path,
    labels_destination: Path,
) -> dict[str, int]:
    """Freeze a reviewer-blind sample enriched for SUT/Judge disagreements."""

    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "numpy and pyarrow are required for diagnostic QA export"
        raise RuntimeError(msg) from exc
    payloads = {row["canonical_pair_id"]: row for row in _read_jsonl(payloads_path)}
    comparisons = pq.read_table(
        comparisons_path,
        columns=[
            "canonical_pair_id",
            "judge_payload_hash",
            "judge_status",
            "has_track_5a",
            "has_track_5b",
            "removal_outcome",
            "cross_group_outcome",
        ],
    ).to_pylist()
    blind_hashes = {row["judge_payload_hash"] for row in _read_jsonl(blind_packet_path)}
    disagreements = [
        row
        for row in comparisons
        if row["judge_status"] == "valid"
        and (row["removal_outcome"] == "wrong_removal" or row["cross_group_outcome"] == "discovered_candidate_fn")
    ]
    eligible = [row for row in disagreements if row["judge_payload_hash"] not in blind_hashes]
    wrong_removal_pool = sorted(
        {
            row["canonical_pair_id"]
            for row in eligible
            if row["has_track_5a"] and row["removal_outcome"] == "wrong_removal"
        }
    )
    wrong_removal_ids = set(wrong_removal_pool)
    cross_group_pool = sorted(
        {
            row["canonical_pair_id"]
            for row in eligible
            if row["has_track_5b"]
            and row["cross_group_outcome"] == "discovered_candidate_fn"
            and row["canonical_pair_id"] not in wrong_removal_ids
        }
    )
    target_size = min(profile.qa_pair_budget, len(wrong_removal_pool) + len(cross_group_pool))
    rng = np.random.default_rng(qa_seed)

    def sample(pool: list[str], size: int) -> list[str]:
        if not pool or size == 0:
            return []
        return rng.choice(pool, size=min(size, len(pool)), replace=False).tolist()

    wrong_removal_target = min(len(wrong_removal_pool), target_size // 2)
    selected_wrong_removals = sample(wrong_removal_pool, wrong_removal_target)
    cross_group_target = min(len(cross_group_pool), target_size - len(selected_wrong_removals))
    selected_cross_group = sample(cross_group_pool, cross_group_target)
    selected = selected_wrong_removals + selected_cross_group
    if len(selected) < target_size:
        selected_set = set(selected)
        remaining = [pair_id for pair_id in [*wrong_removal_pool, *cross_group_pool] if pair_id not in selected_set]
        selected += sample(remaining, target_size - len(selected))
    selected_wrong_removal_count = sum(pair_id in wrong_removal_ids for pair_id in selected)
    qa_pairs = _write_human_qa_export(
        selected_pair_ids=selected,
        payloads=payloads,
        qa_seed=qa_seed,
        id_namespace="human-qa-diagnostic-v1",
        packet_destination=packet_destination,
        labels_destination=labels_destination,
    )
    return {
        "diagnostic_qa_pairs": qa_pairs,
        "diagnostic_wrong_removals": selected_wrong_removal_count,
        "diagnostic_cross_group_duplicates": qa_pairs - selected_wrong_removal_count,
        "diagnostic_disagreements_available": len(disagreements),
        "diagnostic_blind_overlap_excluded": len(disagreements) - len(eligible),
    }


def import_human_qa(*, packet_path: Path, labels_path: Path, destination: Path) -> dict[str, int]:
    packet_ids = {row["qa_pair_id"] for row in _read_jsonl(packet_path)}
    require(packet_ids, "QA_PACKET_EMPTY", "frozen QA packet is empty")
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(
            tuple(reader.fieldnames or ()) in {HUMAN_LABEL_FIELDS, LEGACY_HUMAN_LABEL_FIELDS, HUMAN_EXPORT_FIELDS},
            "QA_SCHEMA_MISMATCH",
            "human label CSV fields or order differ",
        )
        rows = [{field: row.get(field, "") for field in HUMAN_LABEL_FIELDS} for row in reader]
    require(len(rows) == len(packet_ids), "QA_ROW_COUNT_MISMATCH", "human labels must contain one row per QA pair")
    require(
        {row["qa_pair_id"] for row in rows} == packet_ids,
        "QA_ID_MISMATCH",
        "human label IDs differ from frozen packet",
    )
    allowed = {
        "same_duplicate_group": set(DuplicateAnswer) | {"AMBIGUOUS"},
        "a_can_replace_b": set(DuplicateAnswer) | {"AMBIGUOUS"},
        "b_can_replace_a": set(DuplicateAnswer) | {"AMBIGUOUS"},
        "relation_type": set(RelationType) | {"AMBIGUOUS"},
        "material_difference": set(MaterialDifference) | {"AMBIGUOUS"},
        "fuzzy_scope": set(FuzzyScope) | {"AMBIGUOUS"},
    }
    for row in rows:
        row.setdefault("reason_codes", "")
        require(
            row["reviewer_status"] in {"LABELED", "AMBIGUOUS"}, "QA_LABEL_INCOMPLETE", "reviewer status is incomplete"
        )
        for field in HUMAN_REQUIRED_FIELDS:
            require(row[field] in allowed[field], "QA_LABEL_INVALID", "human label value is invalid", field=field)
        for field in HUMAN_OPTIONAL_FIELDS:
            require(
                row[field] == "" or row[field] in allowed[field],
                "QA_LABEL_INVALID",
                "optional human label value is invalid",
                field=field,
            )
        try:
            reasons = json.loads(row["reason_codes"]) if row["reason_codes"] else []
        except json.JSONDecodeError as exc:
            raise DedupEvaluationError(
                "QA_LABEL_INVALID",
                "human reason_codes must be a JSON array",
                field="reason_codes",
            ) from exc
        require(
            isinstance(reasons, list)
            and all(isinstance(reason, str) and reason in REASON_CODES for reason in reasons)
            and len(reasons) == len(set(reasons)),
            "QA_LABEL_INVALID",
            "human reason_codes contains an invalid or duplicate code",
            field="reason_codes",
        )
        row["reason_codes"] = json.dumps(reasons, ensure_ascii=True, separators=(",", ":")) if reasons else ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HUMAN_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"qa_labels": len(rows), "ambiguous": sum(row["reviewer_status"] == "AMBIGUOUS" for row in rows)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_without_appendices(report: str) -> str:
    appendix_start = report.find(APPENDIX_MARKER)
    require(
        appendix_start >= 0,
        "REPORT_APPENDIX_MISSING",
        "published results require a report with an appendix boundary",
    )
    return report[:appendix_start].rstrip() + "\n"


def _replace_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(value.encode("utf-8"))
        file.flush()
        os.fsync(file.fileno())
    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_results_snapshot(*, report: str, destination: Path) -> None:
    _replace_text_atomic(destination, _report_without_appendices(report))


def _qa_packet_inventory(*, run_root: Path, report_destination: Path) -> dict[str, dict[str, Any]]:
    derived_data = report_destination.parent.parent / "data"
    candidates = {
        "blind": [run_root / "data" / "human_qa_packet.jsonl"],
        "diagnostic": [
            run_root / "data" / "human_qa_diagnostic_packet.jsonl",
            derived_data / "human_qa_diagnostic_packet.jsonl",
        ],
    }
    inventory: dict[str, dict[str, Any]] = {}
    for packet_type, paths in candidates.items():
        packet_path = next((path for path in paths if path.is_file()), None)
        if packet_path is None:
            continue
        with packet_path.open(encoding="utf-8") as file:
            rows = sum(1 for line in file if line.strip())
        inventory[packet_type] = {
            "path": str(packet_path),
            "rows": rows,
            "size_bytes": packet_path.stat().st_size,
            "sha256": sha256_file(packet_path),
        }
    return inventory


def _qa_packet_rows(inventory: dict[str, dict[str, Any]], packet_type: str) -> int:
    packet = inventory.get(packet_type)
    return 0 if packet is None else int(packet["rows"])


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _ci_text(low: float | None, high: float | None) -> str:
    return "N/A" if low is None or high is None else f"{_percent(low)}-{_percent(high)}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    top = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(clean(value) for value in row) + " |" for row in rows]
    return "\n".join([top, separator, *body])


def _group_size_bucket(size: int) -> str:
    if size == 1:
        return "singleton"
    if size == 2:
        return "size_2"
    if size <= 5:
        return "size_3_5"
    if size <= 20:
        return "size_6_20"
    return "size_21_plus"


def _ratio_bucket(ratio: float) -> str:
    if ratio < 0.25:
        return "0-0.25"
    if ratio < 0.5:
        return "0.25-0.5"
    if ratio < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def _read_comparisons(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for automated reporting"
        raise RuntimeError(msg) from exc
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["report_group_size_bucket"] = _group_size_bucket(int(row["predicted_group_size_low"]))
        row["report_ratio_bucket"] = _ratio_bucket(float(row["token_length_ratio"]))
    return rows


def _representative_record(
    records: list[dict[str, Any]],
    *,
    used: set[str],
    predicate: Any,
) -> dict[str, Any] | None:
    candidates = [row for row in records if row["pair_id"] not in used and predicate(row)]
    if not candidates:
        return None
    center = median(float(row["token_length_ratio"]) for row in candidates)
    selected = min(
        candidates,
        key=lambda row: (
            abs(float(row["token_length_ratio"]) - center),
            -len(row["evidence"]),
            -float(row["confidence"] or 0.0),
            row["pair_id"],
        ),
    )
    used.add(selected["pair_id"])
    return selected


def _select_report_examples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def wrong(row: dict[str, Any]) -> bool:
        return bool(row["has_5a"] and "wrong_removal" in row["outcomes"])

    def positive(row: dict[str, Any]) -> bool:
        return bool(row["has_5b"] and "discovered_candidate_fn" in row["outcomes"])

    specifications = {
        "removal": [
            ("Large-group wrong removal", lambda row: wrong(row) and row["group_size_bucket"] == "size_21_plus"),
            (
                "Length-mismatch wrong removal",
                lambda row: wrong(row) and 0.5 <= float(row["token_length_ratio"]) < 0.8,
            ),
            ("Same-hostname wrong removal", lambda row: wrong(row) and row["same_hostname"]),
            ("Containment wrong removal", lambda row: wrong(row) and row["relation_type"] == "CONTAINMENT"),
            (
                "Version-related wrong removal",
                lambda row: wrong(row) and row["relation_type"] == "VERSION_RELATED",
            ),
            ("In-scope wrong removal", lambda row: wrong(row) and row["fuzzy_scope"] == "IN_SCOPE"),
            (
                "Safe-removal control",
                lambda row: (
                    row["has_5a"]
                    and "safe_removal" in row["outcomes"]
                    and row["relation_type"] in {"NEAR_SURFACE", "CONTAINMENT", "VERSION_RELATED"}
                ),
            ),
        ],
        "cross": [
            (
                "Lexical-only discovered duplicate",
                lambda row: positive(row) and row["retriever_category"] == "lexical_only",
            ),
            (
                "Semantic-only discovered duplicate",
                lambda row: positive(row) and row["retriever_category"] == "semantic_only",
            ),
            (
                "Retriever-overlap discovered duplicate",
                lambda row: positive(row) and row["retriever_category"] == "both_or_overlap",
            ),
        ],
    }
    used: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {"removal": [], "cross": []}
    for group, specs in specifications.items():
        for label, predicate in specs:
            record = _representative_record(records, used=used, predicate=predicate)
            if record is not None:
                selected[group].append({**record, "example_label": label})
    return selected


def _short_excerpt(value: str, limit: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1].rstrip() + "…"


def _examples_markdown(examples: list[dict[str, Any]], *, dashboard_name: str) -> str:
    if not examples:
        return "No resolved examples were available for this track."
    sections = [
        (
            "Examples are selected deterministically within predeclared slices by median token-length ratio, then "
            "evidence count, Judge confidence, and canonical pair ID. They are automated Judge comparisons, not "
            "human ground truth."
        )
    ]
    for index, record in enumerate(examples, start=1):
        signals = (
            f"group={record['group_size_bucket']}, token ratio={record['token_length_ratio']:.3f}, "
            f"same hostname={record['same_hostname']}"
        )
        if record["retriever_category"]:
            signals += f", retriever={record['retriever_category']}"
        judge = (
            f"relation={record['relation_type']}, material difference={record['material_difference']}, "
            f"fuzzy scope={record['fuzzy_scope']}, confidence={record['confidence']}"
        )
        reasons = ", ".join(record["reason_codes"]) or "None"
        left = html.escape(_short_excerpt(record["left"]["excerpt"]))
        right = html.escape(_short_excerpt(record["right"]["excerpt"]))
        pair_link = f"{dashboard_name}#pair={record['pair_id']}"
        sections.append(
            f"""<details>
<summary>{index}. {html.escape(record["example_label"])} — {record["pair_id"]}</summary>

- Outcome: `{", ".join(record["outcomes"])}`
- Signals: {signals}
- Judge: {judge}
- Reason codes: {reasons}
- [{record["left_role"]} {record["left"]["doc_id"]} vs. {record["right_role"]} {record["right"]["doc_id"]}]({pair_link})

<pre><strong>{record["left_role"]}</strong>\n{left}\n\n<strong>{record["right_role"]}</strong>\n{right}</pre>
</details>"""
        )
    return "\n\n".join(sections)


def _removal_slice(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    selected_rows = [row for row in rows if row["has_track_5a"]]
    result = []
    for value in sorted({str(row.get(field)) for row in selected_rows}):
        selected = [row for row in selected_rows if str(row.get(field)) == value]
        valid = [row for row in selected if row["judge_status"] == "valid"]
        resolved = [row for row in valid if row["removal_outcome"] in {"safe_removal", "wrong_removal"}]
        safe = sum(row["removal_outcome"] == "safe_removal" for row in resolved)
        low, high = wilson_interval(safe, len(resolved))
        result.append(
            {
                "value": value,
                "selected": len(selected),
                "valid": len(valid),
                "resolved": len(resolved),
                "safe": safe,
                "wrong": len(resolved) - safe,
                "precision": safe / len(resolved) if resolved else None,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return result


def _cross_slice(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    selected_rows = [row for row in rows if row["has_track_5b"]]
    result = []
    for value in sorted({str(row.get(field)) for row in selected_rows}):
        selected = [row for row in selected_rows if str(row.get(field)) == value]
        valid = [row for row in selected if row["judge_status"] == "valid"]
        resolved = [row for row in valid if row["cross_group_outcome"] in {"discovered_candidate_fn", "hard_negative"}]
        positives = sum(row["cross_group_outcome"] == "discovered_candidate_fn" for row in resolved)
        low, high = wilson_interval(positives, len(resolved))
        result.append(
            {
                "value": value,
                "selected": len(selected),
                "valid": len(valid),
                "resolved": len(resolved),
                "duplicate_yes": positives,
                "duplicate_no": len(resolved) - positives,
                "yield": positives / len(resolved) if resolved else None,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return result


def _comparison_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    removal = [row for row in rows if row["has_track_5a"]]
    removal_valid = [row for row in removal if row["judge_status"] == "valid"]
    removal_resolved = [row for row in removal_valid if row["removal_outcome"] in {"safe_removal", "wrong_removal"}]
    cross = [row for row in rows if row["has_track_5b"]]
    cross_valid = [row for row in cross if row["judge_status"] == "valid"]
    cross_resolved = [
        row for row in cross_valid if row["cross_group_outcome"] in {"discovered_candidate_fn", "hard_negative"}
    ]
    return {
        "removal_outcome": {
            "selected": len(removal),
            "valid": len(removal_valid),
            "resolved": len(removal_resolved),
            "safe": sum(row["removal_outcome"] == "safe_removal" for row in removal_resolved),
            "wrong": sum(row["removal_outcome"] == "wrong_removal" for row in removal_resolved),
            "unresolved": len(removal_valid) - len(removal_resolved),
            "errors": len(removal) - len(removal_valid),
        },
        "cross_outcome": {
            "selected": len(cross),
            "valid": len(cross_valid),
            "resolved": len(cross_resolved),
            "duplicate_yes": sum(row["cross_group_outcome"] == "discovered_candidate_fn" for row in cross_resolved),
            "duplicate_no": sum(row["cross_group_outcome"] == "hard_negative" for row in cross_resolved),
            "unresolved": len(cross_valid) - len(cross_resolved),
            "errors": len(cross) - len(cross_valid),
        },
        "removal_slices": {
            "predicted group size": _removal_slice(rows, "report_group_size_bucket"),
            "document length": _removal_slice(rows, "length_bucket_low"),
            "token length ratio": _removal_slice(rows, "report_ratio_bucket"),
            "same hostname": _removal_slice(rows, "same_hostname"),
            "relation type": _removal_slice(rows, "relation_type"),
            "material difference": _removal_slice(rows, "material_difference"),
            "fuzzy scope": _removal_slice(rows, "fuzzy_scope"),
        },
        "cross_slices": {
            "retriever source": _cross_slice(rows, "retriever_category"),
            "document length": _cross_slice(rows, "length_bucket_low"),
            "token length ratio": _cross_slice(rows, "report_ratio_bucket"),
            "same hostname": _cross_slice(rows, "same_hostname"),
        },
        "attempt_histogram": dict(sorted(Counter(int(row["judge_attempts"]) for row in rows).items())),
        "confidence_buckets": {
            label: sum(
                row["judge_status"] == "valid"
                and row["confidence"] is not None
                and lower <= float(row["confidence"]) < upper
                for row in rows
            )
            for label, lower, upper in (
                ("0.00-0.50", 0.0, 0.5),
                ("0.50-0.80", 0.5, 0.8),
                ("0.80-0.95", 0.8, 0.95),
                ("0.95-1.00", 0.95, 1.0000001),
            )
        },
    }


def _judge_diagnostics(run_root: Path) -> dict[str, Any]:
    payload_count = 0
    truncated = 0
    window_count = 0
    maximum_tokens = {"A": 0, "B": 0}
    with (run_root / "data" / "judge_payloads.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload_count += 1
            evidence = json.loads(line)["payload"]["long_document_evidence"]
            truncated += bool(evidence["truncated"])
            window_count += len(evidence["windows"])
            for side in ("A", "B"):
                maximum_tokens[side] = max(maximum_tokens[side], int(evidence["token_counts"][side]))
    repair_actions: Counter[str] = Counter()
    with (run_root / "data" / "judge_results.jsonl").open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            for event in json.loads(line).get("deterministic_repair_events", []):
                repair_actions[str(event.get("action", "unknown"))] += 1
    errors = _read_jsonl(run_root / "logs" / "judge_errors.jsonl")
    error_types: Counter[str] = Counter()
    for row in errors:
        for attempt in row.get("errors", []):
            error_types[str(attempt.get("error_type", "unknown"))] += 1
    return {
        "payload_count": payload_count,
        "truncated_payloads": truncated,
        "truncation_rate": truncated / payload_count if payload_count else None,
        "evidence_windows": window_count,
        "maximum_source_tokens": maximum_tokens,
        "repair_actions": dict(sorted(repair_actions.items())),
        "terminal_error_pairs": [row["canonical_pair_id"] for row in errors],
        "terminal_error_attempt_types": dict(sorted(error_types.items())),
    }


def _duration_text(seconds: float) -> str:
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _stage_rows(markers: list[dict[str, Any]]) -> list[list[Any]]:
    result = []
    previous = None
    for marker in markers:
        completed = datetime.fromisoformat(marker["completed_at_utc"])
        duration = "N/A" if previous is None else _duration_text((completed - previous).total_seconds())
        counts = ", ".join(
            f"{key}={value}" for key, value in marker.get("counts", {}).items() if key != "minhash_cache_contract"
        )
        result.append([marker["step"], marker["name"], marker["status"], duration, counts])
        previous = completed
    return result


def _slice_markdown(rows: list[dict[str, Any]], *, removal: bool) -> str:
    if removal:
        return _markdown_table(
            ["Slice", "Selected", "Valid", "Resolved", "Safe", "Wrong", "Precision", "Wilson 95% CI"],
            [
                [
                    row["value"],
                    row["selected"],
                    row["valid"],
                    row["resolved"],
                    row["safe"],
                    row["wrong"],
                    _percent(row["precision"]),
                    _ci_text(row["ci_low"], row["ci_high"]),
                ]
                for row in rows
            ],
        )
    return _markdown_table(
        ["Slice", "Selected", "Valid", "Resolved", "Duplicate YES", "Duplicate NO", "Yield", "Wilson 95% CI"],
        [
            [
                row["value"],
                row["selected"],
                row["valid"],
                row["resolved"],
                row["duplicate_yes"],
                row["duplicate_no"],
                _percent(row["yield"]),
                _ci_text(row["ci_low"], row["ci_high"]),
            ]
            for row in rows
        ],
    )


def _recommendation_schema() -> dict[str, Any]:
    finding = {
        "type": "object",
        "additionalProperties": False,
        "required": ["finding", "evidence_refs"],
        "properties": {
            "finding": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    }
    recommendation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["priority", "action", "rationale", "evidence_refs"],
        "properties": {
            "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "action": {"type": "string"},
            "rationale": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["key_findings", "risks", "recommended_actions"],
        "properties": {
            "key_findings": {"type": "array", "minItems": 1, "maxItems": 3, "items": finding},
            "risks": {"type": "array", "minItems": 1, "maxItems": 3, "items": finding},
            "recommended_actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": recommendation,
            },
        },
    }


def _validate_recommendations(value: Any, allowed_refs: set[str]) -> tuple[dict[str, Any], int]:
    require(isinstance(value, dict), "RECOMMENDATION_SCHEMA_INVALID", "recommendations must be an object")
    require(
        set(value) == {"key_findings", "risks", "recommended_actions"},
        "RECOMMENDATION_SCHEMA_INVALID",
        "recommendation fields differ",
    )
    filtered: dict[str, list[dict[str, Any]]] = {}
    dropped = 0
    for collection in ("key_findings", "risks", "recommended_actions"):
        require(isinstance(value[collection], list), "RECOMMENDATION_SCHEMA_INVALID", "collection must be a list")
        lower, upper = (1, 5) if collection == "recommended_actions" else (1, 3)
        require(
            lower <= len(value[collection]) <= upper,
            "RECOMMENDATION_SCHEMA_INVALID",
            "recommendation collection length is outside the contract",
            collection=collection,
        )
        accepted = []
        for item in value[collection]:
            require(isinstance(item, dict), "RECOMMENDATION_SCHEMA_INVALID", "item must be an object")
            expected = (
                {"priority", "action", "rationale", "evidence_refs"}
                if collection == "recommended_actions"
                else {"finding", "evidence_refs"}
            )
            require(
                set(item) == expected,
                "RECOMMENDATION_SCHEMA_INVALID",
                "recommendation item fields differ",
                collection=collection,
            )
            for field in expected - {"evidence_refs"}:
                require(
                    isinstance(item[field], str) and item[field].strip(),
                    "RECOMMENDATION_SCHEMA_INVALID",
                    "recommendation text must be non-empty",
                    collection=collection,
                    field=field,
                )
            refs = item.get("evidence_refs")
            require(
                isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in allowed_refs for ref in refs),
                "RECOMMENDATION_REFERENCE_INVALID",
                "recommendation contains an unknown evidence reference",
            )
            text = " ".join(str(item[field]) for field in expected - {"evidence_refs"}).lower()
            unsafe_patterns = (
                r"\b(?:boost|improve|increase|low|poor)\w*.{0,40}\brecall\b",
                r"\bmiss(?:es|ing|ed)?\b.{0,80}\b(?:duplicate|match)",
                r"\bretained documents?\b",
                r"\bjudge (?:assigns|precision)\b",
                r"\bzero conflicts\b.{0,80}\bconsisten",
            )
            if any(re.search(pattern, text) for pattern in unsafe_patterns):
                dropped += 1
            else:
                accepted.append(item)
        require(
            accepted,
            "RECOMMENDATION_COLLECTION_EMPTY_AFTER_SCOPE_GUARD",
            "all recommendation items in a required collection exceeded the evaluation scope",
            collection=collection,
        )
        filtered[collection] = accepted
    return filtered, dropped


def _parse_recommendation_json(content: str) -> Any:
    candidate = content.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        require(len(lines) >= 3, "RECOMMENDATION_SCHEMA_INVALID", "invalid fenced JSON response")
        candidate = "\n".join(lines[1:-1]).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    return json.loads(candidate)


def _generate_recommendations(
    judge: JudgeConfig,
    *,
    deterministic_report: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if judge.backend != "nvidia_openai":
        return {
            "schema_version": "dedup-recommendations-v1",
            "status": "skipped",
            "reason": "recommendations require the production NVIDIA OpenAI-compatible backend",
        }
    allowed_refs = set(evidence)
    system_prompt = (
        "You are writing a bounded interpretation and action plan for an audited deduplication evaluation. "
        "Read the supplied deterministic report for context, but use only the allowed evidence map for factual claims. "
        "Do not recompute or alter metrics. "
        "Never claim corpus recall, complete cluster quality, or ground truth. Every factual finding, risk, and "
        "recommendation must cite one or more exact evidence_refs from the allowed evidence map. Return one compact "
        "JSON object only, with 1-3 key findings, 1-3 risks, and 1-5 recommended actions. "
        "Interpret wrong-removal rate only for sampled removed documents, never retained documents. A low Step 5b "
        "candidate-pool yield means a low positive rate or inefficient selected pool; it does not show missed "
        "duplicates, recall, coverage, or completeness. Recommend improving candidate precision/efficiency or creating "
        "a separate recall benchmark, never boosting recall based on this result. Zero conflicts means only that no "
        "conflicts were detected in the partial judged graph. Slice precision is SUT removal precision within a "
        "Judge-defined slice; never call it Judge precision. Cite the exact slice or audit evidence_ref for every "
        "slice-specific number or provenance statement."
    )
    user_payload = json.dumps(
        {
            "report_version": AUTOMATED_REPORT_VERSION,
            "deterministic_report": deterministic_report,
            "allowed_evidence": evidence,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    payload_sha = hashlib.sha256(user_payload.encode("utf-8")).hexdigest()
    errors = []
    finish_reasons = []
    try:
        from openai import OpenAI
    except ImportError as exc:
        return {
            "schema_version": "dedup-recommendations-v1",
            "status": "unavailable",
            "error_type": exc.__class__.__name__,
        }
    credential = os.environ.get(judge.api_key_env, "").strip()
    if not credential:
        return {
            "schema_version": "dedup-recommendations-v1",
            "status": "unavailable",
            "error_type": "MissingCredential",
        }
    client = OpenAI(base_url=judge.base_url, api_key=credential, timeout=judge.timeout_seconds)
    for attempt in range(judge.max_retries + 1):
        try:
            request_system_prompt = system_prompt
            if attempt:
                request_system_prompt += (
                    "\nYour previous response failed local JSON or schema validation. Return a shorter compact JSON "
                    "object satisfying the contract; do not include Markdown or prose outside the object."
                )
            if judge.structured_output_mode == "json_schema":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "dedup_recommendations_v1",
                        "strict": True,
                        "schema": _recommendation_schema(),
                    },
                }
            else:
                response_format = {"type": "json_object"}
                request_system_prompt += "\nThe required JSON Schema is: " + json.dumps(
                    _recommendation_schema(), ensure_ascii=True, separators=(",", ":")
                )
            response = client.chat.completions.create(
                model=judge.model,
                messages=[
                    {"role": "system", "content": request_system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
                top_p=1,
                max_tokens=max(judge.max_output_tokens, 2048),
                response_format=response_format,
                extra_body={"chat_template_kwargs": {"thinking": False}},
                stream=False,
            )
            finish_reasons.append(response.choices[0].finish_reason)
            content = response.choices[0].message.content
            require(isinstance(content, str) and content, "EMPTY_RECOMMENDATION_RESPONSE", "empty response")
            value, dropped = _validate_recommendations(_parse_recommendation_json(content), allowed_refs)
            return {
                "schema_version": "dedup-recommendations-v1",
                "status": "complete",
                "model": judge.model,
                "prompt_version": RECOMMENDATION_PROMPT_VERSION,
                "prompt_sha256": prompt_sha,
                "input_sha256": payload_sha,
                "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "attempts": attempt + 1,
                "scope_guard_dropped_items": dropped,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                **value,
            }
        except Exception as exc:  # noqa: BLE001 - advisory generation must not block factual reporting
            errors.append(exc.issue.code if isinstance(exc, DedupEvaluationError) else exc.__class__.__name__)
    return {
        "schema_version": "dedup-recommendations-v1",
        "status": "unavailable",
        "model": judge.model,
        "prompt_version": RECOMMENDATION_PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "input_sha256": payload_sha,
        "attempts": judge.max_retries + 1,
        "error_types": errors,
        "finish_reasons": finish_reasons,
    }


def _recommendation_markdown(recommendations: dict[str, Any]) -> str:
    if recommendations["status"] != "complete":
        return (
            "AI-generated recommendations were unavailable. This does not affect any automated metric or "
            "the completion status of this report."
        )
    sections = []
    for heading, key, text_key in (
        ("Key findings", "key_findings", "finding"),
        ("Risks", "risks", "finding"),
        ("Recommended actions", "recommended_actions", "action"),
    ):
        sections.append(f"### {heading}")
        items = recommendations[key]
        if not items:
            sections.append("- None returned.")
            continue
        for item in items:
            prefix = f"**{item['priority']} —** " if "priority" in item else ""
            rationale = f" {item['rationale']}" if "rationale" in item else ""
            refs = ", ".join(item["evidence_refs"])
            sections.append(f"- {prefix}{item[text_key]}{rationale} Evidence: {refs}.")
    return "\n\n".join(sections)


def _pipeline_accounting_tree(
    *,
    evaluation_manifest: dict[str, Any],
    markers: list[dict[str, Any]],
    judge: dict[str, Any],
) -> str:
    outcomes = markers[2]["counts"]
    sut = markers[1]["counts"]
    anchors = markers[3]["counts"]
    pairs = markers[4]["counts"]
    return f"""{evaluation_manifest["dataset_row_count"]:,} corpus documents
├── {outcomes["singletons"]:,} singleton documents
└── {sut["grouped_documents"]:,} grouped documents
    ├── {sut["groups"]:,} logical group keepers
    └── {sut["removals"]:,} removals

{pairs["removal_frame_size"]:,} SUT removal decisions
└── {pairs["removal_rows"]:,} uniformly sampled Step 5a keeper-to-removed pairs

{anchors["rows"]:,} Step 4 anchors (Step 5b only)
├── {pairs["cross_lexical_candidates"]:,} relaxed-lexical anchor-candidate records
├── {pairs["cross_semantic_candidates"]:,} semantic anchor-candidate records
└── {pairs["cross_union_candidates"]:,} cross-channel union records
    └── {pairs["cross_unique_selected_pairs"]:,} selected Step 5b cross-group pairs

{judge["requested"]:,} Judge pairs
├── {judge["schema_valid"]:,} schema-valid
│   ├── {judge["resolved"]:,} resolved
│   └── {judge["unresolved"]:,} unresolved
└── {judge["errors"]:,} terminal errors"""


def _step_5b_generation_markdown(
    *,
    evaluation_manifest: dict[str, Any],
    markers: list[dict[str, Any]],
    cross_outcome: dict[str, Any],
    retrieval_config: dict[str, Any],
) -> str:
    anchors = markers[3]["counts"]["rows"]
    pairs = markers[4]["counts"]
    selected_lsh = retrieval_config["selected_lsh"]
    selected_trial = next(
        trial
        for trial in retrieval_config["lexical_trials"]
        if trial["bands"] == selected_lsh["bands"] and trial["rows_per_band"] == selected_lsh["rows_per_band"]
    )
    target = retrieval_config["pilot_candidate_count_target"]
    funnel = _markdown_table(
        ["Step 5b stage", "Count", "Interpretation"],
        [
            ["Step 4 anchors", anchors, "Queries used by Track 5b only"],
            [
                "Relaxed lexical candidates",
                pairs["cross_lexical_candidates"],
                "Cross-group MinHash/LSH results, ranked by exact lexical features",
            ],
            [
                "Semantic candidates",
                pairs["cross_semantic_candidates"],
                f"Cross-group embedding neighbors with top-k={retrieval_config['top_k']}",
            ],
            [
                "Cross-channel union",
                pairs["cross_union_candidates"],
                "Anchor-candidate records after merging lexical and semantic membership",
            ],
            [
                "Selected unique pairs",
                pairs["cross_unique_selected_pairs"],
                "Per-anchor source quotas, deterministic refill, and global pair deduplication",
            ],
            ["Resolved Judge results", cross_outcome["resolved"], "Denominator for candidate-pool positive yield"],
        ],
    )
    missing_resolved_config = not evaluation_manifest.get("upstream_provenance_availability", {}).get(
        "resolved_config", False
    )
    sut_comparison = (
        " The upstream SUT resolved configuration was unavailable, so this report does not claim a numeric "
        "SUT-to-evaluation threshold change such as 0.8 to 0.7."
        if missing_resolved_config
        else ""
    )
    return f"""Track 5b starts from the Step 4 anchors and uses two parallel retrieval channels: a more-permissive
lexical MinHash/LSH configuration and semantic embedding top-k retrieval. The lexical channel was tuned by candidate
volume rather than by changing one scalar similarity threshold: the pilot selected {selected_lsh["bands"]} bands x
{selected_lsh["rows_per_band"]} row per band, producing a median of
{selected_trial["median_cross_group_candidates"]:g} cross-group candidates per pilot anchor against the frozen target
range of {target["minimum"]}-{target["maximum"]}.{sut_comparison}

{funnel}

The selected source counts below are shaped by the frozen per-anchor quotas (up to four lexical-only, four
semantic-only, and two both-channel candidates, followed by deterministic refill). They describe the selected
candidate pool; they are not natural corpus prevalence, channel recall, or an unqualified head-to-head retriever
comparison."""


def _render_deterministic_report(
    *,
    profile: ProfileConfig,
    evaluation_manifest: dict[str, Any],
    metrics: dict[str, Any],
    markers: list[dict[str, Any]],
    analysis: dict[str, Any],
    retrieval_config: dict[str, Any],
    qa_packet_inventory: dict[str, dict[str, Any]],
) -> str:
    removal = metrics["track_5a_removal_frame"]
    cross = metrics["track_5b_candidate_pool"]
    judge = metrics["judge"]
    outcome = analysis["removal_outcome"]
    cross_outcome = analysis["cross_outcome"]
    banner = (
        "**NON-V0 SMOKE — operational validation only**"
        if not profile.formal_v0
        else "**FORMAL V0 — AUTOMATED JUDGE RESULTS**"
    )
    blind_qa_pairs = _qa_packet_rows(qa_packet_inventory, "blind")
    diagnostic_qa_pairs = _qa_packet_rows(qa_packet_inventory, "diagnostic")
    removal_matrix = _markdown_table(
        ["SUT decision", "Judge safe", "Judge wrong", "Unresolved", "Error", "Total"],
        [["REMOVE", outcome["safe"], outcome["wrong"], outcome["unresolved"], outcome["errors"], outcome["selected"]]],
    )
    cross_matrix = _slice_markdown(analysis["cross_slices"]["retriever source"], removal=False)
    pipeline_accounting_tree = _pipeline_accounting_tree(
        evaluation_manifest=evaluation_manifest,
        markers=markers,
        judge=judge,
    )
    step_5b_generation = _step_5b_generation_markdown(
        evaluation_manifest=evaluation_manifest,
        markers=markers,
        cross_outcome=cross_outcome,
        retrieval_config=retrieval_config,
    )
    limitations = [
        "The LLM Judge is the automated reference for these metrics; it is not human ground truth.",
        (
            "Step 5a contains only sampled SUT removals, so removal precision is identifiable but recall, specificity, "
            "and a full SUT confusion matrix are not."
        ),
        (
            "Step 5b is a selected cross-group candidate pool with zero inclusion probability for unseen pairs. "
            "Its positive yield is not corpus recall."
        ),
        (
            "The partial judged constraint graph is not complete ground truth and cannot support corpus-level cluster "
            "precision, recall, or F1."
        ),
        "Track 5a and Track 5b have different sampling frames and are never pooled into one confusion matrix.",
        "The disagreement-enriched human QA diagnostic packet is not a prevalence sample and is never pooled with the blind QA packet.",
        (
            "Upstream provenance is conditionally reproducible because resolved config and several retrieval attestations "
            "were not delivered."
        ),
    ]
    limitations_markdown = "".join(f"- {item}\n" for item in limitations)
    headline = _markdown_table(
        ["Headline metric", "Value", "Numerator / denominator", "Authorized scope"],
        [
            [
                "Judge completion",
                _percent(judge["completion_rate"]),
                f"{judge['schema_valid']:,} / {judge['requested']:,}",
                "All selected pairs",
            ],
            [
                "Removal precision",
                _percent(removal["removal_precision"]),
                f"{removal['safe']:,} / {removal['resolved']:,}",
                "Uniform Step 5a removal frame",
            ],
            [
                "Wrong-removal rate",
                _percent(removal["wrong_removal_rate"]),
                f"{removal['wrong']:,} / {removal['resolved']:,}",
                "Uniform Step 5a removal frame",
            ],
            [
                "Candidate-pool positive yield",
                _percent(cross["positive_yield"]),
                f"{cross['judged_duplicate_yes']:,} / {cross['resolved']:,}",
                "Selected Step 5b candidate pool",
            ],
            [
                "Terminal Judge errors",
                judge["errors"],
                f"{judge['errors']:,} / {judge['requested']:,}",
                "Operational accounting",
            ],
            [
                "Unresolved valid results",
                judge["unresolved"],
                f"{judge['unresolved']:,} / {judge['schema_valid']:,}",
                "Operational accounting",
            ],
        ],
    )
    return f"""# NeMo Curator Dedup Evaluation V0

{banner}

Report version: **{AUTOMATED_REPORT_VERSION}**

## 1. Executive Summary

The automated pipeline completed {judge["requested"]:,} pair evaluations with a schema-valid completion rate of
**{_percent(judge["completion_rate"])}** ({judge["schema_valid"]:,}/{judge["requested"]:,}). The sampled removal
frame produced **{_percent(removal["removal_precision"])} removal precision** with a Wilson 95% confidence interval
of **{_ci_text(*removal["wilson_95_ci"])}**. The complementary wrong-removal rate was
**{_percent(removal["wrong_removal_rate"])}**.

The selected Step 5b cross-group candidate pool produced **{_percent(cross["positive_yield"])} positive yield**
({cross["judged_duplicate_yes"]:,}/{cross["resolved"]:,} resolved candidates). This is candidate-pool yield, not
corpus recall.

{headline}

Human QA is an independent double-check and does not block, replace, calibrate, or rewrite these automated metrics.
The exports include a Judge-independent blind packet of {blind_qa_pairs:,} pairs and a separate
disagreement-enriched diagnostic packet of {diagnostic_qa_pairs:,} pairs. Both expose
only the Judge-visible document payload to reviewers. The diagnostic packet excludes pairs already present in the
blind packet and must not be used to estimate overall prevalence or pooled with the blind packet for unweighted
accuracy metrics.

The [interactive Pair Explorer]({PAIR_EXPLORER_URL}) contains a filterable review queue of wrong removals, discovered
cross-group positives, unresolved/errors, and report control examples. It uses Judge-visible excerpts and evidence;
its labels remain automated Judge results rather than human ground truth.

## 2. Evaluation Scope and Authorized Claims

- Evaluation run: **{evaluation_manifest["evaluation_run_id"]}**
- Dataset: **{evaluation_manifest["dataset_version"]}**, {evaluation_manifest["dataset_row_count"]:,} documents.
- SUT run: **{evaluation_manifest["sut_run_id"]}**
- Judge: **{evaluation_manifest["judge_model"]}**, prompt **{evaluation_manifest["prompt_version"]}**
- Exact deduplication was an upstream precondition and was not rerun.
- Step 5a supports claims about the sampled removal-decision frame.
- Step 5b supports claims about the selected retriever candidate pool only.
- Prohibited claims: {", ".join(metrics["prohibited_claims"])}.

## 3. Pipeline Accounting

<pre>
{pipeline_accounting_tree}
</pre>

Detailed stage-level timing, execution accounting, and Judge operational diagnostics are retained in the full report
appendices.

## 4. Step 5a — Removal Decision Quality

This section reports Track 5a only: the safety of actual SUT keeper-to-removed decisions sampled from the removal
frame. Removal precision is safe resolved removals divided by all resolved sampled removals. The selection was
uniform from a frame of {removal["sampling_frame_size"]:,} removals with inclusion probability
{removal["inclusion_probability"]:.8f}.

{removal_matrix}

There is no SUT-negative sampling frame, so this table is an outcome matrix rather than a full confusion matrix.
Removal recall, specificity, and F1 are not identifiable in V0.

### By predicted group size

{_slice_markdown(analysis["removal_slices"]["predicted group size"], removal=True)}

### By document length

{_slice_markdown(analysis["removal_slices"]["document length"], removal=True)}

### By token length ratio

{_slice_markdown(analysis["removal_slices"]["token length ratio"], removal=True)}

### By hostname relationship

{_slice_markdown(analysis["removal_slices"]["same hostname"], removal=True)}

### By judged relation type

{_slice_markdown(analysis["removal_slices"]["relation type"], removal=True)}

### By material difference

{_slice_markdown(analysis["removal_slices"]["material difference"], removal=True)}

### By fuzzy scope

{_slice_markdown(analysis["removal_slices"]["fuzzy scope"], removal=True)}

### Inspect removal decisions

Use the [Track 5a removal review queue]({PAIR_EXPLORER_URL}?track=5a) to inspect wrong removals and filter by document
length, token-length ratio, hostname relationship, group size, relation type, material difference, or reason code.

## 5. Step 5b — Cross-group Retrieval Analysis

This section reports Track 5b only: missed-duplicate discovery within the selected anchor-based cross-group candidate
pool. It does not estimate corpus recall.

### Candidate generation and selection

{step_5b_generation}

### Positive yield by selected retrieval source

The pool contained {cross_outcome["selected"]:,} selected candidates, of which {cross_outcome["resolved"]:,} were
resolved. Positive yield is Judge duplicate-YES divided by resolved selected candidates.

{cross_matrix}

### By document length

{_slice_markdown(analysis["cross_slices"]["document length"], removal=False)}

### By token length ratio

{_slice_markdown(analysis["cross_slices"]["token length ratio"], removal=False)}

### By hostname relationship

{_slice_markdown(analysis["cross_slices"]["same hostname"], removal=False)}

### Inspect cross-group candidates

Use the [Track 5b cross-group review queue]({PAIR_EXPLORER_URL}?track=5b) to inspect discovered positives and compare
retrieval sources, Judge evidence, document context, and available SUT provenance.

## 6. Methodological Limitations

{limitations_markdown}

## 7. How to Inspect the Results

- Use the [Pair Explorer]({PAIR_EXPLORER_URL}) to inspect automated wrong removals, discovered cross-group positives,
  Judge evidence, provenance availability, and group context.
- Use the [Human QA Dashboard]({HUMAN_QA_DASHBOARD_URL}) to review the blind and diagnostic samples. These dashboards
  require access to the NVIDIA internal network.
- See the [evaluation README]({EVALUATION_README_URL}) for the methodology, runtime profiles, evaluation contract,
  commands, and artifact map.

The report body intentionally excludes document excerpts and pair identifiers. Use the Pair Explorer for pair-level
review.
"""


def _audit_appendices(
    *,
    run_root: Path,
    evaluation_manifest: dict[str, Any],
    metrics: dict[str, Any],
    markers: list[dict[str, Any]],
    analysis: dict[str, Any],
    graph: dict[str, Any],
    diagnostics: dict[str, Any],
    judge_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    sut_manifest: dict[str, Any],
    qa_packet_inventory: dict[str, dict[str, Any]],
) -> str:
    artifacts = [
        [marker["step"], artifact["path"], artifact["size_bytes"], artifact["sha256"]]
        for marker in markers
        for artifact in marker.get("artifacts", [])
    ]
    artifacts.extend(
        ["QA", packet["path"], packet["size_bytes"], packet["sha256"]]
        for packet in qa_packet_inventory.values()
    )
    stage_table = _markdown_table(
        ["Step", "Stage", "Status", "Elapsed since prior stage", "Accounting"],
        _stage_rows(markers),
    )
    error_rows = [[pair_id] for pair_id in diagnostics["terminal_error_pairs"]] or [["None"]]
    judge_table = _markdown_table(
        ["Property", "Value"],
        [
            ["Model", judge_config["model"]],
            ["Structured output", judge_config["structured_output_mode"]],
            ["Completion rate", _percent(metrics["judge"]["completion_rate"])],
            ["Resolution rate among valid", _percent(metrics["judge"]["resolution_rate"])],
            ["Attempt histogram", json.dumps(analysis["attempt_histogram"], sort_keys=True)],
            ["Confidence buckets", json.dumps(analysis["confidence_buckets"], sort_keys=True)],
            [
                "Payload truncation",
                (
                    f"{diagnostics['truncated_payloads']:,}/{diagnostics['payload_count']:,} "
                    f"({_percent(diagnostics['truncation_rate'])})"
                ),
            ],
            ["Evidence windows", f"{diagnostics['evidence_windows']:,}"],
            ["Maximum source tokens", json.dumps(diagnostics["maximum_source_tokens"], sort_keys=True)],
            ["Evidence-only repair actions", json.dumps(diagnostics["repair_actions"], sort_keys=True)],
            [
                "Terminal attempt error types",
                json.dumps(diagnostics["terminal_error_attempt_types"], sort_keys=True),
            ],
        ],
    )
    graph_table = _markdown_table(
        ["Graph metric", "Count"],
        [[key, value] for key, value in sorted(graph.items())],
    )
    audit_table = _markdown_table(
        ["Audit field", "Frozen value"],
        [
            ["Evaluation code revision", evaluation_manifest["evaluation_code_revision"]],
            ["Evaluation source digest", evaluation_manifest["evaluation_source_tree_sha256"]],
            ["Worktree dirty at creation", evaluation_manifest["evaluation_code_worktree_dirty"]],
            ["Upstream reproducibility", evaluation_manifest["upstream_reproducibility_status"]],
            ["SUT repository revision", sut_manifest["repository_revision"]],
            ["Judge contract digest", judge_config["judge_contract_digest"]],
            ["Judge prompt SHA-256", judge_config["prompt_sha256"]],
            ["Tokenizer revision", evaluation_manifest["tokenizer"]["resolved_revision"]],
            ["Embedding SHA-256", evaluation_manifest["embedding_artifact_sha256"]],
            ["MinHash contract digest", retrieval_config["minhash_contract_digest"]],
            ["Selected LSH", json.dumps(retrieval_config["selected_lsh"], sort_keys=True)],
            ["Semantic cosine P90 cutoff", retrieval_config["semantic_cosine_p90"]],
            ["Semantic Jaccard median cutoff", retrieval_config["semantic_jaccard_median"]],
        ],
    )
    missing = ", ".join(
        key for key, available in evaluation_manifest["upstream_provenance_availability"].items() if not available
    )
    artifact_table = _markdown_table(["Step", "Artifact", "Bytes", "SHA-256"], artifacts)
    error_table = _markdown_table(["Canonical pair ID"], error_rows)
    return f"""## Appendix A — Pipeline and Judge Operations

### Stage accounting

{stage_table}

### Judge reliability

{judge_table}

Deterministic repairs only realign or drop evidence offsets against visible text. They do not change duplicate,
replaceability, relation, material-difference, fuzzy-scope, confidence, or reason-code decisions.

Terminal error pair IDs:

{error_table}

## Appendix B — Partial Constraint Graph

{graph_table}

This is a partial judged graph. Conflicting cannot-links are not force-unioned, and the graph is not a complete
corpus reference clustering.

## Appendix C — Reproducibility and Audit

{audit_table}

Missing upstream provenance: {missing}.

## Appendix D — Metric Definitions

- Removal precision = safe resolved Step 5a removals / all resolved Step 5a removals.
- Wrong-removal rate = wrong resolved Step 5a removals / all resolved Step 5a removals.
- Candidate-pool positive yield = Judge duplicate-YES / resolved selected Step 5b candidates.
- Judge completion = schema-valid terminal results / requested pairs.
- Judge resolution = YES-or-NO duplicate decisions / schema-valid results.
- Wilson intervals use a two-sided 95% normal quantile.
- Corpus recall and complete cluster precision/recall/F1 are not defined by this evaluation.

## Appendix E — Artifact Inventory

Run root: {run_root}

{artifact_table}
"""


def _recommendation_evidence(
    metrics: dict[str, Any],
    graph: dict[str, Any],
    analysis: dict[str, Any],
    evaluation_manifest: dict[str, Any],
) -> dict[str, Any]:
    removal = metrics["track_5a_removal_frame"]
    cross = metrics["track_5b_candidate_pool"]
    judge = metrics["judge"]
    upstream = evaluation_manifest.get("upstream_provenance", {})
    evidence = {
        "judge.completion_rate": judge["completion_rate"],
        "judge.errors": judge["errors"],
        "judge.unresolved": judge["unresolved"],
        "track_5a.removal_precision": removal["removal_precision"],
        "track_5a.wrong_removal_rate": removal["wrong_removal_rate"],
        "track_5a.wilson_95_ci": removal["wilson_95_ci"],
        "track_5b.positive_yield": cross["positive_yield"],
        "track_5b.duplicate_yes": cross["judged_duplicate_yes"],
        "graph.must_links": graph["must_links"],
        "graph.cannot_links": graph["cannot_links"],
        "graph.conflicts": graph["conflicts"],
        "audit.upstream_reproducibility": upstream.get("reproducibility_status", "not_recorded"),
        "audit.missing_upstream_provenance": upstream.get("missing", []),
    }
    for slice_name, rows in analysis["cross_slices"].items():
        slice_key = slice_name.replace(" ", "_")
        for row in rows:
            evidence[f"track_5b.slice.{slice_key}.{row['value']}"] = row
    for slice_name, rows in analysis["removal_slices"].items():
        slice_key = slice_name.replace(" ", "_")
        for row in rows:
            evidence[f"track_5a.slice.{slice_key}.{row['value']}"] = row
    return evidence


def publish_report(
    *,
    profile: ProfileConfig,
    run_root: Path,
    final_destination: Path,
    manifest_destination: Path,
    recommendation_judge: JudgeConfig | None = None,
    recommendations_destination: Path | None = None,
    published_results_destination: Path | None = None,
) -> dict[str, Any]:
    """Publish authoritative automated metrics; human QA is an independent diagnostic."""

    require(
        (recommendation_judge is None) == (recommendations_destination is None),
        "RECOMMENDATION_OUTPUT_MISMATCH",
        "recommendation Judge and destination must either both be provided or both be omitted",
    )
    metrics_path = run_root / "reports" / "metrics.json"
    evaluation_manifest_path = run_root / "manifests" / "evaluation_manifest.json"
    metrics = _load_json(metrics_path)
    evaluation_manifest = _load_json(evaluation_manifest_path)
    if profile.formal_v0:
        require(
            metrics["judge"]["completion_rate"] is not None and metrics["judge"]["completion_rate"] >= 0.99,
            "JUDGE_COMPLETION_BELOW_ACCEPTANCE",
            "formal V0 automated report requires at least 99% schema-valid judge results",
        )
    markers = [_load_json(run_root / "logs" / "stages" / f"step_{step:02d}.json") for step in range(1, 10)]
    graph = markers[6]["counts"]
    comparison_rows = _read_comparisons(run_root / "data" / "pair_comparisons.parquet")
    analysis = _comparison_analysis(comparison_rows)
    explorer_records = build_pair_explorer_records(run_root, comparison_rows)
    examples = _select_report_examples(explorer_records)
    dashboard_records = pair_explorer_review_queue(explorer_records, examples)
    group_contexts = attach_group_context(run_root, dashboard_records)
    dashboard_destination = pair_explorer_destination(final_destination)
    write_text_atomic(
        dashboard_destination,
        pair_explorer_html(
            evaluation_run_id=evaluation_manifest["evaluation_run_id"],
            records=dashboard_records,
            group_contexts=group_contexts,
        ),
    )
    diagnostics = _judge_diagnostics(run_root)
    judge_config = _load_json(run_root / "manifests" / "judge_config.json")
    retrieval_config = _load_json(run_root / "manifests" / "retrieval_config.json")
    sut_manifest = _load_json(run_root / "manifests" / "sut_run_manifest.json")
    qa_packet_inventory = _qa_packet_inventory(
        run_root=run_root,
        report_destination=final_destination,
    )
    deterministic_report = _render_deterministic_report(
        profile=profile,
        evaluation_manifest=evaluation_manifest,
        metrics=metrics,
        markers=markers,
        analysis=analysis,
        retrieval_config=retrieval_config,
        qa_packet_inventory=qa_packet_inventory,
    )
    recommendations_status = "not_generated"
    if recommendation_judge is not None and recommendations_destination is not None:
        evidence = _recommendation_evidence(metrics, graph, analysis, evaluation_manifest)
        recommendations = _generate_recommendations(
            recommendation_judge,
            deterministic_report=deterministic_report,
            evidence=evidence,
        )
        write_json_atomic(recommendations_destination, recommendations)
        recommendations_status = recommendations["status"]
    appendices = _audit_appendices(
        run_root=run_root,
        evaluation_manifest=evaluation_manifest,
        metrics=metrics,
        markers=markers,
        analysis=analysis,
        graph=graph,
        diagnostics=diagnostics,
        judge_config=judge_config,
        retrieval_config=retrieval_config,
        sut_manifest=sut_manifest,
        qa_packet_inventory=qa_packet_inventory,
    )
    report = deterministic_report + "\n" + appendices
    write_text_atomic(final_destination, report)
    if published_results_destination is not None:
        _publish_results_snapshot(report=report, destination=published_results_destination)

    outputs = {
        "report_path": str(final_destination),
        "report_sha256": sha256_file(final_destination),
        "pair_explorer_path": str(dashboard_destination),
        "pair_explorer_sha256": sha256_file(dashboard_destination),
        "pair_explorer_pairs": len(dashboard_records),
        "pair_explorer_source_pairs": len(explorer_records),
        "pair_explorer_group_contexts": len(group_contexts),
        "pair_explorer_scope": "wrong removals, cross-group positives, unresolved/errors, and report controls",
        "example_pair_ids": [row["pair_id"] for group in ("removal", "cross") for row in examples[group]],
    }
    if recommendations_destination is not None:
        outputs.update(
            {
                "recommendations_path": str(recommendations_destination),
                "recommendations_sha256": sha256_file(recommendations_destination),
            }
        )
    if published_results_destination is not None:
        outputs.update(
            {
                "published_results_path": str(published_results_destination),
                "published_results_sha256": sha256_file(published_results_destination),
            }
        )
    manifest = {
        "schema_version": "dedup-report-generation-v4",
        "report_version": AUTOMATED_REPORT_VERSION,
        "pair_explorer_version": PAIR_EXPLORER_VERSION,
        "evaluation_run_id": evaluation_manifest["evaluation_run_id"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "human_qa_dependency": "packet metadata only; no human labels",
        "recommendations_status": recommendations_status,
        "renderer_sha256": sha256_file(Path(__file__)),
        "dashboard_renderer_sha256": sha256_file(Path(pair_explorer_html.__code__.co_filename)),
        "inputs": {
            "metrics_sha256": sha256_file(metrics_path),
            "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
            "pair_comparisons_sha256": sha256_file(run_root / "data" / "pair_comparisons.parquet"),
            "judge_results_sha256": sha256_file(run_root / "data" / "judge_results.jsonl"),
            "judge_errors_sha256": sha256_file(run_root / "logs" / "judge_errors.jsonl"),
            "judge_payloads_sha256": sha256_file(run_root / "data" / "judge_payloads.jsonl"),
            "document_outcomes_sha256": sha256_file(run_root / "data" / "document_outcomes.parquet"),
            "pair_provenance_sha256": sha256_file(run_root / "data" / "pair_provenance.parquet"),
            "sut_run_manifest_sha256": sha256_file(run_root / "manifests" / "sut_run_manifest.json"),
            "human_qa_packets": qa_packet_inventory,
        },
        "outputs": outputs,
    }
    write_json_atomic(manifest_destination, manifest)
    return {
        "status": "automated_complete",
        "report_version": AUTOMATED_REPORT_VERSION,
        "human_qa_available": (run_root / "data" / "human_qa_results.csv").is_file(),
        "recommendations_status": recommendations_status,
        "pair_explorer": str(dashboard_destination),
        "published_results": str(published_results_destination) if published_results_destination is not None else None,
    }


def _cohen_kappa(matrix: dict[str, dict[str, int]], labels: list[str]) -> float | None:
    total = sum(matrix[human][judge] for human in labels for judge in labels)
    if total == 0:
        return None
    observed = sum(matrix[label][label] for label in labels) / total
    expected = sum(
        sum(matrix[label][judge] for judge in labels) * sum(matrix[human][label] for human in labels)
        for label in labels
    ) / (total * total)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def _field_agreement(rows: list[tuple[str, str]], labels: list[str]) -> dict[str, Any]:
    matrix = {human: dict.fromkeys(labels, 0) for human in labels}
    for human, judge in rows:
        if human in matrix and judge in matrix[human]:
            matrix[human][judge] += 1
    total = sum(matrix[human][judge] for human in labels for judge in labels)
    agreement = sum(matrix[label][label] for label in labels)
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[human][label] for human in labels if human != label)
        fn = sum(matrix[label][judge] for judge in labels if judge != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        per_class[label] = {
            "support": sum(matrix[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            ),
        }
    f1_values = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    return {
        "labels": labels,
        "confusion_matrix": matrix,
        "evaluated": total,
        "exact_agreement": agreement / total if total else None,
        "cohen_kappa": _cohen_kappa(matrix, labels),
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
    }


def build_human_qa_metrics(
    *,
    packet_path: Path,
    labels_path: Path,
    judge_results_path: Path,
) -> dict[str, Any]:
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        human_labels = {row["qa_pair_id"]: row for row in csv.DictReader(file)}
    packet = _read_jsonl(packet_path)
    payload_to_qa = {row["judge_payload_hash"]: row["qa_pair_id"] for row in packet}
    judge_by_qa = {
        payload_to_qa[row["judge_payload_hash"]]: row
        for row in _read_jsonl(judge_results_path)
        if row["judge_payload_hash"] in payload_to_qa
    }
    field_labels = {
        "same_duplicate_group": list(DuplicateAnswer),
        "a_can_replace_b": list(DuplicateAnswer),
        "b_can_replace_a": list(DuplicateAnswer),
        "relation_type": list(RelationType),
        "material_difference": list(MaterialDifference),
        "fuzzy_scope": list(FuzzyScope),
    }
    fields = {}
    for field in HUMAN_FIELDS:
        pairs = [
            (human[field], judge_by_qa[qa_id][field])
            for qa_id, human in human_labels.items()
            if human[field] not in {"", "AMBIGUOUS"} and qa_id in judge_by_qa
        ]
        fields[field] = _field_agreement(pairs, field_labels[field])
    return {
        "schema_version": HUMAN_QA_REPORT_VERSION,
        "packet_pairs": len(packet),
        "judge_results_available": len(judge_by_qa),
        "ambiguous_reviews": sum(row["reviewer_status"] == "AMBIGUOUS" for row in human_labels.values()),
        "scope": "frozen QA packet only; not corpus-level Judge accuracy",
        "fields": fields,
    }


def publish_human_qa_report(
    *,
    packet_path: Path,
    labels_path: Path,
    judge_results_path: Path,
    metrics_destination: Path,
    report_destination: Path,
) -> dict[str, Any]:
    metrics = build_human_qa_metrics(
        packet_path=packet_path,
        labels_path=labels_path,
        judge_results_path=judge_results_path,
    )
    write_json_atomic(metrics_destination, metrics)
    sections = [
        "# Human QA Double-check",
        "",
        f"Report version: **{HUMAN_QA_REPORT_VERSION}**",
        "",
        (
            "This report is independent from the automated metrics. It does not rewrite or calibrate the canonical "
            "automated report."
        ),
        "",
        _markdown_table(
            ["QA accounting", "Count"],
            [
                ["Frozen packet pairs", metrics["packet_pairs"]],
                ["Judge results available", metrics["judge_results_available"]],
                ["Ambiguous human reviews", metrics["ambiguous_reviews"]],
            ],
        ),
    ]
    for field, result in metrics["fields"].items():
        labels = result["labels"]
        sections.extend(
            [
                "",
                f"## {field}",
                "",
                _markdown_table(
                    ["Metric", "Value"],
                    [
                        ["Evaluated", result["evaluated"]],
                        ["Exact agreement", _percent(result["exact_agreement"])],
                        ["Cohen kappa", result["cohen_kappa"]],
                        ["Macro F1", result["macro_f1"]],
                    ],
                ),
                "",
                _markdown_table(
                    ["Human / Judge", *labels],
                    [[human, *[result["confusion_matrix"][human][judge] for judge in labels]] for human in labels],
                ),
                "",
                _markdown_table(
                    ["Class", "Support", "Precision", "Recall", "F1"],
                    [
                        [
                            label,
                            result["per_class"][label]["support"],
                            _percent(result["per_class"][label]["precision"]),
                            _percent(result["per_class"][label]["recall"]),
                            _percent(result["per_class"][label]["f1"]),
                        ]
                        for label in labels
                    ],
                ),
            ]
        )
    write_text_atomic(report_destination, "\n".join(sections) + "\n")
    return {
        "status": "complete",
        "qa_pairs": metrics["packet_pairs"],
        "report": str(report_destination),
        "metrics": str(metrics_destination),
    }
