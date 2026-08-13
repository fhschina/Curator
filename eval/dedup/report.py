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
import json
from pathlib import Path
from typing import Any

from eval.dedup.config import ProfileConfig
from eval.dedup.contracts import DuplicateAnswer, FuzzyScope, MaterialDifference, RelationType, stable_record_id
from eval.dedup.validation import HumanQAPending, require, write_text_atomic

HUMAN_FIELDS = (
    "same_duplicate_group",
    "a_can_replace_b",
    "b_can_replace_a",
    "relation_type",
    "material_difference",
    "fuzzy_scope",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


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
    packet_rows = []
    for pair_id in selected:
        payload = payloads[pair_id]
        packet_rows.append(
            {
                "qa_pair_id": stable_record_id("human-qa-v1", qa_seed, pair_id),
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
        fields = ["qa_pair_id", *HUMAN_FIELDS, "reviewer_status", "notes"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in packet_rows:
            writer.writerow({"qa_pair_id": row["qa_pair_id"], "reviewer_status": "PENDING"})
    return {"qa_pairs": len(packet_rows)}


def import_human_qa(*, packet_path: Path, labels_path: Path, destination: Path) -> dict[str, int]:
    packet_ids = {row["qa_pair_id"] for row in _read_jsonl(packet_path)}
    require(packet_ids, "QA_PACKET_EMPTY", "frozen QA packet is empty")
    expected_fields = ["qa_pair_id", *HUMAN_FIELDS, "reviewer_status", "notes"]
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        require(reader.fieldnames == expected_fields, "QA_SCHEMA_MISMATCH", "human label CSV fields or order differ")
        rows = list(reader)
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
        require(
            row["reviewer_status"] in {"LABELED", "AMBIGUOUS"}, "QA_LABEL_INCOMPLETE", "reviewer status is incomplete"
        )
        for field in HUMAN_FIELDS:
            require(row[field] in allowed[field], "QA_LABEL_INVALID", "human label value is invalid", field=field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {"qa_labels": len(rows), "ambiguous": sum(row["reviewer_status"] == "AMBIGUOUS" for row in rows)}


def _human_agreement(labels_path: Path, packet_path: Path, judge_results_path: Path) -> dict[str, Any] | None:
    if not labels_path.exists():
        return None
    with labels_path.open("r", encoding="utf-8", newline="") as file:
        labels = {row["qa_pair_id"]: row for row in csv.DictReader(file)}
    packet = _read_jsonl(packet_path)
    payload_to_qa = {row["judge_payload_hash"]: row["qa_pair_id"] for row in packet}
    judge = _read_jsonl(judge_results_path)
    counts = {field: {"agree": 0, "disagree": 0, "ambiguous": 0} for field in HUMAN_FIELDS}
    for result in judge:
        qa_id = payload_to_qa.get(result["judge_payload_hash"])
        if qa_id is None:
            continue
        human = labels[qa_id]
        for field in HUMAN_FIELDS:
            if human[field] == "AMBIGUOUS":
                counts[field]["ambiguous"] += 1
            elif human[field] == result[field]:
                counts[field]["agree"] += 1
            else:
                counts[field]["disagree"] += 1
    for field_counts in counts.values():
        denominator = field_counts["agree"] + field_counts["disagree"]
        field_counts["agreement_rate"] = field_counts["agree"] / denominator if denominator else None
    return counts


def publish_report(
    *,
    profile: ProfileConfig,
    evaluation_manifest: dict[str, Any],
    metrics: dict[str, Any],
    graph_counts: dict[str, int],
    qa_packet_path: Path,
    qa_labels_path: Path,
    judge_results_path: Path,
    draft_destination: Path,
    final_destination: Path,
) -> dict[str, Any]:
    """Render a bounded-claim report and enforce the formal human-QA gate."""

    if profile.formal_v0:
        require(
            metrics["judge"]["completion_rate"] is not None and metrics["judge"]["completion_rate"] >= 0.99,
            "JUDGE_COMPLETION_BELOW_ACCEPTANCE",
            "formal V0 report requires at least 99% schema-valid judge results",
        )
    agreement = _human_agreement(qa_labels_path, qa_packet_path, judge_results_path)
    title = "NeMo Curator Dedup Evaluation V0"
    banner = "**NON-V0 SMOKE — operational validation only**" if not profile.formal_v0 else "**FORMAL V0**"
    status = "complete" if not profile.formal_v0 or agreement is not None else "human_qa_pending"
    upstream_status = evaluation_manifest["upstream_reproducibility_status"]
    report = f"""# {title}

{banner}

## Executive summary

- Evaluation run: `{evaluation_manifest["evaluation_run_id"]}`
- Profile: `{profile.name}`
- Status: `{status}`
- This run optimizes pipeline completion and auditability rather than score quality.

## Scope and frozen configuration

- Dataset: `{evaluation_manifest["dataset_version"]}` with {evaluation_manifest["dataset_row_count"]:,} documents.
- Upstream SUT: `{evaluation_manifest["sut_run_id"]}`; reproducibility is `{upstream_status}`.
- Judge: `{evaluation_manifest["judge_model"]}` using prompt `{evaluation_manifest["prompt_version"]}`.
- Exact deduplication was an upstream precondition and was not rerun here.

## Track 5a — sampled removal-decision frame

```json
{json.dumps(metrics["track_5a_removal_frame"], indent=2, sort_keys=True)}
```

## Track 5b — sampled candidate pool

```json
{json.dumps(metrics["track_5b_candidate_pool"], indent=2, sort_keys=True)}
```

The Step 5b positive yield is not corpus recall.

## Judge operations

```json
{json.dumps(metrics["judge"], indent=2, sort_keys=True)}
```

## Partial judged constraint graph

```json
{json.dumps(graph_counts, indent=2, sort_keys=True)}
```

## Human QA

{json.dumps(agreement, indent=2, sort_keys=True) if agreement is not None else "Human QA labels are pending."}

## Limitations

- The evaluated handoff is approximately 0.419% of the expected full corpus.
- Cross-group retrieval has shared blind spots and zero inclusion probability for unseen pairs.
- The partial judged graph is not complete ground truth and cannot support corpus cluster metrics.
- Model judgments may be uncertain or order-sensitive; unresolved and failed requests remain accounted for.
- Track 5a and Track 5b are separate sampling frames and are never pooled into a headline confusion matrix.

## Next-version backlog

- Minimum-diff removal challenge slice.
- Additional containment, substring, sparse lexical, SimHash, URL/time, and alternative-parser retrieval channels.
- Judge calibration and graph-risk sampling after the frozen V0 run.
"""
    if profile.formal_v0 and agreement is None:
        write_text_atomic(draft_destination, report)
        raise HumanQAPending(
            "HUMAN_QA_PENDING",
            "formal V0 report remains draft until the frozen QA labels are imported",
            draft_path=str(draft_destination),
        )
    write_text_atomic(final_destination, report)
    return {"status": status, "human_qa_available": agreement is not None}
