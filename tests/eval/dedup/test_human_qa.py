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

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eval.dedup.config import ProfileConfig
from eval.dedup.contracts import stable_record_id
from eval.dedup.report import (
    HUMAN_FIELDS,
    export_human_qa,
    import_human_qa,
    publish_human_qa_report,
)


def test_full_human_qa_freezes_exact_100_50_50_and_imports(tmp_path: Path) -> None:
    qa_seed = 26081204
    pair_ids = [f"pair-{index:03d}" for index in range(200)]
    payloads_path = tmp_path / "payloads.jsonl"
    payloads_path.write_text(
        "".join(
            json.dumps(
                {
                    "canonical_pair_id": pair_id,
                    "judge_payload_hash": f"hash-{pair_id}",
                    "payload": {"document_a": {"text": "a"}, "document_b": {"text": "b"}},
                }
            )
            + "\n"
            for pair_id in pair_ids
        )
    )
    provenance = []
    for index, pair_id in enumerate(pair_ids):
        if index < 100:
            track, retriever = "5a", None
        elif index < 150:
            track, retriever = "5b", "lexical_only"
        else:
            track, retriever = "5b", "semantic_only"
        provenance.append({"canonical_pair_id": pair_id, "track": track, "retriever_bitmask": retriever})
    provenance_path = tmp_path / "provenance.parquet"
    pq.write_table(pa.Table.from_pylist(provenance), provenance_path)
    profile = ProfileConfig(
        name="full",
        anchor_quotas={"singleton": 500, "size_2": 125, "size_3_5": 125, "size_6_20": 125, "size_21_plus": 125},
        removal_pair_budget=10_000,
        cross_group_pair_budget=10_000,
        qa_pair_budget=200,
        minimum_diff_budget=0,
        formal_v0=True,
    )
    packet_path = tmp_path / "packet.jsonl"
    labels_path = tmp_path / "labels.csv"
    counts = export_human_qa(
        profile=profile,
        qa_seed=qa_seed,
        payloads_path=payloads_path,
        provenance_path=provenance_path,
        packet_destination=packet_path,
        labels_destination=labels_path,
    )
    assert counts == {"qa_pairs": 200}
    packet = [json.loads(line) for line in packet_path.read_text().splitlines()]
    assert {row["qa_pair_id"] for row in packet} == {
        stable_record_id("human-qa-v1", qa_seed, pair_id) for pair_id in pair_ids
    }

    with labels_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row.update(
            {
                "same_duplicate_group": "NO",
                "a_can_replace_b": "NO",
                "b_can_replace_a": "NO",
                "relation_type": "UNRELATED",
                "material_difference": "MAJOR",
                "fuzzy_scope": "OUT_OF_SCOPE",
                "reviewer_status": "LABELED",
                "notes": "",
            }
        )
    with labels_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["qa_pair_id", *HUMAN_FIELDS, "reviewer_status", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    imported_labels_path = tmp_path / "human_qa_results.csv"
    imported = import_human_qa(
        packet_path=packet_path,
        labels_path=labels_path,
        destination=imported_labels_path,
    )
    assert imported == {"qa_labels": 200, "ambiguous": 0}

    judge_results_path = tmp_path / "judge_results.jsonl"
    judge_results_path.write_text(
        "".join(
            json.dumps(
                {
                    "judge_payload_hash": row["judge_payload_hash"],
                    "same_duplicate_group": "NO",
                    "a_can_replace_b": "NO",
                    "b_can_replace_a": "NO",
                    "relation_type": "UNRELATED",
                    "material_difference": "MAJOR",
                    "fuzzy_scope": "OUT_OF_SCOPE",
                }
            )
            + "\n"
            for row in packet
        )
    )
    metrics_path = tmp_path / "human_qa_metrics.json"
    report_path = tmp_path / "human_qa_report.md"
    qa_result = publish_human_qa_report(
        packet_path=packet_path,
        labels_path=imported_labels_path,
        judge_results_path=judge_results_path,
        metrics_destination=metrics_path,
        report_destination=report_path,
    )
    assert qa_result["status"] == "complete"
    assert metrics_path.is_file()
    qa_metrics = json.loads(metrics_path.read_text())
    assert qa_metrics["fields"]["same_duplicate_group"]["exact_agreement"] == 1.0
    assert qa_metrics["fields"]["same_duplicate_group"]["cohen_kappa"] is None
    report = report_path.read_text()
    assert "Human QA Double-check" in report
    assert "Human / Judge" in report
