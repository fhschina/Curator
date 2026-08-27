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

import json
from pathlib import Path

from eval.dedup.human_qa_dashboard import HUMAN_QA_DASHBOARD_VERSION, publish_human_qa_dashboard


def test_dashboard_is_reviewer_blind_and_exports_the_human_qa_contract(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps(
            {
                "qa_pair_id": "qa-pair-1",
                "judge_payload_hash": "secret-judge-hash",
                "visible_payload": {
                    "document_a": {
                        "metadata": {
                            "url": "https://example.com/a",
                            "language": "en",
                            "character_count": 15,
                        },
                        "text": "Document A </script><script>alert(1)</script>",
                    },
                    "document_b": {
                        "metadata": {
                            "url": "https://example.com/b",
                            "language": "en",
                            "character_count": 10,
                        },
                        "text": "Document B",
                    },
                    "long_document_evidence": {"truncated": False},
                },
            }
        )
        + "\n"
    )
    diagnostic = tmp_path / "diagnostic.jsonl"
    diagnostic_row = json.loads(packet.read_text())
    diagnostic_row["qa_pair_id"] = "qa-diagnostic-1"
    diagnostic_row["visible_payload"]["document_a"]["text"] = "Diagnostic document A"
    diagnostic.write_text(json.dumps(diagnostic_row) + "\n")
    destination = tmp_path / "dashboard.html"

    result = publish_human_qa_dashboard(
        packet_paths={"human_qa_blind": packet, "human_qa_diagnostic": diagnostic},
        destination=destination,
        evaluation_run_id="run-1",
    )

    dashboard = destination.read_text()
    assert result == {
        "dashboard_version": HUMAN_QA_DASHBOARD_VERSION,
        "qa_pairs": 2,
        "packets": {"human_qa_blind": 1, "human_qa_diagnostic": 1},
        "destination": str(destination),
    }
    assert "Human QA Review" in dashboard
    assert "Review set" in dashboard
    assert "Blind sample" in dashboard
    assert "Diagnostic set" in dashboard
    assert "Document A" in dashboard
    assert "Same duplicate group?" in dashboard
    assert "Can A replace B?" in dashboard
    assert "The first three decisions are required" in dashboard
    assert "missing=REQUIRED_FIELDS" in dashboard
    assert "reason_codes" in dashboard
    assert "BOILERPLATE" in dashboard
    assert "Export labels CSV" in dashboard
    assert "Export packet JSON" in dashboard
    assert "CSV_FIELDS=LABEL_FIELDS" in dashboard
    assert 'schema_version:"dedup-human-qa-share-v1"' in dashboard
    assert "review.reason_codes=review.reason_codes?JSON.parse(review.reason_codes):[]" in dashboard
    assert "document_a:pair.document_a" in dashboard
    assert "document_b:pair.document_b" in dashboard
    assert "_review_packet.json" in dashboard
    assert "meta.crawl_timestamp" in dashboard
    assert "secret-judge-hash" not in dashboard
    assert "fuzzy-dedup outcome" in dashboard
    assert "</script><script>alert(1)</script>" not in dashboard
    assert "\\u003c/script\\u003e" in dashboard


def test_dashboard_accepts_blind_packet_without_optional_diagnostic(tmp_path: Path) -> None:
    packet = tmp_path / "packet.jsonl"
    packet.write_text(
        json.dumps(
            {
                "qa_pair_id": "qa-pair-1",
                "visible_payload": {
                    "document_a": {"metadata": {}, "text": "Document A"},
                    "document_b": {"metadata": {}, "text": "Document B"},
                    "long_document_evidence": {"truncated": False},
                },
            }
        )
        + "\n"
    )
    destination = tmp_path / "dashboard.html"

    result = publish_human_qa_dashboard(
        packet_paths={"human_qa_blind": packet},
        destination=destination,
        evaluation_run_id="run-1",
    )

    assert result == {
        "dashboard_version": HUMAN_QA_DASHBOARD_VERSION,
        "qa_pairs": 1,
        "packets": {"human_qa_blind": 1},
        "destination": str(destination),
    }
    dashboard = destination.read_text()
    assert "Blind sample" in dashboard
    assert "Diagnostic set" not in dashboard
