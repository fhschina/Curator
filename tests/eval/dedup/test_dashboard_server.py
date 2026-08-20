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
import os
from pathlib import Path

from eval.dedup.dashboard_server import HUMAN_QA_DASHBOARD_RELATIVE, latest_human_qa_dashboard


def _completed_dashboard(runs_root: Path, run_id: str, completed_at: str, body: str) -> Path:
    run_root = runs_root / run_id / "v0_run"
    dashboard = run_root / HUMAN_QA_DASHBOARD_RELATIVE
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text(body)
    marker = run_root / "logs/stages/step_08.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_at_utc": completed_at,
                "artifacts": [
                    {
                        "path": HUMAN_QA_DASHBOARD_RELATIVE.as_posix(),
                        "size_bytes": dashboard.stat().st_size,
                    }
                ],
            }
        )
    )
    return dashboard


def test_latest_human_qa_dashboard_uses_stage_completion_not_file_mtime(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    older = _completed_dashboard(runs_root, "older", "2026-08-20T10:00:00+00:00", "older")
    newer = _completed_dashboard(runs_root, "newer", "2026-08-20T11:00:00+00:00", "newer")
    os.utime(older, (newer.stat().st_mtime + 60, newer.stat().st_mtime + 60))

    assert latest_human_qa_dashboard(runs_root) == newer


def test_latest_human_qa_dashboard_ignores_incomplete_run_and_preserves_fallback(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    incomplete = runs_root / "incomplete/v0_run/reports/human_qa_dashboard.html"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("incomplete")
    fallback = tmp_path / "original.html"
    fallback.write_text("original")

    assert latest_human_qa_dashboard(runs_root, fallback) == fallback
    completed = _completed_dashboard(runs_root, "complete", "2026-08-20T12:00:00+00:00", "latest")

    assert latest_human_qa_dashboard(runs_root, fallback) == completed
    assert fallback.read_text() == "original"
