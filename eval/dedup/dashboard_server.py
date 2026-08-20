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

import argparse
import json
import shutil
from collections.abc import Sequence
from datetime import datetime
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Network, IPv6Network, ip_address, ip_network
from pathlib import Path
from urllib.parse import urlsplit

HUMAN_QA_DASHBOARD_RELATIVE = Path("reports/human_qa_dashboard.html")
HUMAN_QA_STAGE_MARKER_GLOB = "*/*/logs/stages/step_08.json"
PAIR_EXPLORER_PATHS = {"/", "/index.html", "/dedup-dashboard", "/dedup-dashboard/"}
HUMAN_QA_PATHS = {
    "/dedup-dashboard/human-qa",
    "/dedup-dashboard/human-qa/",
    "/dedup-dashboard/human-qa/index.html",
}
ALLOWED_PATHS = PAIR_EXPLORER_PATHS | HUMAN_QA_PATHS


def latest_pair_explorer(reports_root: Path) -> Path:
    candidates = list(reports_root.glob("pair_explorer*.html"))
    if not candidates:
        raise FileNotFoundError("no Pair Explorer dashboard is available")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _completed_human_qa_candidate(marker_path: Path) -> tuple[float, int, str, Path] | None:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        completed_at = datetime.fromisoformat(marker["completed_at_utc"])
        artifacts = marker["artifacts"]
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if marker.get("status") != "complete" or completed_at.tzinfo is None or not isinstance(artifacts, list):
        return None

    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("path") == HUMAN_QA_DASHBOARD_RELATIVE.as_posix()
        ),
        None,
    )
    if artifact is None:
        return None

    run_root = marker_path.parents[2]
    dashboard = run_root / HUMAN_QA_DASHBOARD_RELATIVE
    try:
        stat = dashboard.stat()
    except FileNotFoundError:
        return None
    if not dashboard.is_file() or artifact.get("size_bytes") != stat.st_size:
        return None
    return (completed_at.timestamp(), stat.st_mtime_ns, str(dashboard), dashboard)


def latest_human_qa_dashboard(runs_root: Path, fallback: Path | None = None) -> Path:
    candidates = [
        candidate
        for marker_path in runs_root.glob(HUMAN_QA_STAGE_MARKER_GLOB)
        if (candidate := _completed_human_qa_candidate(marker_path)) is not None
    ]
    if candidates:
        return max(candidates)[-1]
    if fallback is not None and fallback.is_file():
        return fallback
    raise FileNotFoundError("no completed Human QA dashboard is available")


def dashboard_for_path(
    request_path: str,
    *,
    pair_explorer_reports_root: Path,
    runs_root: Path,
    human_qa_fallback: Path | None,
) -> Path:
    if request_path in HUMAN_QA_PATHS:
        return latest_human_qa_dashboard(runs_root, human_qa_fallback)
    return latest_pair_explorer(pair_explorer_reports_root)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DedupDashboard/1.1"
    sys_version = ""
    pair_explorer_reports_root: Path
    runs_root: Path
    human_qa_fallback: Path | None
    allowed_networks: tuple[IPv4Network | IPv6Network, ...]
    allowed_hosts: frozenset[str]

    def _client_allowed(self) -> bool:
        client = ip_address(self.client_address[0])
        return any(client in network for network in self.allowed_networks)

    def _serve_dashboard(self, *, send_body: bool) -> None:
        if not self._client_allowed():
            self.send_error(403)
            return
        if self.headers.get("Host", "").lower() not in self.allowed_hosts:
            self.send_error(421)
            return
        request_path = urlsplit(self.path).path
        if request_path not in ALLOWED_PATHS:
            self.send_error(404)
            return
        try:
            dashboard = dashboard_for_path(
                request_path,
                pair_explorer_reports_root=self.pair_explorer_reports_root,
                runs_root=self.runs_root,
                human_qa_fallback=self.human_qa_fallback,
            )
            stat = dashboard.stat()
        except FileNotFoundError:
            self.send_error(503)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Last-Modified", formatdate(stat.st_mtime, usegmt=True))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if send_body:
            with dashboard.open("rb") as file:
                shutil.copyfileobj(file, self.wfile, length=1024 * 1024)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        self._serve_dashboard(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._serve_dashboard(send_body=False)

    def log_message(self, message: str, *args: object) -> None:
        if len(args) > 1 and str(args[1]).startswith(("4", "5")):
            super().log_message(message, *args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve immutable dedup dashboards through stable URLs.")
    parser.add_argument("--pair-explorer-reports-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--human-qa-fallback", type=Path)
    parser.add_argument("--port", default=18743, type=int)
    parser.add_argument("--allowed-network", action="append", required=True)
    parser.add_argument("--allowed-host", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    DashboardHandler.pair_explorer_reports_root = args.pair_explorer_reports_root
    DashboardHandler.runs_root = args.runs_root
    DashboardHandler.human_qa_fallback = args.human_qa_fallback
    DashboardHandler.allowed_networks = tuple(ip_network(value) for value in args.allowed_network)
    DashboardHandler.allowed_hosts = frozenset(value.lower() for value in args.allowed_host)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), DashboardHandler)  # noqa: S104
    server.daemon_threads = True
    human_qa = latest_human_qa_dashboard(args.runs_root, args.human_qa_fallback)
    print(
        f"Serving {latest_pair_explorer(args.pair_explorer_reports_root).name} and {human_qa.name} "
        f"on port {args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
