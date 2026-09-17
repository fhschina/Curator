# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Serve the checksum-bound v0.7 Pair Explorer."""

from __future__ import annotations

import argparse
import hashlib
import json
from http.server import ThreadingHTTPServer
from ipaddress import ip_network
from pathlib import Path
from urllib.parse import urlsplit

from eval.dedup.reporting.http import DashboardHandler
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION

VERSION = JUDGE_CONTRACT_VERSION
EXPLORER = "/dedup-dashboard/pair_explorer.html"
FILES = {"pair_explorer.html": "text/html; charset=utf-8"}
REDIRECTS = dict.fromkeys(
    (
        "/",
        "/dedup-dashboard",
        "/dedup-dashboard/",
        "/dedup-dashboard/comparison",
        "/dedup-dashboard/comparison/",
        "/dedup-dashboard/comparison.html",
        "/dedup-dashboard/RESULTS.md",
        "/dedup-dashboard/RESULTS.html",
        "/dedup-dashboard/RESULTS.raw.md",
    ),
    EXPLORER,
)


def make_server(
    root: Path,
    *,
    port: int,
    networks: list[str],
    hosts: list[str],
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    reports = root.resolve() / "reports"
    record = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    if record.get("version") != VERSION:
        raise ValueError("presentation snapshot is not v0.7")
    artifacts = record["artifacts"]

    class Handler(DashboardHandler):
        allowed_networks = tuple(ip_network(network) for network in networks)
        allowed_hosts = frozenset(value.lower() for value in hosts)

        def _serve_dashboard(self, *, send_body: bool) -> None:
            denial = 403 if not self._client_allowed() else None
            if denial is None and self.headers.get("Host", "").lower() not in self.allowed_hosts:
                denial = 421
            if denial is not None:
                self.send_error(denial)
                return
            request = urlsplit(self.path)
            if request.path in REDIRECTS:
                self.send_response(302)
                self.send_header("Location", REDIRECTS[request.path] + ("?" + request.query if request.query else ""))
                self.end_headers()
                return
            prefix = "/dedup-dashboard/"
            name = request.path[len(prefix) :] if request.path.startswith(prefix) else "FORBIDDEN"
            if not name:
                name = "pair_explorer.html"
            mime = FILES.get(name)
            expected = artifacts.get("reports/" + name)
            if not mime or not expected:
                self.send_error(404)
                return
            path = (reports / name).resolve()
            if not path.is_relative_to(reports):
                self.send_error(403)
                return
            try:
                body = path.read_bytes()
            except OSError:
                self.send_error(503)
                return
            if hashlib.sha256(body).hexdigest() != expected:
                self.send_error(503, "Presentation artifact changed")
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if send_body:
                self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18750)
    parser.add_argument("--allowed-network", action="append", required=True)
    parser.add_argument("--allowed-host", action="append", required=True)
    args = parser.parse_args()
    with make_server(args.root, port=args.port, networks=args.allowed_network, hosts=args.allowed_host) as server:
        print(f"Serving {VERSION} Pair Explorer on port {server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
