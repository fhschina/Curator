# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Serve only Exp5 presentation artifacts and a read-only live collection counter."""

from __future__ import annotations

import argparse
import json
import shutil
from http.server import ThreadingHTTPServer
from ipaddress import ip_network
from pathlib import Path
from urllib.parse import urlsplit

from eval.dedup.analysis import exp1_reproduction_recovery as recovery
from eval.dedup.dashboard_server import DashboardHandler

FILES = {
    "/dedup-dashboard/": ("pair_explorer_exp5.html", "text/html; charset=utf-8"),
    "/dedup-dashboard/pair_explorer_exp5.html": ("pair_explorer_exp5.html", "text/html; charset=utf-8"),
    "/dedup-dashboard/comparison.html": ("comparison.html", "text/html; charset=utf-8"),
    "/dedup-dashboard/v05_exp5_pairs.csv": ("v05_exp5_pairs.csv", "text/csv; charset=utf-8"),
    "/dedup-dashboard/comparison.json": ("comparison.json", "application/json"),
    "/dedup-dashboard/RESULTS.md": ("RESULTS.md", "text/markdown; charset=utf-8"),
}
REDIRECTS = {
    "/": "/dedup-dashboard/",
    "/dedup-dashboard": "/dedup-dashboard/",
    "/dedup-dashboard/comparison": "/dedup-dashboard/comparison.html",
    "/dedup-dashboard/comparison/": "/dedup-dashboard/comparison.html",
}


def make_server(
    root: Path,
    *,
    port: int,
    networks: list[str],
    hosts: list[str],
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    root = root.resolve()
    population = recovery.read(root / "manifest.json")["population"]

    class Handler(DashboardHandler):
        allowed_networks = tuple(ip_network(n) for n in networks)
        allowed_hosts = frozenset(h.lower() for h in hosts)

        def _headers(self, size: int, mime: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()

        def _serve_dashboard(self, *, send_body: bool) -> None:  # noqa: PLR0911 - allowlist checks fail closed
            if not self._client_allowed():
                self.send_error(403)
                return
            if self.headers.get("Host", "").lower() not in self.allowed_hosts:
                self.send_error(421)
                return
            path = urlsplit(self.path).path
            if path in REDIRECTS:
                self.send_response(302)
                self.send_header("Location", REDIRECTS[path])
                self.end_headers()
                return
            if path == "/dedup-dashboard/progress.json":
                try:
                    current = recovery.status(root)
                    value = {k: current[k] for k in ("completed", "complete", "running")}
                    value.update(population=population, at_utc=recovery.now())
                except (OSError, ValueError, KeyError):
                    self.send_error(503)
                    return
                body = json.dumps(value).encode()
                self._headers(len(body), "application/json")
                if send_body:
                    self.wfile.write(body)
                return
            if path not in FILES:
                self.send_error(404)
                return
            name, mime = FILES[path]
            presentation = root / "presentation"
            file = (presentation / "current/reports" / name).resolve()
            if not file.is_relative_to(presentation.resolve()):
                self.send_error(403)
                return
            try:
                with file.open("rb") as stream:
                    self._headers(file.stat().st_size, mime)
                    if send_body:
                        shutil.copyfileobj(stream, self.wfile)
            except FileNotFoundError:
                self.send_error(503)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18749)
    parser.add_argument("--allowed-network", action="append", required=True)
    parser.add_argument("--allowed-host", action="append", required=True)
    args = parser.parse_args()
    with make_server(args.root, port=args.port, networks=args.allowed_network, hosts=args.allowed_host) as server:
        print(f"Serving Exp5 presentation on port {args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
