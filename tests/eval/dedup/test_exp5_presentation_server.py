# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import threading

import requests

from eval.dedup.analysis import exp5_presentation_server as subject
from eval.dedup.validation import write_json_atomic, write_text_atomic


def test_server_publishes_bound_views_and_counter_but_not_raw_run_files(tmp_path):
    snapshot = tmp_path / "presentation/preview/reports"
    write_text_atomic(snapshot / "comparison.html", "<h1>Version comparison</h1>")
    write_text_atomic(snapshot / "pair_explorer_exp5.html", "<h1>Pair Explorer</h1>")
    write_text_atomic(snapshot / "v05_exp5_pairs.csv", "pair,status\na,valid\n")
    (tmp_path / "presentation/current").symlink_to(snapshot.parent)
    write_json_atomic(tmp_path / "manifest.json", {"population": 20000})
    write_json_atomic(tmp_path / "results/a.json", {"private": "not an HTTP artifact"})
    with subject.make_server(
        tmp_path, port=0, networks=["127.0.0.0/8"], hosts=["localhost"], host="127.0.0.1"
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            headers = {"Host": "localhost"}
            response = requests.get(base + "/dedup-dashboard/comparison.html", headers=headers, timeout=5)
            assert response.status_code == 200
            assert "Version comparison" in response.text
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            head = requests.head(base + "/dedup-dashboard/", headers=headers, timeout=5)
            assert head.status_code == 200
            assert not head.content
            data = requests.get(base + "/dedup-dashboard/progress.json", headers=headers, timeout=5).json()
            assert data["completed"] == 1
            assert data["population"] == 20000
            assert "pid" not in data
            assert "session" not in data
            assert requests.get(base + "/manifest.json", headers=headers, timeout=5).status_code == 404
            assert (
                requests.get(base + "/dedup-dashboard/results/a.json", headers=headers, timeout=5).status_code == 404
            )
            assert (
                requests.get(base + "/dedup-dashboard/", headers={"Host": "evil.example"}, timeout=5).status_code
                == 421
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
