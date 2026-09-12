# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fresh identical critic trials using bounded server-error recovery."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from eval.dedup.analysis import critic_ordered_experiment as ordered
from eval.dedup.analysis import critic_paired_trial as paired
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging.paced_relay_v2 import TRANSPORT_CONTRACT, PacedRelayV2
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def prepare(root: Path, predecessor: Path, protocol: Path) -> dict:
    from eval.dedup.judging import paced_relay_v2

    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "RESILIENT_ROOT", "new run root required")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    old = json.loads((predecessor / "manifest.json").read_text())
    assessment = json.loads((predecessor / "assessment.json").read_text())
    completion = json.loads((predecessor / "complete.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    errors = {r["review_id"] for c in assessment["cells"].values() for r in c["pairs"] if r["requires_cause_review"]}
    digest = sha256_file(predecessor / "assessment.json")
    require(
        completion["assessment_sha256"] == digest == review.get("assessment_sha256")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == "IDENTICAL_ORDER_TRIAL_WITH_BOUNDED_HTTP500_RECOVERY",
        "RESILIENT_REVIEW",
        "all actual errors and missing-output causes must be reviewed before a fresh run",
    )
    require(
        old["population"] == 96 and old["logical_requests"] == 354,
        "RESILIENT_POPULATION",
        "same complete frozen order trial required",
    )
    for path in (
        Path(__file__).resolve(),
        Path(paced_relay_v2.__file__).resolve(),
        protocol.resolve(),
        predecessor / "assessment.json",
        predecessor / "review_complete.json",
        predecessor / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    for name in ("panel_private.json", "requests_frozen.json"):
        value = json.loads((predecessor / name).read_text())
        write_json_atomic(root / name, value)
        require(
            sha256_file(root / name) == sha256_file(predecessor / name),
            "RESILIENT_IDENTICAL_INPUT",
            "no input or wire-order change",
        )
    result = {k: v for k, v in old.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version=old["contract_version"] + "-transport-v2",
        purpose="IDENTICAL_FRESH_TRIAL_AFTER_UPSTREAM_HTTP500_WITH_BOUNDED_RECOVERY",
        transport_contract=TRANSPORT_CONTRACT,
        immediate_predecessor=str(predecessor),
        predecessor_next_step=assessment["next_step"],
        predecessor_repeat_disagreements=assessment["candidate_repeat_disagreements"],
        semantic_version_changed=False,
        old_responses_reused=0,
        sources=sources,
        artifacts={
            str(root / name): sha256_file(root / name) for name in ("panel_private.json", "requests_frozen.json")
        },
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    require(not (root / "started.json").exists(), "RESILIENT_TRIAL_STARTED", "never restart observed calls")
    require(
        time.time() < selected.base.DEADLINE - selected.base.STOP_ADMISSION_MARGIN,
        "RESILIENT_TRIAL_DEADLINE",
        "deadline",
    )
    manifest = json.loads((root / "manifest.json").read_text())
    require(
        manifest.get("transport_contract") == TRANSPORT_CONTRACT,
        "RESILIENT_TRANSPORT",
        "explicit frozen v2 transport required",
    )
    rows = selected.base.previous._index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen = [ordered.thaw(r) for r in json.loads((root / "requests_frozen.json").read_text())]
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "RESILIENT_TRIAL_CREDENTIAL", "existing credential unavailable")
    write_json_atomic(
        root / "started.json", {"at_utc": datetime.now(UTC).isoformat(), "deadline_utc_epoch": selected.base.DEADLINE}
    )
    profile = selected.base.previous.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=2 * len(frozen),
    )
    with PacedRelayV2(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=selected.base.previous.GENERATION,
    ) as relay:
        relay.set_context(
            selected.base.previous.RelayContext(manifest["contract_version"], 0, root / "transport_events.jsonl")
        )
        for repeat in range(1, manifest["repeat_count"] + 1):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        selected.collect,
                        r,
                        rows[r["canonical_pair_id"]],
                        endpoint=relay.endpoint,
                        output=root / "responses" / f"{repeat}-{r['arm']}-{r['canonical_pair_id']}.json",
                        specs=manifest["specs"],
                    )
                    for r in frozen
                    if r["repeat"] == repeat
                ]
                for future in as_completed(futures):
                    receipt = future.result()
                    print(json.dumps({k: receipt[k] for k in ("repeat", "arm", "status")}), flush=True)
            selected.base.previous.reference.verify_freeze(root / "manifest.json")
    result = selected.assess(root)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    write_json_atomic(
        root / "complete.json",
        {
            "assessment_sha256": sha256_file(root / "assessment.json"),
            "contract_digest": manifest["contract_digest"],
            "external_attempts": sum(e.get("external_request") is True for e in events),
            "local_rejections": sum(e.get("external_request") is False for e in events),
            "http_statuses": dict(
                Counter(str(e.get("http_status")) for e in events if e.get("external_request") is True)
            ),
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("critic_resilient_execution_v1.md"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.predecessor is not None, "RESILIENT_PREDECESSOR", "reviewed predecessor required")
        result = prepare(args.root, args.predecessor, args.protocol)
    else:
        require(args.env_file is not None, "RESILIENT_ENV", "explicit existing credential source required")
        run(args.root, args.env_file)
        result = {"paired": paired.paired_assessment(args.root), "order": ordered.order_assessment(args.root)}
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts", "specs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
