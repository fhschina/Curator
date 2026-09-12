# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schema-property-order trials without semantic prompt or local-contract changes."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from eval.dedup.analysis import critic_paired_trial as paired
from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def with_property_order(request: dict, order: list[str]) -> dict:
    result = deepcopy(request)
    body = result["body"]
    schema = body["response_format"]["json_schema"]["schema"]
    require(
        len(order) == len(set(order)) and set(order) == set(schema["properties"]),
        "CRITIC_PROPERTY_ORDER",
        "order must contain each unchanged property exactly once",
    )
    schema["properties"] = {key: schema["properties"][key] for key in order}
    require(sha256_json(body) == request["request_sha256"], "CRITIC_ORDER_SEMANTICS", "only object order may change")
    result["ordered_body_json"] = json.dumps(body, allow_nan=False)
    result["ordered_body_sha256"] = sha256(result["ordered_body_json"].encode()).hexdigest()
    result["schema_property_order"] = list(order)
    return result


def thaw(request: dict) -> dict:
    if "ordered_body_json" not in request:
        return request
    encoded = request["ordered_body_json"]
    require(
        sha256(encoded.encode()).hexdigest() == request["ordered_body_sha256"],
        "CRITIC_ORDER_BINDING",
        "frozen order-sensitive request changed",
    )
    body = json.loads(encoded)
    require(
        body == request["body"]
        and sha256_json(body) == request["request_sha256"]
        and list(body["response_format"]["json_schema"]["schema"]["properties"]) == request["schema_property_order"],
        "CRITIC_ORDER_REPLAY",
        "ordered transport must preserve complete semantic body and property sequence",
    )
    return {**request, "body": body}


def prepare(root: Path, predecessor: Path, protocol: Path) -> dict:
    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "ORDER_TRIAL_ROOT", "new experiment root required")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    old = json.loads((predecessor / "manifest.json").read_text())
    assessment = json.loads((predecessor / "assessment.json").read_text())
    completion = json.loads((predecessor / "complete.json").read_text())
    review = json.loads((predecessor / "review_complete.json").read_text())
    digest = sha256_file(predecessor / "assessment.json")
    errors = {r["review_id"] for c in assessment["cells"].values() for r in c["pairs"] if r["requires_cause_review"]}
    require(
        old["population"] == 96
        and old["repeat_count"] == 2
        and completion["assessment_sha256"] == digest == review.get("assessment_sha256")
        and review.get("all_observed_disagreements_reviewed") is True
        and errors <= set(review.get("cases", {}))
        and review.get("next_stage") == "V4_SCHEMA_PROPERTY_ORDER_ONLY_SAME96",
        "ORDER_TRIAL_REVIEW",
        "complete predecessor review and explicit order-only experiment decision required",
    )
    spec = old["specs"]["encoding_control"]
    require(
        spec["adapter"] == "eval.dedup.judging.critic_retention_v4",
        "ORDER_TRIAL_SEMANTIC_BASE",
        "v4 semantic anchor required; do not combine v5 prompt changes with ordering",
    )
    rows = json.loads((predecessor / "panel_private.json").read_text())
    original = json.loads((predecessor / "requests_frozen.json").read_text())
    immediate = {(r["repeat"], r["canonical_pair_id"]): r for r in original if r["arm"] == "encoding_control"}
    frozen = []
    order = None
    for original_request in original:
        request = original_request
        if request["arm"] == "candidate":
            request = {**deepcopy(immediate[(request["repeat"], request["canonical_pair_id"])]), "arm": "candidate"}
            keys = list(request["body"]["response_format"]["json_schema"]["schema"]["properties"])
            current_order = ["explanation", *(key for key in keys if key != "explanation")]
            require(order is None or order == current_order, "ORDER_TRIAL_UNIFORM", "same schema order on every pair")
            order = current_order
            request = with_property_order(request, order)
        frozen.append(request)
    require(order is not None and len(frozen) == 354, "ORDER_TRIAL_POPULATION", "complete same 96-case trial")
    for path in (
        Path(__file__).resolve(),
        protocol.resolve(),
        predecessor / "assessment.json",
        predecessor / "review_complete.json",
        predecessor / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "requests_frozen.json", frozen)
    result = {k: v for k, v in old.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version="dedup-critic-retention-v4-schema-order-v1",
        stage="order96",
        purpose="EXPLANATION_PROPERTY_FIRST_ONLY_WITH_V4_SEMANTICS",
        specs={"candidate": spec, "encoding_control": spec},
        immediate_predecessor=str(predecessor),
        semantic_anchor="dedup-critic-retention-v4",
        candidate_schema_property_order=order,
        logical_requests=len(frozen),
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "panel_private.json", root / "requests_frozen.json")},
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    require(not (root / "started.json").exists(), "ORDER_TRIAL_STARTED", "never restart observed calls")
    require(
        time.time() < selected.base.DEADLINE - selected.base.STOP_ADMISSION_MARGIN, "ORDER_TRIAL_DEADLINE", "deadline"
    )
    manifest = json.loads((root / "manifest.json").read_text())
    rows = selected.base.previous._index(json.loads((root / "panel_private.json").read_text()), "panel")
    frozen = [thaw(r) for r in json.loads((root / "requests_frozen.json").read_text())]
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "ORDER_TRIAL_CREDENTIAL", "existing credential unavailable")
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
    with selected.base.previous.PacedRelay(
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


def order_assessment(root: Path) -> dict:
    selected.base.previous.reference.verify_freeze(root / "manifest.json")
    manifest = json.loads((root / "manifest.json").read_text())
    requests = json.loads((root / "requests_frozen.json").read_text())
    expected = manifest["candidate_schema_property_order"]
    valid, matching, missing = 0, 0, []
    for request in requests:
        thaw(request)
        if request["arm"] != "candidate":
            continue
        path = root / "responses" / f"{request['repeat']}-candidate-{request['canonical_pair_id']}.json"
        if not path.exists():
            missing.append(request["canonical_pair_id"])
            continue
        receipt = json.loads(path.read_text())
        if receipt["status"] == "VALID":
            valid += 1
            matching += list(json.loads(receipt["assistant_content"])) == expected
    result = {
        "assessment_sha256": sha256_file(root / "assessment.json"),
        "expected_candidate_order": expected,
        "valid_candidate_outputs": valid,
        "outputs_following_requested_order": matching,
        "missing_candidate_outputs": missing,
        "order_intervention_empirically_applied": valid > 0 and valid == matching,
        "semantic_quality_is_separate": True,
        "release_eligible": False,
    }
    write_json_atomic(root / "order_assessment.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("critic_ordered_experiment_v1.md"))
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.predecessor is not None, "ORDER_TRIAL_PREDECESSOR", "reviewed predecessor required")
        result = prepare(args.root, args.predecessor, args.protocol)
    else:
        if args.command == "run":
            require(args.env_file is not None, "ORDER_TRIAL_ENV", "explicit existing credential source required")
            run(args.root, args.env_file)
        else:
            selected.assess(args.root)
        result = {"paired": paired.paired_assessment(args.root), "order": order_assessment(args.root)}
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts", "specs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
