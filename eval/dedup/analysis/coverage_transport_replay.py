# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unscored transport-only diagnosis of a stopped, immutable coverage experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from eval.dedup.analysis.coverage_diagnostic import CONFIG, bind_coverage
from eval.dedup.analysis.critic_diagnostic import _jsonl, validate_freeze
from eval.dedup.analysis.development_diagnostic import _index
from eval.dedup.judging.coverage_runtime import coverage_renderer
from eval.dedup.judging.coverage_witness import COLUMN
from eval.dedup.judging.local_ndd import _safe_retry_feedback
from eval.dedup.judging.payload_transport import bind_transported_payload
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def bind_coverage_transport(inputs: list[dict], outputs: list[dict], mains: dict) -> list[dict]:
    packets, raw = _index(inputs, "transport inputs"), _index(outputs, "transport outputs")
    require(packets.keys() == raw.keys(), "COVERAGE_MEMBERSHIP", "every input requires exactly one raw output")
    predictions = []
    for pid, row in raw.items():
        bound, audit = bind_transported_payload(packets[pid], row)
        result = bind_coverage([packets[pid]], [bound], mains)[0]
        predictions.append({**result, "payload_transport": audit, "offline_transport_diagnostic_only": True})
    return predictions


def replay_transport(root: Path, output: Path) -> dict:
    manifest = validate_freeze(root)
    require((root / "stopped.json").is_file(), "TRANSPORT_RUN_ACTIVE", "only replay a stopped frozen experiment")
    mains = json.loads((root / "fixed_main.json").read_text())
    folder = root / "preflight_coverage"
    complete = json.loads((folder / "complete.json").read_text())
    require(
        all(sha256_file(p) == digest for p, digest in complete["artifacts"].items()),
        "TRANSPORT_SOURCE_CHANGED",
        "original response or error artifacts changed",
    )
    render = coverage_renderer(CONFIG)
    rows = []
    for attempt in sorted(folder.glob("attempt_*")):
        packets = _index(_jsonl(attempt / "input.jsonl"), "attempt inputs")
        raw = _index([r for p in sorted((attempt / "output").glob("*.jsonl")) for r in _jsonl(p)], "attempt outputs")
        require(packets.keys() == raw.keys(), "TRANSPORT_MEMBERSHIP", "do not exclude missing raw responses")
        for pid, packet in packets.items():
            row = raw[pid]
            entry = {
                "attempt": attempt.name,
                "canonical_pair_id": pid,
                "raw_row_sha256": sha256_json(row),
                "response_contract_valid_after_transport_binding": False,
            }
            try:
                _, audit = bind_transported_payload(packet, row)
                require(
                    render({**packet, "repair_feedback": None}) == render({**row, "repair_feedback": None}),
                    "TRANSPORT_PROMPT_CHANGED",
                    "model-visible payload differs",
                )
                observed_messages = []
                for message in row[COLUMN + "__trace"][:2]:
                    content = message["content"]
                    if isinstance(content, list):
                        content = "".join(block["text"] for block in content)
                    observed_messages.append({"role": message["role"], "content": content})
                entry.update(
                    payload_transport=audit,
                    payload_only_render_identical=True,
                    native_trace_matches_echo_render=observed_messages == render(row),
                    native_trace_matches_original_render=observed_messages == render(packet),
                    native_request_messages_sha256=sha256_json(observed_messages),
                    original_and_echo_render_identical=render(packet) == render(row),
                    repair_feedback_representation_changed=packet.get("repair_feedback") != row.get("repair_feedback"),
                )
                predictions = bind_coverage_transport([packet], [row], mains)
                entry.update(response_contract_valid_after_transport_binding=True, prediction=predictions[0])
            except Exception as exc:  # noqa: BLE001 - preserve every remaining technical failure without relabeling
                entry["issue"] = _safe_retry_feedback(exc)
            rows.append(entry)
    require(len(rows) > 0, "TRANSPORT_RESPONSES_MISSING", "stopped experiment has no raw responses")
    result = {
        "schema_version": "dedup-coverage-transport-replay-v2",
        "source_contract_digest": manifest["diagnostic_contract_digest"],
        "diagnostic_only": True,
        "external_model_calls": 0,
        "source_completion_changed": False,
        "reference_read_or_changed": False,
        "eligible_for_release": False,
        "response_count": len(rows),
        "unique_pair_count": len({r["canonical_pair_id"] for r in rows}),
        "transport_verified_responses": sum("payload_transport" in r for r in rows),
        "repair_feedback_representation_changed": sum(
            r.get("repair_feedback_representation_changed", False) for r in rows
        ),
        "native_request_differs_from_original_render": sum(
            not r.get("native_trace_matches_original_render", False) for r in rows
        ),
        "response_contract_valid_after_transport_binding": sum(
            r["response_contract_valid_after_transport_binding"] for r in rows
        ),
        "remaining_issues": dict(Counter(r["issue"]["code"] for r in rows if "issue" in r)),
        "implementation_artifacts": {
            str(p.resolve()): sha256_file(p)
            for p in (Path(__file__), Path(__file__).parent.parent / "judging/payload_transport.py")
        },
        "source_complete_sha256": sha256_file(folder / "complete.json"),
        "rows": rows,
    }
    write_json_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay_transport(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "implementation_artifacts"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
