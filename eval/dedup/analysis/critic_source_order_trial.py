# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Same-span presentation trials with frozen semantic controls and resilient transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dedup.analysis import critic_paired_trial as paired
from eval.dedup.analysis import critic_resilient_execution as resilient
from eval.dedup.judging import paced_relay_v2
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

SESSION = paired.selected.base.PRIOR.parent / "critic-five-hour-20260911T070052Z"


def prepare(root: Path) -> dict:
    root = root.resolve()
    require(not root.exists(), "SOURCE_ORDER_ROOT", "new trial root required")
    latest = SESSION / "retention-v4-order96-transport-v2"
    review = json.loads((latest / "review_complete.json").read_text())
    require(
        review.get("next_stage") == "V4_ORIGINAL_SPAN_ORDER_ONLY_SAME96",
        "SOURCE_ORDER_DECISION",
        "latest completed cause review must authorize this exact factor",
    )
    protocol = Path(__file__).with_name("critic_source_order_trial_v1.md")
    old = paired.prepare(
        root / "input_freeze",
        SESSION / "retention-v4-paired96",
        "eval.dedup.judging.critic_retention_v6",
        paired.selected.base.previous.RESOURCES / "v06212_retention_v6.yaml",
        protocol,
    )
    sources = paired.selected.base.previous.reference.verify_freeze(root / "input_freeze/manifest.json")
    for path in (
        Path(__file__).resolve(),
        Path(resilient.__file__).resolve(),
        Path(paced_relay_v2.__file__).resolve(),
    ):
        sources[str(path)] = sha256_file(path)
    for name in ("panel_private.json", "requests_frozen.json"):
        write_json_atomic(root / name, json.loads((root / "input_freeze" / name).read_text()))
    result = {k: v for k, v in old.items() if k not in ("sources", "artifacts", "contract_digest")}
    result.update(
        contract_version="dedup-critic-retention-v6-source-order96",
        purpose="ORIGINAL_VISIBLE_SPAN_ORDER_ONLY_WITH_V4_SEMANTICS",
        transport_contract=paced_relay_v2.TRANSPORT_CONTRACT,
        latest_reviewed_run=str(latest),
        semantic_request_anchor=str(SESSION / "retention-v4-paired96"),
        sources=sources,
        artifacts={
            str(root / name): sha256_file(root / name) for name in ("panel_private.json", "requests_frozen.json")
        },
    )
    require(
        "candidate_schema_property_order" not in result, "SOURCE_ORDER_SCHEMA", "do not carry rejected output ordering"
    )
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root)
    else:
        require(args.env_file is not None, "SOURCE_ORDER_ENV", "explicit existing credential source required")
        resilient.run(args.root, args.env_file)
        result = paired.paired_assessment(args.root)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts", "specs")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
