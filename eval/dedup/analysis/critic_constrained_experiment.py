# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Freeze schema-constrained transport as a separate critic experimental factor."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.judging import critic_schema_transport as transport
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def prepare(
    root: Path,
    *,
    stage: str,
    parent: Path | None,
    adapter: str,
    config: Path,
    protocol: Path,
    control_adapter: str,
    control_config: Path,
    compatibility: Path,
    reviewed: Path,
    constrained_arms: tuple[str, ...] = ("candidate",),
) -> dict:
    root = root.resolve()
    require(not root.exists(), "CONSTRAINED_ROOT", "new transport contract needs a new run root")
    compatibility = compatibility.resolve()
    reviewed = reviewed.resolve()
    sources = selected.base.previous.reference.verify_freeze(compatibility / "manifest.json")
    capability = json.loads((compatibility / "assessment.json").read_text())
    mode = "without_unique_items"
    require(
        mode in capability["supported_local_valid_modes"],
        "CONSTRAINED_COMPATIBILITY",
        "actual local-valid schema probe required",
    )
    sources.update(selected.base.previous.reference.verify_freeze(reviewed / "manifest.json"))
    review = json.loads((reviewed / "review_complete.json").read_text())
    assessment = json.loads((reviewed / "assessment.json").read_text())
    errors = {
        row["review_id"]
        for cell in assessment["cells"].values()
        for row in cell["pairs"]
        if row["draft_error"] != "CORRECT" or row["status"] == "ENGINEERING_FAILURE"
    }
    require(
        review.get("all_observed_disagreements_reviewed") is True
        and review.get("assessment_sha256") == sha256_file(reviewed / "assessment.json")
        and errors <= set(review.get("cases", {})),
        "CONSTRAINED_REVIEW",
        "all previous disagreements need assessment-bound reviews",
    )
    for path in (
        Path(__file__).resolve(),
        Path(transport.__file__).resolve(),
        protocol.resolve(),
        compatibility / "assessment.json",
        reviewed / "review_complete.json",
        reviewed / "assessment.json",
    ):
        sources[str(path)] = sha256_file(path)
    original = selected.prepare(
        root / "unconstrained_freeze",
        stage=stage,
        parent=parent,
        adapter=adapter,
        config=config,
        protocol=protocol,
        control_adapter=control_adapter,
        control_config=control_config,
    )
    sources.update(selected.base.previous.reference.verify_freeze(root / "unconstrained_freeze/manifest.json"))
    require(
        set(constrained_arms) <= {"candidate", "encoding_control"},
        "CONSTRAINED_ARMS",
        "legacy control format must remain unchanged",
    )
    frozen = json.loads((root / "unconstrained_freeze/requests_frozen.json").read_text())
    for request in frozen:
        if request["arm"] in constrained_arms:
            module = importlib.import_module(original["specs"][request["arm"]]["adapter"])
            request["unconstrained_request_sha256"] = request["request_sha256"]
            request["body"]["response_format"] = transport.response_format(module.response_schema(), mode)
            request["request_sha256"] = sha256_json(request["body"])
    rows = json.loads((root / "unconstrained_freeze/panel_private.json").read_text())
    write_json_atomic(root / "requests_frozen.json", frozen)
    write_json_atomic(root / "panel_private.json", rows)
    manifest = {k: v for k, v in original.items() if k not in ("sources", "artifacts", "contract_digest")}
    manifest.update(
        contract_version=original["contract_version"] + "-schema-transport-v1",
        constrained_arms=list(constrained_arms),
        transport_projection=mode,
        local_validation_unchanged=True,
        sources=sources,
        artifacts={str(p): sha256_file(p) for p in (root / "requests_frozen.json", root / "panel_private.json")},
    )
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "assess"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "panel", "full"), default="pilot")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--adapter", default=selected.DEFAULT_ADAPTER)
    parser.add_argument("--config", type=Path, default=selected.DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--control-adapter", default=selected.DEFAULT_ADAPTER)
    parser.add_argument("--control-config", type=Path, default=selected.DEFAULT_CONFIG)
    parser.add_argument("--compatibility", type=Path)
    parser.add_argument("--reviewed", type=Path)
    parser.add_argument("--constrain-control", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        require(
            args.compatibility is not None and args.reviewed is not None,
            "CONSTRAINED_INPUTS",
            "actual compatibility and reviewed predecessor required",
        )
        result = prepare(
            args.root,
            stage=args.stage,
            parent=args.parent,
            adapter=args.adapter,
            config=args.config,
            protocol=args.protocol,
            control_adapter=args.control_adapter,
            control_config=args.control_config,
            compatibility=args.compatibility,
            reviewed=args.reviewed,
            constrained_arms=("candidate", "encoding_control") if args.constrain_control else ("candidate",),
        )
    elif args.command == "run":
        require(args.env_file is not None, "CONSTRAINED_CREDENTIAL", "explicit existing env required")
        result = selected.run(args.root, args.env_file)
    else:
        result = selected.assess(args.root)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("sources", "artifacts", "response_artifacts", "cells", "specs")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
