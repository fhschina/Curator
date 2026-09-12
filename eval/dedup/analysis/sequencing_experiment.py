# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired output sequencing with unchanged typed meaning and evidence contracts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from eval.dedup.analysis.coverage_experiment import REPO, common
from eval.dedup.analysis.policy_experiment import PolicyArm
from eval.dedup.analysis.proposition_experiment import PropositionExperiment
from eval.dedup.analysis.selection_scope_replay import POLICY
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT
from eval.dedup.judging.policy_runtime import policy_description, policy_renderer
from eval.dedup.judging.sequencing_runtime import (
    CONTROL,
    SequencingRuntime,
    load_sequencing_config,
    sequencing_renderer,
    sequencing_schema,
)
from eval.dedup.judging.typed_coverage import CONTRACT, typed_coverage_schema
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class SequencingArm(PolicyArm):
    @cached_property
    def renderers(self) -> dict:
        return {"coverage": sequencing_renderer(self.config["coverage"])}

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "SEQUENCING_ARM", "single typed arm required")
        return SequencingRuntime(
            self.config["coverage"],
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )


class SequencingExperiment(PropositionExperiment):
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path.resolve()
        self.spec = json.loads(self.spec_path.read_text())
        require(
            self.spec["control_routing"] == self.spec["coverage_routing"] == ROUTING_CONTRACT
            and self.spec["control_contract"] == self.spec["coverage_contract"] == CONTRACT
            and self.spec["scope_postprocessor"] == POLICY
            and self.spec["max_parallel_requests"] == 16
            and self.spec["public_pairs_per_cell"] == 258
            and self.spec["technical_pairs_per_arm"] == 8,
            "SEQUENCING_EXPERIMENT",
            "fixed semantic contracts, population, routing and budget required",
        )
        self.config = {v: (REPO / self.spec[v + "_config"]).resolve() for v in ("control", "coverage")}
        require(
            self.config["control"] == CONTROL,
            "SEQUENCING_CONTROL",
            "preserve prior complete-proposition typed control",
        )
        load_sequencing_config(self.config["coverage"])
        self.arms = {
            "control": PolicyArm(CONTROL, version=self.spec["version"], evidence_first=True),
            "coverage": SequencingArm(self.config["coverage"], version=self.spec["version"], evidence_first=True),
        }

    def presentation_audit(self, inputs: list[dict], mains: dict) -> dict:
        baseline = policy_renderer(CONTROL)
        policy = policy_description("complete_propositions")
        called, _ = self.layout(inputs, mains, "coverage")
        rows = []
        for packet in called:
            messages = {v: render(packet) for v, render in self.renderers.items()}
            require(
                messages["control"] == baseline(packet),
                "SEQUENCING_CONTROL_CHANGED",
                "prior control messages required",
            )
            require(
                all(m[0]["content"].count(policy["block"]) == 1 for m in messages.values()),
                "SEQUENCING_POLICY_CHANGED",
                "same exact semantic policy block required",
            )
            packets = {}
            for v, m in messages.items():
                text = m[1]["content"]
                require(
                    text.count("<semantic_diff ") == text.count("</semantic_diff>") == 1,
                    "SEQUENCING_PACKET",
                    "one intact visible packet required",
                )
                packets[v] = text.split("<semantic_diff ", 1)[1].split("</semantic_diff>", 1)[0]
            require(
                packets["control"] == packets["coverage"], "SEQUENCING_PACKET_CHANGED", "visible text cannot change"
            )
            rows.append(
                {
                    "canonical_pair_id": packet["canonical_pair_id"],
                    "message_sha256": {v: sha256_json(m) for v, m in messages.items()},
                    "visible_packet_sha256": sha256_json(packets["control"]),
                }
            )
        return {"schema_version": "dedup-sequencing-presentation-v1", "semantic_policy": policy, "called_inputs": rows}

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "SEQUENCING_ROOT_EXISTS", "use a fresh root")
        old = common.validate_freeze(parent)
        require(
            old["schema_version"] == "dedup-typed-policy-experiment-v1",
            "SEQUENCING_PARENT",
            "inherit frozen policy population",
        )
        config = load_sequencing_config(self.config["coverage"])
        files = [
            Path(__file__),
            self.spec_path,
            *self.config.values(),
            boundary,
            REPO / self.spec["design"],
            REPO / self.spec["protocol"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            *[
                self.config["coverage"].parent / config[k]
                for k in ("system_instruction_path", "pair_instruction_path")
            ],
            REPO / "eval/dedup/judging/sequencing_runtime.py",
            REPO / "tests/eval/dedup/test_sequencing_runtime.py",
            REPO / "tests/eval/dedup/test_sequencing_experiment.py",
        ]
        evidence = json.loads(boundary.read_text())
        required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} | {str(self.spec_path)}
        require(
            evidence["passed"]
            and evidence["external_model_calls"] == 0
            and evidence["full_boundary_tests_passed"] == 4
            and required <= evidence["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
            "SEQUENCING_BOUNDARY",
            "current full native first/retry checks required for both arms",
        )
        labels = common._read_csv(Path(old["paths"]["labels"]))
        common.validate_projection(labels, common._read_csv(Path(old["paths"]["reference"])))
        inputs = {
            r: [
                {**p, "repair_feedback": ""}
                for p in common.blind_rows(common._jsonl(parent / f"input_repeat_{r}.jsonl"), repeat=r)
            ]
            for r in (1, 2)
        }
        mains = json.loads((parent / "fixed_main.json").read_text())
        require(
            all(
                len(rows) == 258
                and common._index(rows, "inputs").keys() == mains.keys() == common._index(labels, "labels").keys()
                for rows in inputs.values()
            ),
            "SEQUENCING_MEMBERSHIP",
            "all frozen pairs, labels, weights and main responses required",
        )
        calls = self.call_sets(inputs, mains)
        require(
            all(
                calls[f"repeat_{r}_control"] == calls[f"repeat_{r}_coverage"]
                and len(calls[f"repeat_{r}_control"]["called_pair_ids"]) == 147
                for r in (1, 2)
            ),
            "SEQUENCING_CALLS",
            "same 147 called and 111 owned rows required",
        )
        presentation = self.presentation_audit(inputs[1], mains)
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        schemas = {"control": typed_coverage_schema(), "coverage": sequencing_schema()}
        manifest = {
            **{k: v for k, v in old.items() if k not in {"diagnostic_contract_digest", "policy_contrast_sha256"}},
            "schema_version": "dedup-sequencing-experiment-v1",
            "experiment_spec": str(self.spec_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "parent_diagnostic": str(parent),
            "parent_contract_digest": old["diagnostic_contract_digest"],
            "settings": {**old["settings"], "ray_temp_dir": self.spec["ray_temp_dir"], "max_parallel_requests": 16},
            "paths": {**old["paths"], "protocol": str(REPO / self.spec["protocol"])},
            "frozen_files": old["frozen_files"] | {str(p.resolve()): sha256_file(p) for p in files},
            "variant_contracts": {
                v: {
                    "runner_config": str(self.config[v]),
                    "adapter_policy": POLICY,
                    "routing": ROUTING_CONTRACT,
                    "response_contract": CONTRACT,
                    "semantic_policy": "complete_propositions",
                    "semantic_policy_sha256": presentation["semantic_policy"]["block_sha256"],
                    "response_field_order": "compare_then_bind" if v == "coverage" else "scope_before_sides",
                }
                for v in self.config
            },
            "response_schemas": schemas,
            "coverage_schema_sha256": sha256_json(schemas["coverage"]),
            "call_sets_sha256": sha256_json(calls),
            "token_preflight_sha256": sha256_json(tokens),
            "presentation_audit_sha256": sha256_json(presentation),
            "formal_requests": 588,
            "formal_public_pairs": 1032,
        }
        manifest["diagnostic_contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "manifest.json", manifest)
        for name, value in (
            ("fixed_main", mains),
            ("call_sets", calls),
            ("token_preflight", tokens),
            ("coverage_schema", schemas["coverage"]),
            ("response_schemas", schemas),
            ("presentation_audit", presentation),
        ):
            write_json_atomic(root / f"{name}.json", value)
        for r, rows in inputs.items():
            common._write_jsonl(root / f"input_repeat_{r}.jsonl", rows)
        write_json_atomic(
            root / "input_freeze.json",
            {str(p): sha256_file(p) for p in root.glob("*.json*") if p.name != "manifest.json"},
        )
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment = SequencingExperiment(args.spec)
    if args.action == "prepare":
        require(
            args.parent is not None and args.boundary is not None, "SEQUENCING_PREPARE", "parent and boundary required"
        )
        print(
            json.dumps(
                {"contract": experiment.prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}
            )
        )
    elif args.action == "run":
        experiment.run(args.root)
    else:
        require(
            args.output is not None and not args.output.exists(), "SEQUENCING_REPORT", "fresh report path required"
        )
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
