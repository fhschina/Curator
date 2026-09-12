# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired semantic blocks under identical typed contracts and fixed main responses."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from eval.dedup.analysis.coverage_experiment import REPO, common
from eval.dedup.analysis.proposition_experiment import PropositionExperiment
from eval.dedup.analysis.selection_scope_replay import POLICY
from eval.dedup.analysis.typed_experiment import TypedArm
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT
from eval.dedup.judging.policy_runtime import (
    BASE,
    PolicyCoverageRuntime,
    load_policy_config,
    policy_description,
    policy_renderer,
)
from eval.dedup.judging.typed_coverage import CONTRACT, typed_coverage_schema
from eval.dedup.judging.typed_coverage_runtime import typed_renderer
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class PolicyArm(TypedArm):
    @cached_property
    def renderers(self) -> dict:
        return {"coverage": policy_renderer(self.config["coverage"])}

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "POLICY_ARM", "single typed arm required")
        return PolicyCoverageRuntime(
            self.config["coverage"],
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )


class PolicyExperiment(PropositionExperiment):
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
            "POLICY_EXPERIMENT_CONTRACT",
            "identical typed contracts, routing, postprocessor and budget required",
        )
        self.config = {v: (REPO / self.spec[v + "_config"]).resolve() for v in ("control", "coverage")}
        self.policies = {
            v: policy_description(load_policy_config(p)["semantic_policy"]) for v, p in self.config.items()
        }
        require(
            self.policies["control"]["name"] == "concise_boundaries"
            and self.policies["coverage"]["name"] == "complete_propositions",
            "POLICY_EXPERIMENT_ARMS",
            "predeclared semantic arms required",
        )
        self.arms = {
            v: PolicyArm(p, version=self.spec["version"], evidence_first=True) for v, p in self.config.items()
        }

    def contrast(self, inputs: list[dict], mains: dict) -> dict:
        baseline = typed_renderer(BASE)
        called, _ = self.layout(inputs, mains, "coverage")
        records = []
        for packet in called:
            messages = {v: render(packet) for v, render in self.renderers.items()}
            common_messages = {}
            for v, value in messages.items():
                block = self.policies[v]["block"]
                require(value[0]["content"].count(block) == 1, "POLICY_CONTRAST_BLOCK", "one policy block required")
                common_messages[v] = [
                    {**value[0], "content": value[0]["content"].replace(block, "<STATIC_SEMANTIC_POLICY>")},
                    *value[1:],
                ]
            require(
                common_messages["control"] == common_messages["coverage"] and messages["coverage"] == baseline(packet),
                "POLICY_CONTRAST_CHANGED",
                "only the semantic block may vary; candidate must preserve the prior typed request",
            )
            records.append(
                {
                    "canonical_pair_id": packet["canonical_pair_id"],
                    "request_sha256": {v: sha256_json(m) for v, m in messages.items()},
                    "common_messages_sha256": sha256_json(common_messages["control"]),
                }
            )
        return {"schema_version": "dedup-policy-contrast-v1", "policies": self.policies, "called_inputs": records}

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "POLICY_ROOT_EXISTS", "new run root required")
        old = common.validate_freeze(parent)
        require(
            old["schema_version"] == "dedup-typed-coverage-experiment-v1", "POLICY_PARENT", "typed parent required"
        )
        files = [
            Path(__file__),
            self.spec_path,
            *self.config.values(),
            boundary,
            REPO / self.spec["design"],
            REPO / self.spec["protocol"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            REPO / "eval/dedup/judging/policy_runtime.py",
            REPO / "tests/eval/dedup/test_policy_runtime.py",
            REPO / "tests/eval/dedup/test_policy_experiment.py",
        ]
        evidence = json.loads(boundary.read_text())
        required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml"}} | {str(self.spec_path)}
        require(
            evidence["passed"]
            and evidence["external_model_calls"] == 0
            and evidence["full_boundary_tests_passed"] == 4
            and required <= evidence["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
            "POLICY_BOUNDARY",
            "both typed arms require current native first/retry evidence",
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
            "POLICY_MEMBERSHIP",
            "all original pairs, weights and fixed mains required",
        )
        calls = self.call_sets(inputs, mains)
        require(
            all(
                calls[f"repeat_{r}_control"] == calls[f"repeat_{r}_coverage"]
                and len(calls[f"repeat_{r}_control"]["called_pair_ids"]) == 147
                for r in (1, 2)
            ),
            "POLICY_CALLS",
            "same 147 called and 111 owned rows required",
        )
        contrast = self.contrast(inputs[1], mains)
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        schemas = {v: typed_coverage_schema() for v in self.config}
        manifest = {
            **{k: v for k, v in old.items() if k != "diagnostic_contract_digest"},
            "schema_version": "dedup-typed-policy-experiment-v1",
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
                    "evidence_first": True,
                    "semantic_policy": self.policies[v]["name"],
                    "semantic_policy_sha256": self.policies[v]["block_sha256"],
                }
                for v in self.config
            },
            "response_schemas": schemas,
            "coverage_schema_sha256": sha256_json(typed_coverage_schema()),
            "call_sets_sha256": sha256_json(calls),
            "token_preflight_sha256": sha256_json(tokens),
            "policy_contrast_sha256": sha256_json(contrast),
            "formal_requests": 588,
            "formal_public_pairs": 1032,
        }
        manifest["diagnostic_contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "manifest.json", manifest)
        for name, value in (
            ("fixed_main", mains),
            ("call_sets", calls),
            ("token_preflight", tokens),
            ("coverage_schema", typed_coverage_schema()),
            ("response_schemas", schemas),
            ("policy_contrast", contrast),
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
    experiment = PolicyExperiment(args.spec)
    if args.action == "prepare":
        require(
            args.parent is not None and args.boundary is not None, "POLICY_PREPARE", "parent and boundary required"
        )
        print(
            json.dumps(
                {"contract": experiment.prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}
            )
        )
    elif args.action == "run":
        experiment.run(args.root)
    else:
        require(args.output is not None and not args.output.exists(), "POLICY_REPORT", "new report path required")
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
