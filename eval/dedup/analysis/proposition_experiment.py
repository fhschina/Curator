# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired V2 selection prompts with identical ownership and scope postprocessing."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.coverage_experiment import REPO, CoverageExperiment, common
from eval.dedup.analysis.selection_experiment import SelectionExperiment
from eval.dedup.analysis.selection_scope_replay import POLICY, probe_scope
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT
from eval.dedup.judging.coverage_selection import COLUMN, CONTRACT, selection_schema
from eval.dedup.judging.proposition_runtime import PropositionRuntime, proposition_renderer, proposition_schema
from eval.dedup.judging.selection_runtime import SelectionRuntime, selection_renderer
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class SelectionArm(SelectionExperiment):
    """Inject one explicit prompt/runtime into the frozen selection binder, not its run driver."""

    def __init__(self, config: Path, *, version: str, evidence_first: bool):
        self.config = {"coverage": config}
        self.spec = {"version": version}
        self.evidence_first = evidence_first

    @cached_property
    def renderers(self) -> dict:
        factory = proposition_renderer if self.evidence_first else selection_renderer
        return {"coverage": factory(self.config["coverage"])}

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "PROPOSITION_ARM", "single-arm binding requires coverage ownership")
        cls = PropositionRuntime if self.evidence_first else SelectionRuntime
        return cls(
            self.config["coverage"],
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        require(variant == "coverage", "PROPOSITION_ARM", "single-arm binding requires coverage ownership")
        rows = super().bind(inputs, outputs, mains, variant)
        packets, raw = common._index(inputs, "inputs"), common._index(outputs, "outputs")
        result = []
        for row in rows:
            pid = row["canonical_pair_id"]
            before, after, audit = probe_scope(mains[pid], raw[pid][COLUMN], packets[pid]["payload"])
            require(
                all(row[k] == v for k, v in before.items()), "PROPOSITION_BASE_CHANGED", "strict base binding differs"
            )
            result.append(
                {
                    **row,
                    **after,
                    "scope_postprocessor": POLICY,
                    "base_public_sha256": sha256_json(before),
                    "scope_postprocessor_audit": audit,
                }
            )
        return result


class PropositionExperiment(CoverageExperiment):
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path.resolve()
        self.spec = json.loads(self.spec_path.read_text())
        require(
            self.spec["control_routing"] == self.spec["coverage_routing"] == ROUTING_CONTRACT
            and self.spec["coverage_contract"] == CONTRACT
            and self.spec["scope_postprocessor"] == POLICY
            and self.spec["max_parallel_requests"] == 16,
            "PROPOSITION_CONTRACT",
            "both arms require identical routing, schema, postprocessor and budget",
        )
        self.config = {v: (REPO / self.spec[v + "_config"]).resolve() for v in ("control", "coverage")}
        require(
            self.config["control"] == REPO / "eval/dedup/resources/local_ndd/hs_v06223_selection_qwen_c16.yaml",
            "PROPOSITION_CONTROL",
            "control must use the unmodified historical selection resource",
        )
        self.arms = {
            v: SelectionArm(self.config[v], version=self.spec["version"], evidence_first=v == "coverage")
            for v in self.config
        }

    @cached_property
    def renderers(self) -> dict:
        return {v: arm.renderers["coverage"] for v, arm in self.arms.items()}

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        return self.arms[variant].layout(inputs, mains, "coverage")

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        return self.arms[variant].runtime("coverage", endpoint, settings)

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        return self.arms[variant].bind(inputs, outputs, mains, "coverage")

    def run_cell(
        self, root: Path, inputs: list[dict], mains: dict, runtime: Any, relay: Any, settings: dict, variant: str
    ) -> dict:
        return self.arms[variant].run_cell(root, inputs, mains, runtime, relay, settings, "coverage")

    def replay_cell(
        self, root: Path, name: str, variant: str, inputs: list[dict], mains: dict
    ) -> tuple[list[dict], dict]:
        return self.arms[variant].replay_cell(root, name, "coverage", inputs, mains)

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "PROPOSITION_ROOT_EXISTS", "new run root required")
        old = common.validate_freeze(parent)
        require(
            old["schema_version"] == "dedup-selection-experiment-v1",
            "PROPOSITION_PARENT",
            "inherit the frozen selection population",
        )
        resources = [
            self.config["coverage"].parent / yaml.safe_load(self.config["coverage"].read_text())[k]
            for k in ("system_prompt_path", "prompt_path")
        ]
        files = [
            Path(__file__),
            self.spec_path,
            *self.config.values(),
            *resources,
            boundary,
            REPO / self.spec["design"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            REPO / "eval/dedup/judging/proposition_runtime.py",
            REPO / "eval/dedup/analysis/selection_scope_replay.py",
            common.ANALYSIS / "v06224_design.md",
            REPO / "tests/eval/dedup/test_proposition_runtime.py",
            REPO / "tests/eval/dedup/test_proposition_experiment.py",
            REPO / "tests/eval/dedup/test_selection_scope_replay.py",
        ]
        report = json.loads(boundary.read_text())
        required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} | {str(self.spec_path)}
        require(
            report["passed"]
            and report["external_model_calls"] == 0
            and report["full_boundary_tests_passed"] == 4
            and required <= report["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in report["artifacts"].items()),
            "PROPOSITION_BOUNDARY",
            "both fresh/retry native chains must be verified before freeze",
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
            "PROPOSITION_MEMBERSHIP",
            "all 258 original pairs, weights and mains required",
        )
        calls = self.call_sets(inputs, mains)
        for r in (1, 2):
            require(
                calls[f"repeat_{r}_control"] == calls[f"repeat_{r}_coverage"]
                and len(calls[f"repeat_{r}_control"]["called_pair_ids"]) == 147,
                "PROPOSITION_CALLS",
                "both arms need the same 147 called and 111 owned pairs",
            )
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        manifest = {
            **{k: v for k, v in old.items() if k != "diagnostic_contract_digest"},
            "schema_version": "dedup-proposition-experiment-v1",
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
                    "evidence_first": v == "coverage",
                }
                for v in self.config
            },
            "response_schemas": {"control": selection_schema(), "coverage": proposition_schema()},
            "call_sets_sha256": sha256_json(calls),
            "token_preflight_sha256": sha256_json(tokens),
            "formal_requests": 588,
            "formal_public_pairs": 1032,
        }
        manifest["diagnostic_contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "manifest.json", manifest)
        for name, value in (
            ("fixed_main", mains),
            ("call_sets", calls),
            ("token_preflight", tokens),
            ("coverage_schema", selection_schema()),
            ("response_schemas", manifest["response_schemas"]),
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
    experiment = PropositionExperiment(args.spec)
    if args.action == "prepare":
        require(
            args.parent is not None and args.boundary is not None, "PROPOSITION_PREPARE", "parent/boundary required"
        )
        print(
            json.dumps(
                {"contract": experiment.prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}
            )
        )
    elif args.action == "run":
        experiment.run(args.root)
    else:
        require(args.output is not None and not args.output.exists(), "PROPOSITION_REPORT", "new output required")
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
