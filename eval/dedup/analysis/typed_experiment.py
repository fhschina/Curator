# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Typed-versus-flat coverage encoding with shared proposition policy and scope arbitration."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.coverage_experiment import REPO, common, native
from eval.dedup.analysis.proposition_experiment import PropositionExperiment, SelectionArm
from eval.dedup.analysis.selection_scope_replay import POLICY, probe_scope
from eval.dedup.judging.coverage_routing import ROUTING_CONTRACT
from eval.dedup.judging.coverage_selection import COLUMN
from eval.dedup.judging.coverage_selection import CONTRACT as FLAT_CONTRACT
from eval.dedup.judging.payload_transport import bind_transported_payload
from eval.dedup.judging.proposition_runtime import proposition_schema
from eval.dedup.judging.typed_coverage import (
    CONTRACT,
    bind_typed_response,
    compile_typed_coverage,
    typed_coverage_schema,
)
from eval.dedup.judging.typed_coverage_runtime import TypedCoverageRuntime, typed_renderer
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class TypedArm(SelectionArm):
    @cached_property
    def renderers(self) -> dict:
        return {"coverage": typed_renderer(self.config["coverage"])}

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        require(variant == "coverage", "TYPED_ARM", "single-arm coverage binding required")
        return TypedCoverageRuntime(
            self.config["coverage"],
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

        require(variant == "coverage", "TYPED_ARM", "single-arm coverage binding required")
        packets, raw = common._index(inputs, "typed inputs"), common._index(outputs, "typed outputs")
        require(packets.keys() == raw.keys(), "TYPED_OUTPUT_MISSING", "every called pair requires its response")
        called, owned = self.layout(inputs, mains, variant)
        require(not owned and len(called) == len(inputs), "FOLLOWUP_BOUNDARY_ROUTING", "owned branch was submitted")
        predictions = []
        for pid, row in raw.items():
            packet = packets[pid]
            _, payload_audit = bind_transported_payload(packet, row)
            require(
                isinstance(packet["repair_feedback"], str) and row.get("repair_feedback") == packet["repair_feedback"],
                "FOLLOWUP_BOUNDARY_FEEDBACK",
                "actual feedback changed",
            )
            require(
                not ({"qwen_dedup_semantic_judge", common.RECORD_BINDING_CRITIC_COLUMN} & row.keys()),
                "COVERAGE_COLUMN_LEAK",
                "typed reviewer cannot observe main or another critic",
            )
            trace = row.get(COLUMN + "__trace")
            require(
                isinstance(trace, list) and len(trace) >= 3 and all(isinstance(m, dict) for m in trace),
                "FOLLOWUP_BOUNDARY_TRACE",
                "full native conversation required",
            )
            messages = [{"role": m.get("role"), "content": native.trace_text(m.get("content"))} for m in trace[:2]]
            require(
                messages == self.renderers[variant](packet),
                "FOLLOWUP_BOUNDARY_PROMPT",
                "actual initial request differs",
            )
            assistants = [m for m in trace if m.get("role") == "assistant"]
            require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "assistant response missing")
            value = StructuredResponseRecipe(typed_coverage_schema(), pruning=False).parse(
                native.trace_text(assistants[-1].get("content"))
            )
            bound, response_audit = bind_typed_response(value, row.get(COLUMN))
            compiled = compile_typed_coverage(bound, packet["payload"])
            before, public, scope_audit = probe_scope(mains[pid], compiled.selection, packet["payload"])
            predictions.append(
                {
                    "canonical_pair_id": pid,
                    "diagnostic_only": True,
                    "diagnostic_version": self.spec["version"],
                    "fixed_main_sha256": sha256_json(mains[pid]),
                    "critic_request_status": "REQUESTED",
                    "coverage_contract": CONTRACT,
                    "typed_response_sha256": sha256_json(value),
                    "compiled_selection_sha256": sha256_json(compiled.selection),
                    "typed_compilation": compiled.audit,
                    "payload_transport": payload_audit,
                    "response_transport": response_audit,
                    "native_request_messages_sha256": sha256_json(messages),
                    "native_trace_sha256": sha256_json(trace),
                    "raw_output_sha256": sha256_json(row),
                    "native_corrections": len(assistants) - 1,
                    "scope_postprocessor": POLICY,
                    "scope_postprocessor_audit": scope_audit,
                    "base_public_sha256": sha256_json(before),
                    **public,
                }
            )
        return predictions


class TypedExperiment(PropositionExperiment):
    def __init__(self, spec_path: Path):
        self.spec_path = spec_path.resolve()
        self.spec = json.loads(self.spec_path.read_text())
        require(
            self.spec["control_routing"] == self.spec["coverage_routing"] == ROUTING_CONTRACT
            and self.spec["control_contract"] == FLAT_CONTRACT
            and self.spec["coverage_contract"] == CONTRACT
            and self.spec["scope_postprocessor"] == POLICY
            and self.spec["max_parallel_requests"] == 16,
            "TYPED_EXPERIMENT_CONTRACT",
            "fixed routing, typed/flat contracts, scope policy and budget required",
        )
        self.config = {v: (REPO / self.spec[v + "_config"]).resolve() for v in ("control", "coverage")}
        require(
            self.config["control"] == REPO / "eval/dedup/resources/local_ndd/hs_v06225_proposition_qwen_c16.yaml",
            "TYPED_EXPERIMENT_CONTROL",
            "flat control must preserve the frozen proposition prompt",
        )
        self.arms = {
            "control": SelectionArm(self.config["control"], version=self.spec["version"], evidence_first=True),
            "coverage": TypedArm(self.config["coverage"], version=self.spec["version"], evidence_first=True),
        }

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "TYPED_ROOT_EXISTS", "use a new run root")
        old = common.validate_freeze(parent)
        require(
            old["schema_version"] == "dedup-proposition-experiment-v1",
            "TYPED_PARENT",
            "inherit frozen proposition population",
        )
        config = yaml.safe_load(self.config["coverage"].read_text())
        files = [
            Path(__file__),
            self.spec_path,
            *self.config.values(),
            boundary,
            REPO / self.spec["design"],
            REPO / self.spec["protocol"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            *[self.config["coverage"].parent / config[k] for k in ("system_prompt_path", "prompt_path")],
            REPO / "eval/dedup/judging/typed_coverage.py",
            REPO / "eval/dedup/judging/typed_coverage_runtime.py",
            REPO / "tests/eval/dedup/test_typed_coverage.py",
            REPO / "tests/eval/dedup/test_typed_coverage_runtime.py",
            REPO / "tests/eval/dedup/test_typed_experiment.py",
        ]
        evidence = json.loads(boundary.read_text())
        required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} | {str(self.spec_path)}
        require(
            evidence["passed"]
            and evidence["external_model_calls"] == 0
            and evidence["full_boundary_tests_passed"] == 4
            and required <= evidence["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in evidence["artifacts"].items()),
            "TYPED_BOUNDARY",
            "current mixed-branch first/retry native boundary required for both arms",
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
            "TYPED_MEMBERSHIP",
            "all frozen pairs, mains and original weights required",
        )
        calls = self.call_sets(inputs, mains)
        require(
            all(
                calls[f"repeat_{r}_control"] == calls[f"repeat_{r}_coverage"]
                and len(calls[f"repeat_{r}_control"]["called_pair_ids"]) == 147
                for r in (1, 2)
            ),
            "TYPED_CALLS",
            "both arms require same 147 called and 111 owned rows",
        )
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        schemas = {"control": proposition_schema(), "coverage": typed_coverage_schema()}
        manifest = {
            **{k: v for k, v in old.items() if k != "diagnostic_contract_digest"},
            "schema_version": "dedup-typed-coverage-experiment-v1",
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
                    "response_contract": self.spec[v + "_contract"],
                    "evidence_first": True,
                }
                for v in self.config
            },
            "response_schemas": schemas,
            "coverage_schema_sha256": sha256_json(typed_coverage_schema()),
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
            ("coverage_schema", typed_coverage_schema()),
            ("response_schemas", schemas),
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
    experiment = TypedExperiment(args.spec)
    if args.action == "prepare":
        require(args.parent is not None and args.boundary is not None, "TYPED_PREPARE", "parent and boundary required")
        print(
            json.dumps(
                {"contract": experiment.prepare(args.root, args.parent, args.boundary)["diagnostic_contract_digest"]}
            )
        )
    elif args.action == "run":
        experiment.run(args.root)
    else:
        require(args.output is not None and not args.output.exists(), "TYPED_REPORT", "new output path required")
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
