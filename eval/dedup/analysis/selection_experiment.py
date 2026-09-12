# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""V2 selection binding on the shared frozen paired-experiment execution engine."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.coverage_experiment import REPO, CoverageExperiment, common, native
from eval.dedup.judging.coverage_selection import (
    CONTRACT,
    adapt_selection,
    compile_selection,
    selection_schema,
    span_inventory,
)
from eval.dedup.judging.payload_transport import bind_transported_payload
from eval.dedup.judging.selection_runtime import SelectionRuntime, selection_renderer
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


class SelectionExperiment(CoverageExperiment):
    def __init__(self, spec_path: Path):
        super().__init__(spec_path)
        require(self.spec["coverage_contract"] == CONTRACT, "SELECTION_EXPERIMENT_CONTRACT", "wrong selection schema")
        require(
            self.spec["max_parallel_requests"] == 16,
            "SELECTION_EXPERIMENT_BUDGET",
            "both arms use the predeclared concurrency",
        )

    @cached_property
    def renderers(self) -> dict:
        return {"control": native.renderer("control"), "coverage": selection_renderer(self.config["coverage"])}

    def layout(self, inputs: list[dict], mains: dict, variant: str) -> tuple[list[dict], list[dict]]:
        called, owned = super().layout(inputs, mains, variant)
        if variant == "coverage":
            for packet in called:
                span_inventory(packet["payload"])
        return called, owned

    def runtime(self, variant: str, endpoint: str, settings: dict) -> Any:
        if variant == "control":
            return super().runtime(variant, endpoint, settings)
        return SelectionRuntime(
            self.config[variant],
            endpoint=endpoint,
            model_name=settings["logical_model"],
            ray_temp_dir=settings["ray_temp_dir"],
        )

    def bind(self, inputs: list[dict], outputs: list[dict], mains: dict, variant: str) -> list[dict]:
        from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe

        if variant == "control":
            return super().bind(inputs, outputs, mains, variant)
        packets, raw = common._index(inputs, "selection inputs"), common._index(outputs, "selection outputs")
        require(packets.keys() == raw.keys(), "SELECTION_OUTPUT_MISSING", "every called input needs one response")
        called, owned = self.layout(inputs, mains, variant)
        require(not owned and len(called) == len(inputs), "FOLLOWUP_BOUNDARY_ROUTING", "owned branch was submitted")
        predictions = []
        for pid, row in raw.items():
            packet = packets[pid]
            _, audit = bind_transported_payload(packet, row)
            require(
                isinstance(packet["repair_feedback"], str) and row.get("repair_feedback") == packet["repair_feedback"],
                "FOLLOWUP_BOUNDARY_FEEDBACK",
                "actual feedback changed",
            )
            require(
                not ({"qwen_dedup_semantic_judge", common.RECORD_BINDING_CRITIC_COLUMN} & row.keys()),
                "COVERAGE_COLUMN_LEAK",
                "selection cannot observe main or control",
            )
            trace = row.get(common.COLUMN + "__trace")
            require(
                isinstance(trace, list) and len(trace) >= 3 and all(isinstance(m, dict) for m in trace),
                "FOLLOWUP_BOUNDARY_TRACE",
                "full native conversation required",
            )
            messages = [{"role": m.get("role"), "content": native.trace_text(m.get("content"))} for m in trace[:2]]
            require(
                messages == self.renderers[variant](packet),
                "FOLLOWUP_BOUNDARY_PROMPT",
                "actual initial request differs from frozen renderer",
            )
            assistants = [m for m in trace if m.get("role") == "assistant"]
            require(bool(assistants), "FOLLOWUP_BOUNDARY_TRACE", "missing assistant response")
            value = StructuredResponseRecipe(selection_schema(), pruning=False).parse(
                native.trace_text(assistants[-1].get("content"))
            )
            require(
                value == row.get(common.COLUMN),
                "SELECTION_RAW_PRUNED",
                "strict original response differs from parsed column",
            )
            compiled = compile_selection(value, packet["payload"])
            public = adapt_selection(mains[pid], value, packet["payload"])
            predictions.append(
                {
                    "canonical_pair_id": pid,
                    "diagnostic_only": True,
                    "diagnostic_version": self.spec["version"],
                    "fixed_main_sha256": sha256_json(mains[pid]),
                    "critic_request_status": "REQUESTED",
                    "selection_response_sha256": sha256_json(value),
                    "compiled_coverage_sha256": sha256_json(compiled),
                    "coverage_contract": CONTRACT,
                    "payload_transport": audit,
                    "native_request_messages_sha256": sha256_json(messages),
                    "native_trace_sha256": sha256_json(trace),
                    "raw_output_sha256": sha256_json(row),
                    "native_corrections": len(assistants) - 1,
                    **public,
                }
            )
        return predictions

    def prepare(self, root: Path, parent: Path, boundary: Path) -> dict:
        from transformers import AutoTokenizer

        require(not root.exists(), "SELECTION_ROOT_EXISTS", "use a fresh run root")
        old = common.validate_freeze(parent)
        require(
            old["schema_version"] == "dedup-routed-coverage-experiment-v1",
            "SELECTION_PARENT",
            "use the frozen routed population",
        )
        coverage = yaml.safe_load(self.config["coverage"].read_text())
        files = [
            Path(__file__),
            self.spec_path,
            *self.config.values(),
            boundary,
            REPO / self.spec["design"],
            REPO / self.spec["protocol"],
            parent / "manifest.json",
            parent / "input_freeze.json",
            self.config["coverage"].parent / coverage["system_prompt_path"],
            self.config["coverage"].parent / coverage["prompt_path"],
            REPO / "eval/dedup/judging/coverage_selection.py",
            REPO / "eval/dedup/judging/selection_runtime.py",
            REPO / "tests/eval/dedup/test_selection_experiment.py",
            REPO / "tests/eval/dedup/test_selection_runtime.py",
            REPO / "tests/eval/dedup/test_coverage_selection.py",
        ]
        report = json.loads(boundary.read_text())
        required = {str(p.resolve()) for p in files if p.suffix in {".py", ".yaml", ".jinja"}} | {str(self.spec_path)}
        require(
            report["passed"]
            and report["external_model_calls"] == 0
            and report["full_boundary_tests_passed"] == 4
            and required <= report["artifacts"].keys()
            and all(sha256_file(p) == h for p, h in report["artifacts"].items()),
            "SELECTION_BOUNDARY_UNVERIFIED",
            "both arms need current full-pipeline first/retry evidence",
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
                len(rows) == self.spec["public_pairs_per_cell"] == 258
                and common._index(rows, "inputs").keys() == common._index(labels, "labels").keys() == mains.keys()
                for rows in inputs.values()
            ),
            "SELECTION_MEMBERSHIP",
            "all frozen pairs and mains required",
        )
        calls = self.call_sets(inputs, mains)
        tokens = self.token_preflight(
            inputs[1], mains, AutoTokenizer.from_pretrained(common.TOKENIZER, local_files_only=True)
        )
        manifest = {
            **{k: v for k, v in old.items() if k != "diagnostic_contract_digest"},
            "schema_version": "dedup-selection-experiment-v1",
            "experiment_spec": str(self.spec_path),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "parent_diagnostic": str(parent),
            "parent_contract_digest": old["diagnostic_contract_digest"],
            "settings": {
                **old["settings"],
                "ray_temp_dir": self.spec["ray_temp_dir"],
                "max_parallel_requests": self.spec["max_parallel_requests"],
            },
            "paths": {**old["paths"], "protocol": str(REPO / self.spec["protocol"])},
            "frozen_files": old["frozen_files"] | {str(p.resolve()): sha256_file(p) for p in files},
            "variant_contracts": {
                "control": old["variant_contracts"]["control"],
                "coverage": {
                    "runner_config": str(self.config["coverage"]),
                    "adapter_policy": CONTRACT,
                    "routing": self.spec["coverage_routing"],
                },
            },
            "coverage_schema_sha256": sha256_json(selection_schema()),
            "call_sets_sha256": sha256_json(calls),
            "token_preflight_sha256": sha256_json(tokens),
            "formal_requests": sum(
                len(calls[f"repeat_{r}_{v}"]["called_pair_ids"]) for r in (1, 2) for v in ("control", "coverage")
            ),
        }
        manifest["diagnostic_contract_digest"] = sha256_json(manifest)
        write_json_atomic(root / "manifest.json", manifest)
        for name, value in (
            ("fixed_main", mains),
            ("call_sets", calls),
            ("token_preflight", tokens),
            ("coverage_schema", selection_schema()),
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
    experiment = SelectionExperiment(args.spec)
    if args.action == "prepare":
        require(
            args.parent is not None and args.boundary is not None,
            "SELECTION_PREPARE_INPUT",
            "supply parent and boundary evidence",
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
            args.output is not None and not args.output.exists(),
            "SELECTION_OUTPUT_REQUIRED",
            "supply fresh report path",
        )
        result = experiment.summarize(args.root, args.output)
        print(json.dumps({"status": result["status"], "eligible_for_release": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
