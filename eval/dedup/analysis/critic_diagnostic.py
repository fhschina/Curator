# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paired critic-only experiments with frozen main responses; never release evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.development_diagnostic import _index, evaluate_primary_guards, load_run, matched_raw_outputs
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, evaluate_predictions
from eval.dedup.analysis.local_iteration import compare_rows, guard_report, local_gates, validate_projection
from eval.dedup.analysis.policy_review import _read_csv
from eval.dedup.judging.local_ndd import (
    RECORD_BINDING_CRITIC_COLUMN,
    _read_output_rows,
    _safe_retry_feedback,
    adapt_ndd_judge_output,
)
from eval.dedup.judging.payload import assert_blind_payload, validate_evidence_offsets
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

ANALYSIS = Path(__file__).resolve().parent
RESOURCES = ANALYSIS.parent / "resources/local_ndd"
VARIANTS = ("control", "consent", "pointer")
TARGETS = {"consent": "H0347", "pointer": "H0748"}


def critic_config(variant: str) -> Path:
    require(variant in VARIANTS, "CRITIC_VARIANT", "unknown diagnostic arm")
    return RESOURCES / f"hs_v06216_{variant}_qwen_c64.yaml"


def validate_critic_config(path: Path) -> None:
    """The control must keep the actual NDD judge name and rubric, with no main column."""
    source = yaml.safe_load((RESOURCES / "hs_v06214_qwen_c64.yaml").read_text())
    config = yaml.safe_load(path.read_text())
    judges = [judge for stage in config["execution"]["stages"] for judge in stage["judges"]]
    original = source["execution"]["stages"][0]["judges"][1]
    require(
        len(judges) == 1 and judges[0]["name"] == original["name"],
        "CRITIC_ONLY",
        "only the original critic column is allowed",
    )
    require(config["models"] == source["models"], "CRITIC_MODEL_CHANGED", "diagnostic model defaults differ")
    require(
        {s["name"]: s["options"] for s in judges[0]["scores"]}
        == {s["name"]: s["options"] for s in original["scores"]},
        "CRITIC_SCHEMA_CHANGED",
        "critic fields and options must remain unchanged",
    )
    if path == critic_config("control"):
        require(
            judges[0] == original, "CRITIC_CONTROL_CHANGED", "control prompt and rubric must equal the original critic"
        )


def blind_rows(payloads: list[dict], *, repeat: int) -> list[dict]:
    """Neither fixed main responses, review IDs, references nor arm names reach NDD."""
    _index(payloads, "payloads")
    rows = []
    for packet in payloads:
        assert_blind_payload(packet["payload"])
        rows.append(
            {
                "canonical_pair_id": packet["canonical_pair_id"],
                "judge_payload_hash": sha256_json(packet["payload"]),
                "payload": packet["payload"],
                "repair_feedback": None,
            }
        )
    return sorted(rows, key=lambda r: sha256_json(["v06216-paired-order", repeat, r["canonical_pair_id"]]))


def bind_fixed_main(inputs: list[dict], outputs: list[dict], main_by_pair: dict[str, dict]) -> list[dict]:
    expected, actual = _index(inputs, "critic inputs"), _index(outputs, "critic outputs")
    require(actual.keys() == expected.keys(), "CRITIC_MEMBERSHIP", "critic output membership differs")
    predictions = []
    for pair_id, row in actual.items():
        packet = expected[pair_id]
        require(
            row.get("judge_payload_hash") == packet["judge_payload_hash"],
            "CRITIC_PAYLOAD_CHANGED",
            "critic response belongs to another payload",
        )
        require(
            "qwen_dedup_semantic_judge" not in row,
            "CRITIC_MAIN_CALLED",
            "a diagnostic must never invoke or replace main",
        )
        value = row.get(RECORD_BINDING_CRITIC_COLUMN)
        require(isinstance(value, dict), "CRITIC_OUTPUT_MISSING", "critic output is missing")
        public = adapt_ndd_judge_output(
            main_by_pair[pair_id],
            "dedup-judge-output-v3",
            payload=packet["payload"],
            record_binding_critic=value,
            record_binding_policy="v8",
        )
        validate_evidence_offsets(public, packet["payload"])
        predictions.append(
            {
                "canonical_pair_id": pair_id,
                "diagnostic_only": True,
                "fixed_main_sha256": sha256_json(main_by_pair[pair_id]),
                "critic_response_sha256": sha256_json(value),
                **public,
            }
        )
    return predictions


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))


def prepare(root: Path, source_root: Path, baseline_root: Path, protocol: Path) -> dict:
    from eval.dedup.rejudge_comparison import _resource_hashes, _source_digest

    require(not root.exists(), "CRITIC_ROOT_EXISTS", "use an empty diagnostic root; never reuse judge cache")
    manifest, _, source, payloads = load_run(source_root)
    _, _, _, baseline_payloads = load_run(baseline_root)
    require(
        manifest["settings"]["prompt_version"] == "dedup-judge-hs-v0.6.2.14",
        "CRITIC_SOURCE",
        "fixed main must come from .14",
    )
    paths = {
        "labels": ANALYSIS / "v06213_local_evaluation_labels.csv",
        "reference": ANALYSIS / "v0628_policy_reconciled_labels_1000.csv",
        "negative_guards": ANALYSIS / "v06212_translation_route_replay.json",
        "additional_guards": ANALYSIS / "v06215_critic_repaired_guards.json",
        "protocol": protocol.resolve(),
    }
    labels = _read_csv(paths["labels"])
    validate_projection(labels, _read_csv(paths["reference"]))
    require(
        sha256_file(paths["labels"]) == "977eb95ba066655d03ff533ba98d5c5fe92fa3cb3f63148856159f39d09c1f2a"
        and sha256_file(paths["reference"]) == "6f685cdf717df61a9332e431c685a9d860348a6c83e3665356075e8dc6c28096",
        "CRITIC_REFERENCE_CHANGED",
        "this experiment requires the unchanged frozen reference and weights",
    )
    require(
        len(labels) == 258 and _index(labels, "labels").keys() == _index(source, "source").keys(),
        "CRITIC_MEMBERSHIP",
        "all frozen 258 pairs are required",
    )
    baseline_packets = _index(baseline_payloads, "baseline payloads")
    require(
        all(r["payload"] == baseline_packets[r["canonical_pair_id"]]["payload"] for r in payloads),
        "CRITIC_PAYLOAD_CHANGED",
        "baseline visible payloads differ",
    )
    raw, digests = matched_raw_outputs(source_root, source)
    inputs = blind_rows(payloads, repeat=1)
    mains = {p: values[0] for p, values in raw.items()}
    replay = bind_fixed_main(
        inputs, [{**row, RECORD_BINDING_CRITIC_COLUMN: raw[row["canonical_pair_id"]][1]} for row in inputs], mains
    )
    published = _index(source, "source")
    for row in replay:
        require(
            all(row[k] == v for k, v in published[row["canonical_pair_id"]].items() if k in row),
            "CRITIC_REPLAY_CHANGED",
            "historical v8 replay differs",
        )
    resources = {}
    for variant in VARIANTS:
        path = critic_config(variant)
        validate_critic_config(path)
        resources.update({str(path.parent / p): d for p, d in _resource_hashes(path).items()})
    for run in (source_root, baseline_root):
        for name in (
            "run_manifest.json",
            "run_complete.json",
            "data/judge_results.jsonl",
            "data/judge_payloads.jsonl",
        ):
            paths[f"{run.name}/{name}"] = run / name
    frozen = {str(path.resolve()): sha256_file(path) for path in paths.values()}
    frozen.update(resources)
    frozen.update(digests)
    frozen[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    frozen[str(ANALYSIS / "local_iteration.py")] = sha256_file(ANALYSIS / "local_iteration.py")
    frozen[str(ANALYSIS / "development_diagnostic.py")] = sha256_file(ANALYSIS / "development_diagnostic.py")
    frozen[str(ANALYSIS / "judge_calibration.py")] = sha256_file(ANALYSIS / "judge_calibration.py")
    settings = dict(manifest["settings"])
    settings["ray_temp_dir"] = "/raid/hfang/ihb/r16diag"
    schedule = [(repeat, arm) for repeat, arms in ((1, VARIANTS), (2, tuple(reversed(VARIANTS)))) for arm in arms]
    result = {
        "schema_version": "dedup-critic-paired-diagnostic-v1",
        "diagnostic_only": True,
        "eligible_for_release": False,
        "source_root": str(source_root),
        "baseline_root": str(baseline_root),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_implementation_sha256": _source_digest(),
        "settings": settings,
        "frozen_files": frozen,
        "paths": {k: str(v) for k, v in paths.items()},
        "schedule": [{"repeat": repeat, "variant": arm} for repeat, arm in schedule],
        "preflight_pairs_per_arm": 8,
        "formal_pairs_per_cell": 258,
        "formal_requests": 1548,
        "fixed_main_sha256": sha256_json(mains),
    }
    result["diagnostic_contract_digest"] = sha256_json(result)
    write_json_atomic(root / "manifest.json", result)
    write_json_atomic(root / "fixed_main.json", mains)
    for repeat in (1, 2):
        _write_jsonl(root / f"input_repeat_{repeat}.jsonl", blind_rows(payloads, repeat=repeat))
    write_json_atomic(
        root / "input_freeze.json",
        {str(p): sha256_file(p) for p in sorted(root.glob("input_*.jsonl"))}
        | {str(root / "fixed_main.json"): sha256_file(root / "fixed_main.json")},
    )
    return result


def validate_freeze(root: Path) -> dict:
    from eval.dedup.rejudge_comparison import _source_digest

    manifest = json.loads((root / "manifest.json").read_text())
    digest = manifest["diagnostic_contract_digest"]
    require(
        sha256_json({k: v for k, v in manifest.items() if k != "diagnostic_contract_digest"}) == digest,
        "CRITIC_MANIFEST_CHANGED",
        "diagnostic manifest changed",
    )
    require(
        _source_digest() == manifest["source_implementation_sha256"],
        "CRITIC_CODE_CHANGED",
        "arbitration/runtime source changed",
    )
    frozen = manifest["frozen_files"] | json.loads((root / "input_freeze.json").read_text())
    require(
        all(sha256_file(p) == d for p, d in frozen.items()),
        "CRITIC_FREEZE_CHANGED",
        "frozen source, prompt, labels, payload or main response changed",
    )
    require(
        sha256_json(json.loads((root / "fixed_main.json").read_text())) == manifest["fixed_main_sha256"],
        "CRITIC_MAIN_CHANGED",
        "fixed main changed",
    )
    return manifest


def run_cell(root: Path, inputs: list[dict], mains: dict, runtime: Any, relay: Any, settings: dict) -> dict:
    from eval.dedup.judging.request_relay import RelayContext

    require(not root.exists(), "CRITIC_CELL_EXISTS", "never reuse a partially observed cell")
    pending, accepted, retried = inputs, [], set()
    for attempt in range(1, settings["max_retries"] + 2):
        attempt_root = root / f"attempt_{attempt:02d}"
        _write_jsonl(attempt_root / "input.jsonl", pending)
        relay.set_context(RelayContext(root.name, attempt, attempt_root / "events.jsonl"))
        runtime.run(
            input_path=str(attempt_root / "input.jsonl"),
            input_format="jsonl",
            output_path=str(attempt_root / "output"),
            output_format="jsonl",
            checkpoint_path=str(attempt_root / "checkpoints"),
            files_per_partition=1,
            inference_parameter_overrides={
                "judge": {
                    "temperature": settings["temperature"],
                    "top_p": settings["top_p"],
                    "max_tokens": settings["max_output_tokens"],
                    "timeout": settings["timeout_seconds"],
                    "max_parallel_requests": settings["max_parallel_requests"],
                }
            },
        )
        outputs = _index(_read_output_rows(attempt_root / "output"), "critic output")
        require(outputs.keys() <= _index(pending, "pending").keys(), "CRITIC_MEMBERSHIP", "unexpected output pair")
        next_pending, errors = [], []
        for row in pending:
            pair_id = row["canonical_pair_id"]
            try:
                prediction = bind_fixed_main([row], [outputs[pair_id]] if pair_id in outputs else [], mains)[0]
                accepted.append({**prediction, "attempts": attempt, "retried": attempt > 1})
            except Exception as exc:  # noqa: BLE001 - schema failures share the bounded outer retry contract
                feedback = _safe_retry_feedback(exc)
                errors.append({"canonical_pair_id": pair_id, "issue": feedback})
                next_pending.append({**row, "repair_feedback": feedback})
                if attempt <= settings["max_retries"]:
                    retried.add(pair_id)
        write_json_atomic(attempt_root / "validation_errors.json", errors)
        pending = next_pending
        if not pending:
            break
    _write_jsonl(root / "predictions.jsonl", accepted)
    events = [e for p in root.glob("attempt_*/events.jsonl") for e in _jsonl(p)]
    complete = {
        "requested": len(inputs),
        "valid": len(accepted),
        "errors": len(pending),
        "retried": len(retried),
        "http_statuses": dict(Counter(str(e["http_status"]) for e in events)),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {str(p): sha256_file(p) for p in sorted(root.rglob("*.jsonl"))},
    }
    write_json_atomic(root / "complete.json", complete)
    return complete


def run(root: Path) -> None:
    from eval.dedup.judging.request_relay import RequestRelay
    from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

    manifest = validate_freeze(root)
    require(
        not (root / "started.json").exists(), "CRITIC_ALREADY_STARTED", "do not silently repeat observed experiments"
    )
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(bool(key), "CRITIC_CREDENTIAL_MISSING", "required inference credential is unavailable")
    mains = json.loads((root / "fixed_main.json").read_text())
    inputs = {repeat: _jsonl(root / f"input_repeat_{repeat}.jsonl") for repeat in (1, 2)}
    write_json_atomic(
        root / "started.json",
        {
            "at_utc": datetime.now(UTC).isoformat(),
            "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
        },
    )
    with RequestRelay(
        logical_model=settings["logical_model"],
        upstream_base_url=settings["hub_base_url"],
        upstream_model=settings["hub_model"],
        upstream_api_key=key,
        timeout_seconds=settings["timeout_seconds"],
        expected_generation_parameters={
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_output_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as relay:

        def runtime(variant: str) -> Any:
            return ExternalJudgeRuntime(
                critic_config(variant),
                endpoint=relay.endpoint,
                provider_api_key="unused",  # pragma: allowlist secret
                served_model_overrides={"judge": settings["logical_model"]},
                ray_temp_dir=settings["ray_temp_dir"],
                num_cpus=None,
            )

        with runtime("control"):
            schedule = [{"variant": v, "repeat": 0} for v in VARIANTS] + manifest["schedule"]
            for cell in schedule:
                validate_freeze(root)
                repeat, variant = cell["repeat"], cell["variant"]
                name = f"repeat_{repeat}_{variant}" if repeat else f"preflight_{variant}"
                selected = inputs[repeat] if repeat else inputs[1][: manifest["preflight_pairs_per_arm"]]
                with runtime(variant) as borrowed:
                    complete = run_cell(root / name, selected, mains, borrowed, relay, settings)
                print(
                    json.dumps({"cell": name, **{k: v for k, v in complete.items() if k != "artifacts"}}), flush=True
                )
                require(
                    complete["valid"] == complete["requested"] and complete["errors"] == 0,
                    "CRITIC_CELL_FAILED",
                    "incomplete cell stops the diagnostic without dropping pairs",
                )
    validate_freeze(root)
    write_json_atomic(
        root / "run_complete.json",
        {
            "diagnostic_only": True,
            "eligible_for_release": False,
            "formal_requests": 1548,
            "preflight_requests": 24,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def primary_changes(labels: list[dict], before: list[dict], after: list[dict]) -> list[dict]:
    return [
        r
        for r in compare_rows(labels, before, after)
        if any(r["source"][k] != r["candidate"][k] for k in PRIMARY_FIELDS)
    ]


def summarize(root: Path, output: Path) -> dict:
    manifest = validate_freeze(root)
    require((root / "run_complete.json").is_file(), "CRITIC_INCOMPLETE", "all scheduled cells are required")
    labels = _read_csv(Path(manifest["paths"]["labels"]))
    _, _, source, _ = load_run(Path(manifest["source_root"]))
    _, _, baseline, _ = load_run(Path(manifest["baseline_root"]))
    ids = _index(labels, "labels").keys()
    baseline = [r for r in baseline if r["canonical_pair_id"] in ids]
    negatives = json.loads(Path(manifest["paths"]["negative_guards"]).read_text())["critic_repaired_negative_guards"]
    additional = json.loads(Path(manifest["paths"]["additional_guards"]).read_text())
    require(len(negatives) == 38 and len(additional) == 22, "CRITIC_GUARDS", "complete frozen guard sets are required")
    metrics = {"baseline": evaluate_predictions(labels, baseline), "historical": evaluate_predictions(labels, source)}
    cells, predictions = {}, {}
    for cell in manifest["schedule"]:
        name = f"repeat_{cell['repeat']}_{cell['variant']}"
        folder = root / name
        complete = json.loads((folder / "complete.json").read_text())
        require(
            all(sha256_file(p) == d for p, d in complete["artifacts"].items()),
            "CRITIC_RESULTS_CHANGED",
            "cell raw responses or predictions changed",
        )
        rows = _jsonl(folder / "predictions.jsonl")
        require(
            _index(rows, name).keys() == ids and len(rows) == complete["valid"] == 258,
            "CRITIC_MEMBERSHIP",
            "incomplete scored cell",
        )
        predictions[name] = rows
        score = evaluate_predictions(labels, rows)
        guards = guard_report(labels, rows, baseline, negatives)
        guards["additional_critic_repair_guards"] = evaluate_primary_guards(additional, rows)
        cells[name] = {
            "metrics": score,
            "guards": guards,
            "operations": {k: v for k, v in complete.items() if k != "artifacts"},
            "local_gates": local_gates(score, metrics["baseline"], guards, complete, 258),
            "vs_historical": primary_changes(labels, source, rows),
        }
    paired, repeat_changes, decisions = {}, {}, {}
    for variant in VARIANTS:
        repeat_changes[variant] = primary_changes(
            labels, predictions[f"repeat_1_{variant}"], predictions[f"repeat_2_{variant}"]
        )
        if variant == "control":
            continue
        passing = []
        target = next(r for r in labels if r["review_id"] == TARGETS[variant])
        for repeat in (1, 2):
            name, control = f"repeat_{repeat}_{variant}", f"repeat_{repeat}_control"
            paired[name] = primary_changes(labels, predictions[control], predictions[name])
            target_prediction = _index(predictions[name], name)[target["canonical_pair_id"]]
            checks = {
                "local_gates": cells[name]["local_gates"]["passed"],
                "target_correct": all(target_prediction[k] == target[f"human_{k}"] for k in PRIMARY_FIELDS),
                "primary_not_below_paired_control": cells[name]["metrics"]["weighted"]["primary_decision_exact"]
                >= cells[control]["metrics"]["weighted"]["primary_decision_exact"],
                "over_group_not_above_paired_control": cells[name]["metrics"]["over_group"]
                <= cells[control]["metrics"]["over_group"],
            }
            passing.append(all(checks.values()))
            cells[name]["diagnostic_checks"] = checks
        decisions[variant] = {
            "target": TARGETS[variant],
            "supported_for_separate_fresh_local_validation": all(passing),
            "eligible_for_release": False,
        }
    result = {
        "schema_version": "dedup-critic-paired-assessment-v1",
        "diagnostic_only": True,
        "diagnostic_contract_digest": manifest["diagnostic_contract_digest"],
        "metrics": metrics,
        "cells": cells,
        "paired_changes": paired,
        "within_arm_repeat_changes": repeat_changes,
        "decisions": decisions,
        "eligible_for_release": False,
    }
    write_json_atomic(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.14-local-development"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.12-full-development"))
    parser.add_argument("--protocol", type=Path, default=ANALYSIS / "v06216_design.md")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        manifest = prepare(root, args.source_root.resolve(), args.baseline_root.resolve(), args.protocol)
        print(
            json.dumps(
                {
                    "contract": manifest["diagnostic_contract_digest"],
                    "formal_critic_requests": manifest["formal_requests"],
                }
            )
        )
    elif args.action == "run":
        run(root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "CRITIC_OUTPUT_EXISTS",
            "provide a fresh assessment path",
        )
        print(json.dumps(summarize(root, args.output)["decisions"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
