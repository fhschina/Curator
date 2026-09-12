# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Original-line supplements to complete span quotations; a fixed-main diagnostic only."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval.dedup.analysis.critic_diagnostic import (
    ANALYSIS,
    RESOURCES,
    _jsonl,
    _write_jsonl,
    run_cell,
    validate_freeze,
)
from eval.dedup.analysis.presentation_diagnostic import presentation_record as validate_span_alignment
from eval.dedup.analysis.presentation_diagnostic import summarize
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic

VARIANTS = ("control", "ordered")
MAX_LINE_CHARS = 640
MAX_CONTEXT_CHARS = 8192
TOKENIZER = Path(
    "/raid/hfang/dedup_eval_cache/huggingface/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
)


def runner_config(variant: str) -> Path:
    require(variant in VARIANTS, "PRESENTATION_VARIANT", "unknown presentation arm")
    return RESOURCES / (
        "hs_v06216_control_qwen_c64.yaml" if variant == "control" else "hs_v06218_context_qwen_c64.yaml"
    )


def validate_presentation_config() -> None:
    control = yaml.safe_load(runner_config("control").read_text())
    ordered = yaml.safe_load(runner_config("ordered").read_text())
    ordered["execution"]["stages"][0]["name"] = control["execution"]["stages"][0]["name"]
    ordered["execution"]["stages"][0]["judges"][0]["prompt_path"] = control["execution"]["stages"][0]["judges"][0][
        "prompt_path"
    ]
    require(ordered == control, "PRESENTATION_POLICY_CHANGED", "only pair presentation and stage name may differ")


def presentation_record(row: dict) -> dict:
    """Reconnect lexical fragments to short original lines without replacing any span."""
    payload = row["payload"]
    packet = payload["semantic_diff_evidence"]
    complete = (
        packet["status"] == "COMPLETE"
        and payload["long_document_evidence"]["truncated"] is False
        and all(isinstance(payload[f"document_{side}"]["text"], str) for side in ("a", "b"))
    )
    if not complete:
        return {**row, "local_context": "", "local_context_status": "UNAVAILABLE_INCOMPLETE"}
    # Reuse the frozen alignment validator; its preview/index is not sent by this arm.
    validate_span_alignment(row)
    contexts = []
    skipped_long = 0
    for side in ("A", "B"):
        text = payload[f"document_{side.lower()}"]["text"]
        coordinates = []
        for span in packet["spans"]:
            if span["kind"] == "SHARED":
                prefix = f"{side.lower()}_"
            elif span["side"] == side:
                prefix = ""
            else:
                continue
            coordinates.append((span["span_id"], span["kind"], span[f"{prefix}start_char"], span[f"{prefix}end_char"]))
        start = 0
        for line in text.split("\n"):
            end = start + len(line)
            unique = [
                sid for sid, kind, left, right in coordinates if kind != "SHARED" and start <= left < right <= end
            ]
            shared = [
                sid for sid, kind, left, right in coordinates if kind == "SHARED" and left < end and right > start
            ]
            if unique and shared:
                if len(line) <= MAX_LINE_CHARS:
                    ids = [sid for sid, _, left, right in coordinates if left < end and right > start]
                    contexts.append(
                        {"side": side, "start_char": start, "end_char": end, "span_ids": ids, "text": line}
                    )
                else:
                    skipped_long += 1
            start = end + 1
    total_chars = sum(len(c["text"]) for c in contexts)
    if total_chars > MAX_CONTEXT_CHARS:
        return {
            **row,
            "local_context": "",
            "local_context_status": "UNAVAILABLE_SUPPLEMENT_BUDGET",
            "local_context_skipped_long_lines": skipped_long,
        }
    content = "\n".join(
        f"{c['side']}[{c['start_char']}:{c['end_char']}] existing spans {', '.join(c['span_ids'])}: "
        + json.dumps(c["text"], ensure_ascii=False)
        for c in contexts
    )
    return {
        **row,
        "local_context": content,
        "local_context_status": "AVAILABLE" if contexts else "NO_FRAGMENTED_SHORT_LINE",
        "local_context_skipped_long_lines": skipped_long,
    }


def message_renderer(variant: str) -> Callable[[dict], list[dict]]:
    """Use NDD's actual response recipe, including its generated schema instructions."""
    from data_designer.engine.column_generators.utils.prompt_renderer import (
        PromptType,
        RecordBasedPromptRenderer,
        create_response_recipe,
    )

    from eval.llm_judge.run_llm_judge import build_config_builder

    path = runner_config(variant)
    config = yaml.safe_load(path.read_text())
    builder, _ = build_config_builder(
        path,
        endpoint="http://127.0.0.1:1/v1",
        models=config["models"],
        judges=config["execution"]["stages"][0]["judges"],
    )
    column = builder.get_column_configs()[0]
    renderer = RecordBasedPromptRenderer(create_response_recipe(column))

    def render(row: dict) -> list[dict]:
        record = presentation_record(row)
        return [
            {"role": role, "content": renderer.render(prompt_template=template, record=record, prompt_type=kind)}
            for role, template, kind in (
                ("system", column.system_prompt, PromptType.SYSTEM_PROMPT),
                ("user", column.prompt, PromptType.USER_PROMPT),
            )
        ]

    return render


def token_preflight(
    inputs: list[dict],
    tokenizer: Any,
    *,
    context_budget: int = 32768,
    output_budget: int = 4096,
    safety_margin: int = 2048,
) -> dict:
    """No clipping or pair exclusion may hide the cost of repeating visible evidence."""
    rows = []
    for variant in VARIANTS:
        render = message_renderer(variant)
        for row in inputs:
            messages = render(row)
            count = len(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
                )
            )
            rows.append(
                {
                    "variant": variant,
                    "canonical_pair_id": row["canonical_pair_id"],
                    "message_sha256": sha256_json(messages),
                    "input_tokens": count,
                }
            )
    maximum = max(r["input_tokens"] for r in rows)
    require(
        maximum + output_budget + safety_margin <= context_budget,
        "PRESENTATION_CONTEXT_BUDGET",
        "presentation exceeds conservative local budget; do not silently truncate or drop pairs",
        maximum=maximum,
    )
    return {
        "context_budget": context_budget,
        "output_budget": output_budget,
        "safety_margin": safety_margin,
        "budget_source": "client safety budget from local runner declaration, not a claim about the Hub server limit",
        "max_input_tokens": {v: max(r["input_tokens"] for r in rows if r["variant"] == v) for v in VARIANTS},
        "rows": rows,
    }


def prepare(root: Path, parent: Path) -> dict:
    from transformers import AutoTokenizer

    from eval.dedup.rejudge_comparison import _resource_hashes

    require(not root.exists(), "PRESENTATION_ROOT_EXISTS", "use a new diagnostic root")
    old = validate_freeze(parent)
    require(
        old["schema_version"] == "dedup-presentation-diagnostic-v1"
        and len(_jsonl(parent / "input_repeat_1.jsonl")) == 258,
        "PRESENTATION_PARENT",
        "requires the frozen .17 258-pair diagnostic inputs",
    )
    validate_presentation_config()
    source_manifest = json.loads((Path(old["source_root"]) / "run_manifest.json").read_text())
    tokenizer_files = {
        str(TOKENIZER / name): digest for name, digest in source_manifest["tokenizer"]["asset_checksums"].items()
    }
    require(
        all(sha256_file(p) == digest for p, digest in tokenizer_files.items()),
        "PRESENTATION_TOKENIZER_CHANGED",
        "local tokenizer differs from the source contract",
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    inputs = _jsonl(parent / "input_repeat_1.jsonl")
    token_audit = token_preflight(inputs, tokenizer)
    fixed_main = json.loads((parent / "fixed_main.json").read_text())
    paths = dict(old["paths"])
    paths["protocol"] = str(ANALYSIS / "v06218_design.md")
    paths["reference_review"] = str(ANALYSIS / "v06217_reference_review.md")
    frozen = dict(old["frozen_files"]) | tokenizer_files
    dependencies = [
        Path(__file__),
        parent / "manifest.json",
        parent / "input_freeze.json",
        Path(paths["protocol"]),
        Path(paths["reference_review"]),
        ANALYSIS / "v06218_main_ownership_audit.json",
        ANALYSIS.parent.parent / "llm_judge/run_llm_judge.py",
    ]
    frozen.update({str(p.resolve()): sha256_file(p) for p in dependencies})
    for variant in VARIANTS:
        config = runner_config(variant)
        frozen.update({str(config.parent / name): digest for name, digest in _resource_hashes(config).items()})
    settings = {**old["settings"], "ray_temp_dir": "/raid/hfang/ihb/r18diag"}
    manifest = {
        "schema_version": "dedup-local-context-diagnostic-v1",
        "diagnostic_only": True,
        "eligible_for_release": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "parent_diagnostic": str(parent),
        "parent_contract_digest": old["diagnostic_contract_digest"],
        "source_root": old["source_root"],
        "baseline_root": old["baseline_root"],
        "settings": settings,
        "paths": paths,
        "source_implementation_sha256": old["source_implementation_sha256"],
        "frozen_files": frozen,
        "fixed_main_sha256": old["fixed_main_sha256"],
        "formal_pairs_per_cell": 258,
        "formal_requests": 1032,
        "preflight_pairs_per_arm": 8,
        "schedule": [
            {"repeat": r, "variant": v} for r, arms in ((1, VARIANTS), (2, tuple(reversed(VARIANTS)))) for v in arms
        ],
        "token_preflight_sha256": sha256_json(token_audit),
    }
    manifest["diagnostic_contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(root / "fixed_main.json", fixed_main)
    write_json_atomic(root / "token_preflight.json", token_audit)
    for repeat in (1, 2):
        _write_jsonl(
            root / f"input_repeat_{repeat}.jsonl",
            [presentation_record(row) for row in _jsonl(parent / f"input_repeat_{repeat}.jsonl")],
        )
    write_json_atomic(
        root / "input_freeze.json",
        {
            str(p): sha256_file(p)
            for p in [
                root / "fixed_main.json",
                root / "token_preflight.json",
                root / "input_repeat_1.jsonl",
                root / "input_repeat_2.jsonl",
            ]
        },
    )
    return manifest


def run(root: Path) -> None:
    from eval.dedup.judging.request_relay import RequestRelay
    from eval.llm_judge.run_llm_judge import ExternalJudgeRuntime

    manifest = validate_freeze(root)
    require(
        not (root / "started.json").exists(),
        "PRESENTATION_ALREADY_STARTED",
        "do not repeat partially observed experiments",
    )
    settings = manifest["settings"]
    key = os.environ.get(settings["api_key_env"], "").strip()
    require(bool(key), "PRESENTATION_CREDENTIAL_MISSING", "inference credential unavailable")
    inputs = {r: _jsonl(root / f"input_repeat_{r}.jsonl") for r in (1, 2)}
    mains = json.loads((root / "fixed_main.json").read_text())
    write_json_atomic(
        root / "started.json",
        {"at_utc": datetime.now(UTC).isoformat(), "contract": manifest["diagnostic_contract_digest"]},
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

        def runtime(variant: str) -> ExternalJudgeRuntime:
            return ExternalJudgeRuntime(
                runner_config(variant),
                endpoint=relay.endpoint,
                provider_api_key="unused",  # pragma: allowlist secret
                served_model_overrides={"judge": settings["logical_model"]},
                ray_temp_dir=settings["ray_temp_dir"],
                num_cpus=None,
            )

        with runtime("control"):
            for spec in [{"repeat": 0, "variant": v} for v in VARIANTS] + manifest["schedule"]:
                validate_freeze(root)
                repeat, variant = spec["repeat"], spec["variant"]
                cell = f"repeat_{repeat}_{variant}" if repeat else f"preflight_{variant}"
                batch = inputs[repeat] if repeat else inputs[1][: manifest["preflight_pairs_per_arm"]]
                with runtime(variant) as borrowed:
                    complete = run_cell(root / cell, batch, mains, borrowed, relay, settings)
                print(
                    json.dumps({"cell": cell, **{k: v for k, v in complete.items() if k != "artifacts"}}), flush=True
                )
                require(
                    complete["valid"] == complete["requested"] and complete["errors"] == 0,
                    "PRESENTATION_CELL_FAILED",
                    "incomplete cell stops the diagnostic",
                )
    validate_freeze(root)
    write_json_atomic(
        root / "run_complete.json",
        {
            "diagnostic_only": True,
            "eligible_for_release": False,
            "formal_requests": 1032,
            "preflight_requests": 16,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("/raid/hfang/ihb/runs/v0.6.2.17-presentation-diagnostic"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        print(prepare(root, args.parent.resolve())["diagnostic_contract_digest"])
    elif args.action == "run":
        run(root)
    else:
        require(
            args.output is not None and not args.output.exists(),
            "PRESENTATION_OUTPUT_EXISTS",
            "provide a new assessment path",
        )
        result = summarize(root, args.output)
        print(
            json.dumps(
                {
                    "supported_for_separate_fresh_local_validation": result[
                        "supported_for_separate_fresh_local_validation"
                    ],
                    "checks": result["checks"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
