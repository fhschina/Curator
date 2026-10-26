# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare, run and audit the pinned retention-policy experiment using Sarah's workflows."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from critics.models import Decision
from policy_cases import (
    CONSENSUS,
    aligned,
    blind_row,
    controlled,
    digest,
    discussion,
    load_module,
    mirrored,
    select_holdout,
)
from policy_runtime import EnvironmentAuditStage, RecordedWorkflowMixin, environment_receipt, existing_qwen_service

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL_POPULATION_SIZE = 20000
MIN_HUMAN_PAIRS = 80
MIN_DIRECTION_ACCURACY = 0.95
MIN_PAIR_ACCURACY = 0.90
MAX_UNRESOLVED_RATE = 0.05

POLICY_VERSION = "retention-policy-v1"
MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
FIELDS = ("a_can_replace_b", "b_can_replace_a")
LAYERS = {"main": "pair_semantic_judgment", "coverage": "coverage_decision", "final": "final_decision"}


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    return [
        json.loads(line) for file in files for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def exact_index(rows: list[dict], expected: set[str] | None = None) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        key = row["pair_id"]
        if key in indexed:
            msg = f"Duplicate output pair_id: {key}"
            raise ValueError(msg)
        indexed[key] = row
    if expected is not None and set(indexed) != expected:
        msg = f"Output ID mismatch: missing={sorted(expected - set(indexed))[:10]}, extra={sorted(set(indexed) - expected)[:10]}"
        raise ValueError(msg)
    return indexed


def verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, expected in files.items():
        if digest((root / relative).read_bytes()) != expected:
            msg = f"Frozen artifact changed: {relative}"
            raise ValueError(msg)


def full_population(source: Path, provenance: Path) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq

    index = read_json(source / "panel_index.json")
    if len(index) != FULL_POPULATION_SIZE or len({row["canonical_pair_id"] for row in index}) != FULL_POPULATION_SIZE:
        msg = "The full population must have exactly 20K distinct frozen pairs"
        raise ValueError(msg)
    meta = {row["canonical_pair_id"]: row for row in pq.read_table(provenance).to_pylist()}
    source_manifest = read_json(source / "manifest.json")
    rows, private = [], []
    for item in index:
        key = item["canonical_pair_id"]
        path = source / "inputs" / f"{key}.json"
        content = path.read_bytes()
        if source_manifest["artifacts"].get(str(path)) != digest(content):
            msg = f"Original population hash mismatch: {key}"
            raise ValueError(msg)
        payload = json.loads(content)["payload"]
        row = meta[key]
        a, b = str(row["presented_doc_a"]), str(row["presented_doc_b"])
        track = "keeper_removed" if row["has_track_5a"] else "cross_group_sample"
        keeper_direction = None
        if track == "keeper_removed":
            if {a, b} != {str(row["keeper_doc_id"]), str(row["removed_doc_id"])}:
                msg = f"Keeper mapping mismatch: {key}"
                raise ValueError(msg)
            keeper_direction = FIELDS[0] if a == str(row["keeper_doc_id"]) else FIELDS[1]
        rows.append({"pair_id": key, "text_a": payload["document_a"]["text"], "text_b": payload["document_b"]["text"]})
        private.append(
            {"pair_id": key, "doc_id_a": a, "doc_id_b": b, "pair_type": track, "keeper_direction": keeper_direction}
        )
    if Counter(row["pair_type"] for row in private) != {"keeper_removed": 10000, "cross_group_sample": 10000}:
        msg = "Expected 10K removals and 10K cross-group pairs"
        raise ValueError(msg)
    return rows, private


def freeze_variant(repo: Path, destination: Path) -> None:
    source = repo / "tutorials/eval/dedup"
    shutil.copytree(source / "judge_config", destination / "judge_config")
    shutil.copytree(source / "critics/prompts", destination / "critics")
    contract = (
        "\nRESPONSE FORMAT\n"
        "Return exactly one completed answer object inside a ```json Markdown code fence.\n"
        "Return actual field values, never the response schema or its properties.\n"
    )
    for path in (destination / "critics").glob("*system.jinja"):
        text = path.read_text()
        if contract.strip() not in text:
            path.write_text(text.rstrip() + "\n" + contract)
    for path in [destination / "judge_config/pair.jinja", *(destination / "critics").glob("*pair.jinja")]:
        text = path.read_text()
        if not text.startswith("Audit record:"):
            path.write_text(
                "Audit record: {{ pair_id }}\n"
                "The record identifier is for tracing only, not evidence for classification.\n\n" + text
            )


def review_page(rows: list[dict]) -> str:
    storage_key = "retention-policy-v1-" + digest(
        json.dumps([{k: row[k] for k in ("pair_id", "text_a", "text_b")} for row in rows], sort_keys=True)
    )
    blocks = []
    for row in rows:
        key = html.escape(row["pair_id"], quote=True)
        blocks.append(
            f'<section data-id="{key}" data-a="{digest(row["text_a"])}" data-b="{digest(row["text_b"])}"><h2>{key}</h2>'
            f'<div class="docs"><pre>{html.escape(row["text_a"] or "")}</pre><pre>{html.escape(row["text_b"] or "")}</pre></div>'
            '<p>A replaces B: <select class="a"><option></option><option>yes</option><option>no</option><option>unresolved</option></select> '
            'B replaces A: <select class="b"><option></option><option>yes</option><option>no</option><option>unresolved</option></select></p>'
            '<textarea placeholder="Concrete lost contribution, or unresolved policy question"></textarea></section>'
        )
    header = """<!doctype html><meta charset="utf-8"><title>Blind retention review</title>
<style>body{font:16px sans-serif;margin:2em}section{border-top:2px solid #aaa;padding:1em 0}.docs{display:grid;grid-template-columns:1fr 1fr;gap:2em}pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{width:95%;height:5em}button{position:sticky;top:0;padding:1em}</style>
<h1>Blind retention review — 100 pairs</h1><p>Left: A. Right: B. Keep A/delete B and keep B/delete A are separate questions. Read both complete texts. Use unresolved for policy questions. No model answers are included. Votes are saved in this browser and can be exported as JSON.</p>
<details><summary>Retention policy v1</summary><ul><li>Protect explanations, facts, instructions, explicit limits and obligations. For No, identify a concrete useful contribution missing from the keeper.</li><li>City/destination slots alone are harmless only when procedures, actual prices and conditions are unchanged.</li><li>Keep coherent substantive body versions in different languages, including faithful translations. Localized menus alone are harmless.</li><li>Empty zero-value carts with different currencies are equivalent when terms are unchanged.</li><li>Incidental publication dates/bylines do not block replacement of a complete article; dates inside the factual body can matter.</li><li>A URL referencing already stated license terms is harmless only if no new condition is introduced.</li><li>Assess both replacement directions independently. Use unresolved for unsettled policy significance.</li></ul></details>
<button onclick="save()">Export votes as JSON</button>"""
    script = """<script>
const STORAGE_KEY=__KEY__;
function collect(){return [...document.querySelectorAll('section')].map(s=>({pair_id:s.dataset.id,text_a_sha256:s.dataset.a,text_b_sha256:s.dataset.b,a_can_replace_b:s.querySelector('.a').value,b_can_replace_a:s.querySelector('.b').value,reason:s.querySelector('textarea').value,label_status:'human_independent'}));}
function store(){localStorage.setItem(STORAGE_KEY,JSON.stringify(collect()));}
function save(){store();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(collect(),null,2)],{type:'application/json'}));a.download='independent_votes.json';a.click();}
const old=JSON.parse(localStorage.getItem(STORAGE_KEY)||'[]');for(const r of old){const s=[...document.querySelectorAll('section')].find(x=>x.dataset.id===r.pair_id&&x.dataset.a===r.text_a_sha256&&x.dataset.b===r.text_b_sha256);if(s){s.querySelector('.a').value=r.a_can_replace_b;s.querySelector('.b').value=r.b_can_replace_a;s.querySelector('textarea').value=r.reason||'';}}
document.addEventListener('input',store);document.addEventListener('change',store);
</script>""".replace("__KEY__", json.dumps(storage_key))
    return header + "\n".join(blocks) + script


def prepare(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    if root.exists():
        msg = "prepare requires a new experiment directory"
        raise ValueError(msg)
    snapshot = read_json(args.discussion_snapshot)
    raw, labels = discussion(snapshot, args.runs_root)
    prototypes = controlled()
    gate = [
        {**row, **{k: v for k, v in label.items() if k in (*FIELDS, "label_status")}}
        for row, label in zip(raw, labels, strict=True)
        if row["pair_id"] in CONSENSUS
    ] + prototypes
    gate = [item for row in gate for item in (row, mirrored(row))]
    root.mkdir(parents=True)
    shutil.copyfile(args.discussion_snapshot, root / "discussion_snapshot.json")
    panels = {"gate": gate, "diagnostic": raw}
    panel_labels = {"gate": gate, "diagnostic": labels}
    private = {}
    if args.holdout_candidates:
        if not args.exclusions:
            msg = "Holdout preparation requires explicit historical exclusions"
            raise ValueError(msg)
        chosen = select_holdout(read_rows(args.holdout_candidates), read_rows(args.exclusions))
        panels["holdout"] = [{**row, "pair_id": f"H{i:03d}"} for i, row in enumerate(chosen, 1)]
        private["holdout"] = [{"pair_id": f"H{i:03d}", "source": row} for i, row in enumerate(chosen, 1)]
        panel_labels["holdout"] = [
            {
                "pair_id": row["pair_id"],
                **dict.fromkeys(FIELDS, ""),
                "reason": "",
                "label_status": "pending_human",
                "text_a_sha256": digest(row["text_a"]),
                "text_b_sha256": digest(row["text_b"]),
            }
            for row in panels["holdout"]
        ]
        (root / "blind_review.html").write_text(review_page(panels["holdout"]), encoding="utf-8")
        write_json(root / "independent_votes.json", panel_labels["holdout"])
    if args.full_source:
        rows, private["full20k"] = full_population(args.full_source, args.provenance)
        panels["full20k"] = rows
    for name, records in panels.items():
        prepared = aligned(
            [{"pair_id": row["pair_id"], "text_a": row["text_a"], "text_b": row["text_b"]} for row in records]
        )
        exact_index(prepared)
        write_rows(root / "panels" / name / "input.jsonl", [blind_row(row) for row in prepared])
    for name, records in panel_labels.items():
        write_json(
            root / "labels" / f"{name}.json",
            [{k: v for k, v in row.items() if k not in {"text_a", "text_b"}} for row in records],
        )
    write_json(root / "private_index.json", private)
    for variant, repo in (("baseline", args.baseline_repo), ("candidate", REPO)):
        freeze_variant(repo, root / "variants" / variant)
    write_json(
        root / "excluded_sources.json",
        {
            "candidates": str(args.holdout_candidates),
            "exclusions": str(args.exclusions),
            "candidates_sha256": digest(args.holdout_candidates.read_bytes()) if args.holdout_candidates else None,
            "exclusions_sha256": digest(args.exclusions.read_bytes()) if args.exclusions else None,
        },
    )
    files = {
        str(p.relative_to(root)): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "independent_votes.json"
    }
    manifest = {
        "policy_version": POLICY_VERSION,
        "seed": 42,
        "panels": {k: len(v) for k, v in panels.items()},
        "files": files,
        "baseline_commit": subprocess.check_output(  # noqa: S603
            [shutil.which("git"), "-C", str(args.baseline_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_commit": subprocess.check_output(  # noqa: S603
            [shutil.which("git"), "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "created_at": time.time(),
        "human_holdout_required": True,
        "shared_format_adapter": "Native Data Designer Markdown JSON contract and audit record identifiers; no baseline retention-rule changes",
        "source_snapshot_sha256": digest(args.discussion_snapshot.read_bytes()),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def decision(row: dict, layer: str) -> Decision | None:
    from critics.models import Decision

    try:
        value = (
            Decision.from_judge(row.get(LAYERS[layer]))
            if layer == "main"
            else Decision.model_validate(row.get(LAYERS[layer]))
        )
        return value if not value.inconsistencies() else None
    except ValueError:
        return None


def audit(row: dict, original: dict) -> list[str]:  # noqa: C901, PLR0912
    from critics.spans import SpanPacket

    issues = []
    for key in ("text_a", "text_b", "semantic_diff", "truncated"):
        if row.get(key) != original[key]:
            issues.append("INPUT_CHANGED:" + key)
    try:
        packet = SpanPacket.from_record(original)
        ids = set(packet.spans)
    except ValueError:
        packet = None
        ids = set()
        issues.append("INVALID_INPUT_PACKET")
    for value in (row.get("pair_semantic_judgment") or {}).values():
        if isinstance(value, dict) and set(re.findall(r"\b[ABS]\d{3}\b", str(value.get("reasoning", "")))) - ids:
            issues.append("INVALID_MAIN_CITATION")
    previous = decision(row, "main")
    judgment = row.get("pair_semantic_judgment") or {}
    if previous and packet:
        for field, side in zip(FIELDS, ("B", "A"), strict=True):
            if getattr(previous, field) != "no":
                continue
            reasons = " ".join(
                str((judgment.get(key) or {}).get("reasoning", ""))
                for key in (field, "span_" + side.lower() + "_delta")
            )
            cited = re.findall(r"\b[ABS]\d{3}\b", reasons)
            if not any(span_id in packet.spans and packet.spans[span_id].excerpt(side) for span_id in cited):
                issues.append("MISSING_MAIN_LOSS_EVIDENCE:" + field)
    for layer in ("coverage", "final"):
        current = decision(row, layer)
        if current is None:
            issues.append("INVALID_DECISION:" + layer)
        elif previous and any(getattr(previous, f) == "no" and getattr(current, f) == "yes" for f in FIELDS):
            issues.append("CRITIC_PROMOTED_NEGATIVE:" + layer)
        previous = current
    if decision(row, "main") is None:
        issues.append("INVALID_DECISION:main")
    for prefix in ("coverage", "subject"):
        reason, action = str(row.get(prefix + "_reason", "")), str(row.get(prefix + "_action", ""))
        if "INVALID" in reason or "UNVALIDATED" in action:
            issues.append(prefix.upper() + "_UNVALIDATED")
        if action.startswith("REJECT") and not row.get(prefix + "_evidence"):
            issues.append(prefix.upper() + "_MISSING_VETO_EVIDENCE")
        for evidence in row.get(prefix + "_evidence") or []:
            side = evidence["side"].lower()
            if original["text_" + side][evidence["start_char"] : evidence["end_char"]] != evidence["quote"]:
                issues.append("VETO_QUOTE_MISMATCH")
    if original["truncated"] or original["semantic_diff"]["status"] != "COMPLETE":
        final = decision(row, "final")
        if final and any(getattr(final, f) != "unresolved" for f in FIELDS):
            issues.append("CONCLUSIVE_INCOMPLETE_EVIDENCE")
    return sorted(set(issues))


def metrics(rows: list[dict], labels: list[dict], inputs: list[dict]) -> dict:
    expected = exact_index(inputs)
    indexed = exact_index(rows, set(expected))
    scored = [
        label
        for label in labels
        if label.get("label_status") in {"consensus", "controlled", "human_independent", "agent_assisted_blind"}
        and all(label.get(f) in {"yes", "no"} for f in FIELDS)
        and not expected[label["pair_id"]]["truncated"]
    ]
    issues = {key: found for key, row in indexed.items() if (found := audit(row, expected[key]))}
    layers = {}
    for layer in LAYERS:
        exact_pairs = correct = harmful = abstained = 0
        for label in scored:
            value = decision(indexed[label["pair_id"]], layer) if label["pair_id"] not in issues else None
            answers = [getattr(value, f) if value else "invalid" for f in FIELDS]
            matches = [answer == label[f] for f, answer in zip(FIELDS, answers, strict=True)]
            correct += sum(matches)
            exact_pairs += all(matches)
            harmful += sum(answer == "yes" and label[f] == "no" for f, answer in zip(FIELDS, answers, strict=True))
            abstained += any(answer in {"invalid", "unresolved"} for answer in answers)
        n = len(scored)
        layers[layer] = {
            "scored_pairs": n,
            "direction_accuracy": correct / (2 * n) if n else None,
            "pair_accuracy": exact_pairs / n if n else None,
            "unsafe_allows": harmful,
            "unresolved_pair_rate": abstained / n if n else None,
        }
    final = layers["final"]
    gate_passed = bool(scored) and not issues and final["pair_accuracy"] == 1
    holdout_passed = (
        all(label.get("label_status") == "human_independent" for label in labels)
        and len(scored) >= MIN_HUMAN_PAIRS
        and not issues
        and final["direction_accuracy"] >= MIN_DIRECTION_ACCURACY
        and final["pair_accuracy"] >= MIN_PAIR_ACCURACY
        and final["unsafe_allows"] == 0
        and final["unresolved_pair_rate"] <= MAX_UNRESOLVED_RATE
    )
    return {
        "input_pairs": len(inputs),
        "output_pairs": len(rows),
        "labels": len(labels),
        "label_sources": dict(Counter(label.get("label_status", "unknown") for label in labels)),
        "unscored_labels": len(labels) - len(scored),
        "truncated_pairs": sum(row["truncated"] for row in inputs),
        "issues": issues,
        "failure_counts": dict(Counter(issue for found in issues.values() for issue in found)),
        "skip_reasons": {
            prefix: dict(
                Counter(str(row.get(prefix + "_reason")) for row in rows if row.get(prefix + "_action") == "SKIP")
            )
            for prefix in ("coverage", "subject")
        },
        "layers": layers,
        "gate_passed": gate_passed,
        "holdout_passed": holdout_passed,
    }


def stage_effects(rows: list[dict], labels: list[dict], inputs: list[dict]) -> list[dict]:
    """Keep provisional disagreement separate from errors against confirmed labels."""
    expected, visible = exact_index(labels), exact_index(inputs)
    result = []
    for row in rows:
        key = row["pair_id"]
        label = expected.get(key, {})
        eligible = not visible[key]["truncated"] and all(label.get(f) in {"yes", "no"} for f in FIELDS)
        technical_issues = ";".join(audit(row, visible[key]))
        for layer in LAYERS:
            value = decision(row, layer)
            result.append(
                {
                    "pair_id": key,
                    "layer": layer,
                    "label_status": label.get("label_status", "unlabeled"),
                    "complete_input": not visible[key]["truncated"],
                    **{f: getattr(value, f) if value else "invalid" for f in FIELDS},
                    "reference_agrees": all(value and getattr(value, f) == label[f] for f in FIELDS)
                    if eligible
                    else None,
                    "technical_issues": technical_issues,
                }
            )
    return result


def validate_human_votes(labels: list[dict], inputs: list[dict]) -> None:
    validate_review_votes(labels, inputs, "human")


def validate_review_votes(labels: list[dict], inputs: list[dict], source: str) -> None:
    expected = exact_index(inputs)
    exact_index(labels, set(expected))
    for row in labels:
        if any(row.get(key + "_sha256") != digest(expected[row["pair_id"]][key]) for key in ("text_a", "text_b")):
            msg = "Human votes do not match the frozen texts and A/B presentation"
            raise ValueError(msg)
        status = "human_independent" if source == "human" else "agent_assisted_blind"
        if row.get("label_status") != status or any(
            row.get(field) not in {"yes", "no", "unresolved"} for field in FIELDS
        ):
            msg = f"Every case needs explicit {source} votes with the correct label provenance"
            raise ValueError(msg)
        if not str(row.get("reason", "")).strip():
            msg = "Every independent case needs a concrete reason or an unresolved policy question"
            raise ValueError(msg)


def run_config(root: Path, variant: str, run_dir: Path, replicas: int) -> Path:
    source = root / "variants" / variant
    config = yaml.safe_load((source / "judge_config/fuzzy_pair_judge.yaml").read_text())
    if config.get("filters") or any(stage.get("filters") for stage in config["execution"]["stages"]):
        msg = "Evaluation cannot enable row-dropping score filters"
        raise ValueError(msg)
    model = config["models"][0]
    model["model"] = "/raid/hfang/hf_cache/Qwen3.8-27B-FP8"
    model["served_model_name"] = "Qwen/Qwen3.8-27B-FP8"
    model["dynamo_model"]["num_replicas"] = replicas
    model["dynamo_model"]["engine_kwargs"].update(
        tensor_parallel_size=1, max_model_len=32768, max_num_seqs=8, gpu_memory_utilization=0.8
    )
    model["inference_parameters"].update(temperature=0.2, max_tokens=6144, timeout=600, max_parallel_requests=8)
    config["execution"]["stages"][0]["num_workers"] = replicas
    for stage in config["execution"]["stages"]:
        for judge in stage["judges"]:
            judge["with_trace"] = "all_messages"
            for key in ("prompt_path", "system_prompt_path"):
                judge[key] = str((source / "judge_config" / judge[key]).resolve())
    path = run_dir / "judge.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def run(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0915
    import data_designer.config as dd
    import ray

    from nemo_curator.core.client import RayClient
    from nemo_curator.eval.llm_judge.workflow import LLMJudgeWorkflow

    root = args.root.resolve()
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    if args.panel == "full20k":
        acceptance = read_json(root / "acceptance.json")
        if not acceptance.get("passed"):
            msg = "20K requires three passing candidate gates and independent human acceptance"
            raise ValueError(msg)
    label_source = getattr(args, "label_source", "human")
    labels_path = root / ("independent_votes.json" if label_source == "human" else "agent_votes.json")
    if args.panel == "holdout":
        validate_review_votes(read_json(labels_path), read_rows(root / "panels/holdout"), label_source)
    run_dir = root / "runs" / f"{args.variant}-{args.panel}-{args.repeat}"
    if run_dir.exists():
        msg = "A fresh run directory is required; never overwrite or silently reuse judgments"
        raise ValueError(msg)
    run_dir.mkdir(parents=True)
    env = environment_receipt(REPO)
    marker = read_json(Path("/raid/hfang/hf_cache/Qwen3.8-27B-FP8/.hosting-benchmark-revision.json"))
    if marker["revision"] != MODEL_REVISION:
        msg = "Qwen checkpoint revision mismatch"
        raise ValueError(msg)
    config = run_config(root, args.variant, run_dir, args.replicas)
    code_files = [
        *HERE.rglob("*.py"),
        REPO / "nemo_curator/eval/llm_judge/workflow.py",
        REPO / "nemo_curator/stages/synthetic/nemo_data_designer/data_designer.py",
    ]
    receipt = {
        "run_id": uuid.uuid4().hex,
        "variant": args.variant,
        "panel": args.panel,
        "repeat": args.repeat,
        "model_revision": MODEL_REVISION,
        "environment": env,
        "experiment_manifest_sha256": digest((root / "manifest.json").read_bytes()),
        "judge_config_sha256": digest(config.read_bytes()),
        "code_files": {str(p): digest(p.read_bytes()) for p in code_files},
        "started": time.time(),
        "status": "running",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),  # noqa: S607
        "ray_temp_dirs": {},
        "label_source": label_source,
        "labels_sha256": digest(labels_path.read_bytes()) if args.panel == "holdout" else None,
    }
    if endpoint := getattr(args, "inference_endpoint", None):
        if args.replicas != 1 or not getattr(args, "service_container", None):
            msg = "An existing service requires --service-container and currently supports one replica only"
            raise ValueError(msg)
        receipt.update(
            inference_endpoint=endpoint, serving_container=existing_qwen_service(args.service_container, endpoint)
        )
    for path in code_files:
        target = run_dir / "source" / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(run_dir / "run_manifest.json", receipt)

    class Main(RecordedWorkflowMixin, LLMJudgeWorkflow):
        pass

    class Critics(RecordedWorkflowMixin, load_module("6_run_critics.py").CriticWorkflow):
        pass

    incoming = str(root / "panels" / args.panel)
    try:
        for stage, cls in (("main", Main), ("critics", Critics)):
            output = run_dir / stage
            options = (
                {"subject": True, "critic_prompt_dir": root / "variants" / args.variant / "critics"}
                if stage == "critics"
                else {}
            )
            workflow = cls(
                judge_config=config,
                input_path=incoming,
                output_path=str(output),
                keep_failed_rows=True,
                run_config=dd.RunConfig(disable_early_shutdown=True),
                row_id_column="pair_id",
                inference_endpoint=getattr(args, "inference_endpoint", None),
                checkpoint_path=str(run_dir / (stage + "-checkpoint")),
                preprocessing_stages=[EnvironmentAuditStage(str(REPO), str(run_dir / "environment"))],
                **options,
            )
            workflow.receipt_dir = run_dir / "receipts" / stage
            runtime = root.parent.parent / "runtime"
            runtime.mkdir(exist_ok=True)
            physical = Path(tempfile.mkdtemp(prefix="ray-", dir=runtime))
            short = Path(tempfile.gettempdir()) / ("rp-" + uuid.uuid4().hex[:8])
            short.symlink_to(physical, target_is_directory=True)
            ray_temp = str(short)
            receipt["ray_temp_dirs"][stage] = ray_temp
            write_json(run_dir / "run_manifest.json", receipt)
            with RayClient(ray_temp_dir=ray_temp):
                workflow.run()
            ray.shutdown()
            inputs = read_rows(Path(incoming))
            exact_index(read_rows(output), {r["pair_id"] for r in inputs})
            incoming = str(output)
        receipt.update(completed=time.time(), status="complete")
    except (Exception, KeyboardInterrupt) as error:
        receipt.update(completed=time.time(), status="failed", error=str(error))
        raise
    finally:
        write_json(run_dir / "run_manifest.json", receipt)
    return report(args)


def report(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0915
    root = args.root.resolve()
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    run_dir = root / "runs" / f"{args.variant}-{args.panel}-{args.repeat}"
    receipt = read_json(run_dir / "run_manifest.json")
    if receipt["status"] != "complete":
        msg = "Incomplete inference run cannot pass acceptance"
        raise ValueError(msg)
    inputs = read_rows(root / "panels" / args.panel)
    labels_path = root / (
        "agent_votes.json" if receipt.get("label_source") == "agent-assisted" else "independent_votes.json"
    )
    labels = (
        read_json(labels_path)
        if args.panel == "holdout"
        else read_json(root / "labels" / f"{args.panel}.json")
        if (root / "labels" / f"{args.panel}.json").exists()
        else []
    )
    if args.panel == "holdout":
        validate_review_votes(labels, inputs, receipt.get("label_source", "human"))
        if digest(labels_path.read_bytes()) != receipt["labels_sha256"]:
            msg = "Review labels changed after the inference run was started"
            raise ValueError(msg)
    rows = read_rows(run_dir / "critics")
    result = metrics(rows, labels, inputs)
    result.update(variant=args.variant, panel=args.panel, repeat=args.repeat, model_revision=receipt["model_revision"])
    private = read_json(root / "private_index.json").get(args.panel, [])
    indexed = exact_index(rows)
    risk = Counter()
    for meta in private:
        value = decision(indexed[meta["pair_id"]], "final") if meta["pair_id"] not in result["issues"] else None
        if meta.get("keeper_direction"):
            risk["keeper_direction_" + (getattr(value, meta["keeper_direction"]) if value else "invalid")] += 1
        elif meta.get("pair_type") == "cross_group_sample":
            directions = [getattr(value, f) for f in FIELDS] if value else []
            risk[
                "cross_group_"
                + (
                    "yes"
                    if "yes" in directions
                    else "unresolved"
                    if not directions or "unresolved" in directions
                    else "no"
                )
            ] += 1
    result["diagnostic_signals"] = dict(risk)
    effects = stage_effects(rows, labels, inputs)
    result["reference_agreement_by_source"] = {
        status: {
            layer: {
                "eligible_pairs": len(
                    group := [
                        e
                        for e in effects
                        if e["label_status"] == status and e["layer"] == layer and e["reference_agrees"] is not None
                    ]
                ),
                "agreeing_pairs": sum(e["reference_agrees"] and not e["technical_issues"] for e in group),
            }
            for layer in LAYERS
        }
        for status in sorted({e["label_status"] for e in effects})
    }
    result["stage_direction_changes"] = {
        f"{before}_to_{after}": sum(
            getattr(a, field) != getattr(b, field)
            for row in rows
            if (a := decision(row, before)) and (b := decision(row, after))
            for field in FIELDS
        )
        for before, after in (("main", "coverage"), ("coverage", "final"))
    }
    result["truncated_ids"] = [row["pair_id"] for row in inputs if row["truncated"]]
    result["execution_manifest_sha256"] = digest((run_dir / "run_manifest.json").read_bytes())
    result["experiment_manifest_sha256"] = digest((root / "manifest.json").read_bytes())
    result["output_files"] = {str(p): digest(p.read_bytes()) for p in sorted((run_dir / "critics").rglob("*.jsonl"))}
    result["interpretation"] = "Purpose-selected diagnostics and automated judgments; not corpus-wide error rates."
    write_json(run_dir / "report.json", result)
    with (run_dir / "stage_effects.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    with (run_dir / "decisions.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["pair_id", "layer", *FIELDS, "issues"])
        writer.writeheader()
        for row in rows:
            for layer in LAYERS:
                value = decision(row, layer)
                writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "layer": layer,
                        **{f: getattr(value, f) if value else "invalid" for f in FIELDS},
                        "issues": ";".join(result["issues"].get(row["pair_id"], [])),
                    }
                )
    review = []
    for row in rows:
        review.append(
            "<h2>"
            + html.escape(row["pair_id"])
            + "</h2><pre>"
            + html.escape(
                json.dumps(
                    {k: v for k, v in row.items() if k not in {"text_a", "text_b", "semantic_diff"}},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + '</pre><div style="display:flex;gap:2em"><pre style="white-space:pre-wrap;width:50%">'
            + html.escape(row["text_a"] or "")
            + '</pre><pre style="white-space:pre-wrap;width:50%">'
            + html.escape(row["text_b"] or "")
            + "</pre></div>"
        )
    (run_dir / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Retention audit</h1><pre>'
        + html.escape(json.dumps(result, indent=2))
        + "</pre>"
        + "".join(review)
    )
    gates = []
    for repeat in (1, 2, 3):
        p = root / "runs" / f"candidate-gate-{repeat}" / "report.json"
        gates.append(p.exists() and read_json(p)["gate_passed"])
    hold = root / "runs/candidate-holdout-1/report.json"
    base = root / "runs/baseline-holdout-1/report.json"
    hold_pass = (
        hold.exists()
        and base.exists()
        and read_json(hold)["holdout_passed"]
        and read_json(hold)["layers"]["final"]["direction_accuracy"]
        >= read_json(base)["layers"]["final"]["direction_accuracy"]
    )
    write_json(
        root / "acceptance.json",
        {
            "passed": all(gates) and bool(hold_pass),
            "candidate_gate_repeats": gates,
            "independent_holdout": bool(hold_pass),
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--discussion-snapshot", type=Path, required=True)
    prep.add_argument("--runs-root", type=Path, required=True)
    prep.add_argument("--baseline-repo", type=Path, required=True)
    prep.add_argument("--holdout-candidates", type=Path)
    prep.add_argument("--exclusions", type=Path)
    prep.add_argument("--full-source", type=Path)
    prep.add_argument("--provenance", type=Path)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--variant", choices=("baseline", "candidate"), required=True)
        command.add_argument("--panel", choices=("gate", "diagnostic", "holdout", "full20k"), required=True)
        command.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
        command.add_argument("--replicas", type=int, choices=(1, 8), default=1)
        command.add_argument("--label-source", choices=("human", "agent-assisted"), default="human")
        command.add_argument("--inference-endpoint", help="Existing local OpenAI-compatible base URL, including /v1")
        command.add_argument("--service-container", help="Existing container to record for serving provenance")
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
