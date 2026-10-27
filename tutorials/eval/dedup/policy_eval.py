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
    MAX_VISIBLE_CHARS,
    aligned,
    blind_row,
    controlled,
    digest,
    discussion,
    load_module,
    mirrored,
    select_holdout,
)
from policy_runtime import (
    EnvironmentAuditStage,
    RecordedWorkflowMixin,
    audit_attempt,
    environment_receipt,
    existing_qwen_service,
    service_pool,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FULL_POPULATION_SIZE = 20000
POOL_REPLICAS = 8
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
        if {a, b} != {str(row["doc_id_low"]), str(row["doc_id_high"])} or a == b:
            msg = f"Presentation mapping mismatch: {key}"
            raise ValueError(msg)
        languages = {
            str(row["doc_id_low"]): row.get("language_low") or "unknown",
            str(row["doc_id_high"]): row.get("language_high") or "unknown",
        }
        track = "keeper_removed" if row["has_track_5a"] else "cross_group_sample"
        keeper_direction = None
        if track == "keeper_removed":
            if {a, b} != {str(row["keeper_doc_id"]), str(row["removed_doc_id"])}:
                msg = f"Keeper mapping mismatch: {key}"
                raise ValueError(msg)
            keeper_direction = FIELDS[0] if a == str(row["keeper_doc_id"]) else FIELDS[1]
        rows.append({"pair_id": key, "text_a": payload["document_a"]["text"], "text_b": payload["document_b"]["text"]})
        private.append(
            {
                "pair_id": key,
                "doc_id_a": a,
                "doc_id_b": b,
                "pair_type": track,
                "keeper_direction": keeper_direction,
                "language_a": languages[a],
                "language_b": languages[b],
            }
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
        "The first response line must be ```json and the last must be ```. Do not return bare JSON.\n"
    )
    for path in (destination / "critics").glob("*system.jinja"):
        text = path.read_text()
        if contract.strip() not in text:
            path.write_text(text.rstrip() + "\n" + contract)
    for path in [destination / "judge_config/pair.jinja", *(destination / "critics").glob("*pair.jinja")]:
        text = path.read_text().replace('starting with {"a_loss_span_id": ...}', "inside a Markdown JSON code fence")
        path.write_text(text)
        if not text.startswith("Audit record:"):
            path.write_text(
                "Audit record: {{ pair_id }}\n"
                "The record identifier is for tracing only, not evidence for classification.\n\n" + text
            )
        text = path.read_text()
        if contract.strip() not in text:
            path.write_text(text.rstrip() + "\n" + contract)


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


def missing_source(row: dict) -> bool:
    return any(
        row.get(key) is None or (isinstance(row.get(key), str) and not row[key].strip())
        for key in ("text_a", "text_b")
    )


def prepare_from_source(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0912, PLR0915
    source, root = args.source_experiment.resolve(), args.root.resolve()
    original = read_json(source / "manifest.json")
    verify_files(source, original["files"])
    if root.exists():
        msg = "prepare requires a new experiment directory"
        raise ValueError(msg)
    root.mkdir(parents=True)
    for name in ("panels", "labels", "input_indexes"):
        shutil.copytree(source / name, root / name)
    for name in (
        "discussion_snapshot.json",
        "private_index.json",
        "excluded_sources.json",
        "agent_votes.json",
        "independent_votes.json",
    ):
        if (source / name).exists():
            shutil.copyfile(source / name, root / name)
    if args.adjudications:
        adjudications = read_json(args.adjudications)
        inputs = exact_index(read_rows(root / "panels/holdout"))
        for entry in adjudications["entries"]:
            row = inputs[entry["pair_id"]]
            if any(entry[key + "_sha256"] != digest(row[key]) for key in ("text_a", "text_b")):
                msg = "Adjudication text or orientation mismatch"
                raise ValueError(msg)
        if adjudications["original_agent_votes_sha256"] != digest((root / "agent_votes.json").read_bytes()):
            msg = "Adjudication references different frozen agent labels"
            raise ValueError(msg)
        write_json(root / "adjudications.json", adjudications)
    else:
        msg = "Reused development panels require explicit adjudications"
        raise ValueError(msg)
    h049 = exact_index(read_rows(root / "panels/holdout"))["H049"]
    extra = [
        {**h049, "a_can_replace_b": "yes", "b_can_replace_a": "no", "label_status": "human_adjudicated"},
        {
            "pair_id": "LIST_COVERAGE",
            "text_a": "Products: X, price 71.50, in stock. Y, price 30.00, in stock. Total: 2.",
            "text_b": "Filter: X. Products: X, price 71.50, in stock. Total: 1.",
            "a_can_replace_b": "yes",
            "b_can_replace_a": "no",
            "label_status": "controlled",
        },
        {
            "pair_id": "LIST_PRICE_CONFLICT",
            "text_a": "Products: X, price 71.50, in stock. Y, price 30.00, in stock. Total: 2.",
            "text_b": "Filter: X. Products: X, price 81.50, in stock. Total: 1.",
            "a_can_replace_b": "no",
            "b_can_replace_a": "no",
            "label_status": "controlled",
        },
    ]
    presentations = [item for row in extra for item in (row, mirrored(row))]
    prepared = aligned([{k: row[k] for k in ("pair_id", "text_a", "text_b")} for row in presentations])
    write_rows(root / "panels/coverage-regression/input.jsonl", [blind_row(row) for row in prepared])
    write_json(
        root / "labels/coverage-regression.json",
        [{k: v for k, v in row.items() if k not in {"text_a", "text_b", "semantic_diff"}} for row in presentations],
    )
    diagnostic = read_rows(root / "panels/diagnostic")
    absent = sorted(
        [row for row in read_rows(root / "panels/full20k") if missing_source(row)], key=lambda row: row["pair_id"]
    )[:8]
    if len(absent) != POOL_REPLICAS:
        msg = "Preflight requires eight source-missing pairs"
        raise ValueError(msg)
    write_rows(root / "panels/preflight/input.jsonl", diagnostic + absent)
    write_json(root / "labels/preflight.json", read_json(root / "labels/diagnostic.json"))
    for variant in ("baseline", "candidate"):
        if variant == "baseline":
            shutil.copytree(source / "variants/baseline", root / "variants/baseline")
        else:
            freeze_variant(REPO, root / "variants/candidate")
    if not args.engineering_receipt:
        msg = "Reused panels require an engineering receipt"
        raise ValueError(msg)
    for panel in ("coverage-regression", "preflight"):
        write_rows(
            root / "input_indexes" / (panel + ".jsonl"),
            [
                {
                    "pair_id": row["pair_id"],
                    **{
                        key + "_sha256": digest(row[key]) if isinstance(row[key], str) else None
                        for key in ("text_a", "text_b")
                    },
                    "evidence_sha256": digest(json.dumps(row["semantic_diff"], sort_keys=True)),
                    "truncated": row["truncated"],
                }
                for row in read_rows(root / "panels" / panel)
            ],
        )
    receipt = read_json(args.engineering_receipt)
    if receipt.get("exit_code") != 0:
        msg = "Engineering checks must pass"
        raise ValueError(msg)
    for path, checksum in receipt["files"].items():
        if digest(Path(path).read_bytes()) != checksum:
            msg = "Engineering log hash mismatch"
            raise ValueError(msg)
    write_json(root / "engineering_checks.json", receipt)
    files = {str(path.relative_to(root)): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()}
    manifest = {
        **{k: v for k, v in original.items() if k != "files"},
        "files": files,
        "purpose": args.purpose,
        "source_experiment": str(source),
        "source_manifest_sha256": digest((source / "manifest.json").read_bytes()),
        "created_at": time.time(),
        "candidate_commit": subprocess.check_output(  # noqa: S603
            [shutil.which("git"), "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "panels": {**original["panels"], "coverage-regression": 6, "preflight": len(diagnostic) + len(absent)},
        "holdout_usage": "development_after_review",
        "formal_independent_human_acceptance": False,
        "semantic_prompt_revision_budget": 1,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def validation(row: dict, layer: str, issues: list[str]) -> dict:
    errors = layer_issues(issues, layer)
    value = decision(row, layer) if not errors else None
    status = (
        "invalid"
        if value is None
        else "unresolved"
        if "unresolved" in (value.a_can_replace_b, value.b_can_replace_a)
        else "valid"
    )
    return {
        "validation_status": status,
        "validation_errors": ";".join(errors),
        **{field: getattr(value, field) if value else "invalid" for field in FIELDS},
    }


def adjudicated_view(
    rows: list[dict], labels: list[dict], inputs: list[dict], adjudications: dict, issues: dict
) -> dict:
    entries = {entry["pair_id"]: entry for entry in adjudications["entries"]}
    excluded = {key for key, entry in entries.items() if entry.get("strict_accuracy_eligible") is False}
    revised = []
    for label in labels:
        if label["pair_id"] in excluded:
            continue
        entry = entries.get(label["pair_id"], {})
        revised.append({**label, **{f: entry[f] for f in FIELDS if f in entry}})
    result = metrics(rows, revised, inputs, issues)
    return {
        "reference": "post_result_adjudicated_agent_reference",
        "independent_human_accuracy": False,
        "excluded_ambiguous_ids": sorted(excluded),
        "accepted_answers": {k: entries[k]["accepted_pair_decisions"] for k in sorted(excluded)},
        "adjudicated_at": adjudications.get("updated_at_utc", adjudications.get("recorded_at_utc")),
        "layers": result["layers"],
        "all_input_failure_counts": dict(Counter(i for found in issues.values() for i in found)),
    }


def prepare(args: argparse.Namespace) -> dict:
    if getattr(args, "source_experiment", None):
        return prepare_from_source(args)
    if not all(getattr(args, key, None) for key in ("discussion_snapshot", "runs_root", "baseline_repo")):
        msg = "Fresh preparation requires discussion snapshot, runs root and baseline repo"
        raise ValueError(msg)
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
        write_rows(
            root / "input_indexes" / (name + ".jsonl"),
            [
                {
                    "pair_id": row["pair_id"],
                    **{
                        key + "_sha256": digest(row[key]) if isinstance(row[key], str) else None
                        for key in ("text_a", "text_b")
                    },
                    "evidence_sha256": digest(json.dumps(row["semantic_diff"], sort_keys=True)),
                    "truncated": row["truncated"],
                    "evidence_status": row["semantic_diff"]["status"],
                }
                for row in prepared
            ],
        )
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
            "full_source": str(args.full_source) if args.full_source else None,
            "full_source_index_sha256": digest((args.full_source / "panel_index.json").read_bytes())
            if args.full_source
            else None,
            "full_source_manifest_sha256": digest((args.full_source / "manifest.json").read_bytes())
            if args.full_source
            else None,
            "provenance": str(args.provenance) if args.provenance else None,
            "provenance_sha256": digest(args.provenance.read_bytes()) if args.provenance else None,
        },
    )
    files = {
        str(p.relative_to(root)): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "independent_votes.json"
    }
    manifest = {
        "policy_version": POLICY_VERSION,
        "purpose": getattr(args, "purpose", "acceptance"),
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
        packet = None if missing_source(original) else SpanPacket.from_record(original)
        ids = set(packet.spans) if packet else set()
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
        if reason.startswith("INVALID_CRITIC_OUTPUT") or "UNVALIDATED" in action:
            issues.append(prefix.upper() + "_UNVALIDATED")
        if action.startswith("REJECT") and not row.get(prefix + "_evidence"):
            issues.append(prefix.upper() + "_MISSING_VETO_EVIDENCE")
        for evidence in row.get(prefix + "_evidence") or []:
            side = evidence["side"].lower()
            if original["text_" + side][evidence["start_char"] : evidence["end_char"]] != evidence["quote"]:
                issues.append(prefix.upper() + "_VETO_QUOTE_MISMATCH")
    if original["truncated"] or original["semantic_diff"]["status"] != "COMPLETE":
        for layer in LAYERS:
            current = decision(row, layer)
            if current and any(getattr(current, f) != "unresolved" for f in FIELDS):
                issues.append(
                    "CONCLUSIVE_INCOMPLETE_EVIDENCE" if layer == "final" else "CONCLUSIVE_INCOMPLETE_EVIDENCE:" + layer
                )
    return sorted(set(issues))


def layer_issues(issues: list[str], layer: str) -> list[str]:
    if layer == "main":
        return [
            issue
            for issue in issues
            if not (
                issue.startswith(("COVERAGE_", "SUBJECT_", "CRITIC_PROMOTED_NEGATIVE:"))
                or issue in {"INVALID_DECISION:coverage", "INVALID_DECISION:final", "CONCLUSIVE_INCOMPLETE_EVIDENCE"}
            )
        ]
    if layer == "coverage":
        return [
            issue
            for issue in issues
            if not (
                issue.startswith("SUBJECT_")
                or issue
                in {
                    "INVALID_DECISION:final",
                    "CRITIC_PROMOTED_NEGATIVE:final",
                    "CONCLUSIVE_INCOMPLETE_EVIDENCE",
                }
            )
        ]
    return issues


def integrity_failures(issues: dict[str, list[str]], *, model_errors_fatal: bool = True) -> dict[str, list[str]]:
    """Source corruption, missing audit traces and unauthorized promotions stop a run."""
    return {
        key: fatal
        for key, found in issues.items()
        if (
            fatal := [
                issue
                for issue in found
                if issue.startswith(("INPUT_CHANGED:", "CRITIC_PROMOTED_NEGATIVE:"))
                or issue.endswith("MISSING_INFERENCE_RECEIPT")
                or issue == "INVALID_INPUT_PACKET"
                or (model_errors_fatal and issue.startswith("CONCLUSIVE_INCOMPLETE_EVIDENCE"))
            ]
        )
    }


def raw_directions(row: dict, layer: str) -> dict:
    payload = row.get(LAYERS[layer])
    if not isinstance(payload, dict):
        return {}
    return {field: value.get("score") if isinstance(value := payload.get(field), dict) else value for field in FIELDS}


def metrics(
    rows: list[dict], labels: list[dict], inputs: list[dict], extra_issues: dict[str, list[str]] | None = None
) -> dict:
    expected = exact_index(inputs)
    indexed = exact_index(rows, set(expected))
    scored = [
        label
        for label in labels
        if label.get("label_status")
        in {"consensus", "controlled", "human_adjudicated", "human_independent", "agent_assisted_blind"}
        and all(label.get(f) in {"yes", "no"} for f in FIELDS)
        and not expected[label["pair_id"]]["truncated"]
    ]
    issues = {key: found for key, row in indexed.items() if (found := audit(row, expected[key]))}
    for key, found in (extra_issues or {}).items():
        issues[key] = sorted(set(issues.get(key, []) + found))
    layers = {}
    for layer in LAYERS:
        exact_pairs = correct = harmful = abstained = failed = 0
        for label in scored:
            raw_answers = raw_directions(indexed[label["pair_id"]], layer)
            value = (
                decision(indexed[label["pair_id"]], layer)
                if not layer_issues(issues.get(label["pair_id"], []), layer)
                else None
            )
            answers = [getattr(value, f) if value else "invalid" for f in FIELDS]
            matches = [answer == label[f] for f, answer in zip(FIELDS, answers, strict=True)]
            correct += sum(matches)
            exact_pairs += all(matches)
            harmful += sum(str(raw_answers.get(f, "")).casefold() == "yes" and label[f] == "no" for f in FIELDS)
            failed += value is None
            abstained += value is not None and any(answer == "unresolved" for answer in answers)
        n = len(scored)
        layers[layer] = {
            "scored_pairs": n,
            "direction_accuracy": correct / (2 * n) if n else None,
            "pair_accuracy": exact_pairs / n if n else None,
            "unsafe_allows": harmful,
            "unresolved_pair_rate": abstained / n if n else None,
            "invalid_pair_rate": failed / n if n else None,
        }
    final = layers["final"]
    gate_passed = bool(scored) and not issues and all(value["pair_accuracy"] == 1 for value in layers.values())
    holdout_passed = (
        all(label.get("label_status") == "human_independent" for label in labels)
        and len(scored) >= MIN_HUMAN_PAIRS
        and not integrity_failures(issues)
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
        "unsafe_allow_definition": "Raw yes in a reference-no direction, including judgments with invalid schema or evidence; validity is reported separately.",
        "unscored_labels": len(labels) - len(scored),
        "truncated_pairs": sum(row["truncated"] for row in inputs),
        "issues": issues,
        "integrity_failures": integrity_failures(issues),
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


def diagnostic_signals(rows: list[dict], private: list[dict], issues: dict[str, list[str]]) -> dict[str, int]:
    indexed = exact_index(rows)
    risk = Counter()
    for meta in private:
        value = decision(indexed[meta["pair_id"]], "final") if meta["pair_id"] not in issues else None
        if meta.get("keeper_direction"):
            risk["keeper_direction_" + (getattr(value, meta["keeper_direction"]) if value else "invalid")] += 1
        elif meta.get("pair_type") == "cross_group_sample":
            directions = [getattr(value, field) for field in FIELDS] if value else []
            category = (
                "invalid"
                if value is None
                else "yes"
                if "yes" in directions
                else "unresolved"
                if "unresolved" in directions
                else "no"
            )
            risk["cross_group_" + category] += 1
    return dict(risk)


def stage_effects(
    rows: list[dict], labels: list[dict], inputs: list[dict], extra_issues: dict[str, list[str]] | None = None
) -> list[dict]:
    """Keep provisional disagreement separate from errors against confirmed labels."""
    expected, visible = exact_index(labels), exact_index(inputs)
    result = []
    for row in rows:
        key = row["pair_id"]
        label = expected.get(key, {})
        eligible = not visible[key]["truncated"] and all(label.get(f) in {"yes", "no"} for f in FIELDS)
        technical_issues = sorted(set(audit(row, visible[key]) + (extra_issues or {}).get(key, [])))
        for layer in LAYERS:
            value = decision(row, layer) if not layer_issues(technical_issues, layer) else None
            result.append(
                {
                    "pair_id": key,
                    "layer": layer,
                    "label_status": label.get("label_status", "unlabeled"),
                    "complete_input": not visible[key]["truncated"],
                    **validation(row, layer, technical_issues),
                    "reference_agrees": all(value and getattr(value, f) == label[f] for f in FIELDS)
                    if eligible
                    else None,
                    "technical_issues": ";".join(layer_issues(technical_issues, layer)),
                }
            )
    return result


def inference_attempts(run_dir: Path) -> tuple[list[dict], dict[str, list[str]]]:
    attempts, issues = [], {}
    for path in sorted((run_dir / "receipts").rglob("*.json")):
        try:
            receipt = read_json(path)
        except (OSError, ValueError) as error:
            attempts.append(
                {
                    "pair_id": None,
                    "column": "unknown",
                    "status": "malformed_receipt",
                    "error": str(error),
                    "receipt": str(path),
                }
            )
            continue
        if receipt.get("method") != "POST":
            continue
        result = {
            **audit_attempt(receipt),
            "stage": path.relative_to(run_dir / "receipts").parts[0],
            "receipt": str(path),
        }
        attempts.append(result)
        if result["status"] != "valid" and result["pair_id"]:
            prefix = {"CoverageReview": "COVERAGE", "SubjectReview": "SUBJECT", "SubjectVerification": "SUBJECT"}.get(
                result["column"], "MAIN"
            )
            issues.setdefault(result["pair_id"], []).append(prefix + "_ATTEMPT_" + result["status"].upper())
    return attempts, issues


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


def dispatch_inputs(root: Path, panel: str, run_dir: Path, replicas: int) -> str:
    """Give native reader tasks separate shards without changing frozen pair identity."""
    source = root / "panels" / panel
    if replicas == 1:
        return str(source)
    rows = read_rows(source)
    indexed = exact_index(rows)
    destination = run_dir / "input_dispatch"
    for index in range(replicas):
        write_rows(destination / f"part-{index:02d}.jsonl", rows[index::replicas])
    dispatched = exact_index(read_rows(destination), set(indexed))
    if dispatched != indexed:
        msg = "Dispatch sharding changed frozen pair content"
        raise ValueError(msg)
    return str(destination)


def run(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0912, PLR0915
    import data_designer.config as dd
    import ray

    from nemo_curator.core.client import RayClient
    from nemo_curator.eval.llm_judge.workflow import LLMJudgeWorkflow

    root = args.root.resolve()
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    if args.panel == "full20k":
        diagnostic_mode = manifest.get("purpose", "acceptance") == "diagnostic"
        readiness = diagnostic_readiness(root) if diagnostic_mode else acceptance_state(root)
        write_json(root / ("diagnostic_readiness.json" if diagnostic_mode else "acceptance.json"), readiness)
        if not readiness.get("passed"):
            raise ValueError("20K prerequisites failed: " + json.dumps(readiness))
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
        "git_commit": subprocess.check_output([shutil.which("git"), "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),  # noqa: S603
        "ray_temp_dirs": {},
        "label_source": label_source,
        "labels_sha256": digest(labels_path.read_bytes()) if args.panel == "holdout" else None,
    }
    verified_services = []
    if getattr(args, "service_manifest", None):
        if getattr(args, "inference_endpoint", None) or getattr(args, "service_container", None):
            msg = "Use either a service manifest or a legacy single service"
            raise ValueError(msg)
        verified_services = service_pool(args.service_manifest, args.replicas)
        receipt["serving_pool"] = verified_services
        receipt["service_manifest_sha256"] = digest(args.service_manifest.read_bytes())
        args.inference_endpoint = verified_services[0]["endpoint"]
    elif endpoint := getattr(args, "inference_endpoint", None):
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

    incoming = dispatch_inputs(root, args.panel, run_dir, args.replicas)
    receipt["input_dispatch_files"] = {
        str(path): digest(path.read_bytes()) for path in sorted(Path(incoming).glob("*.jsonl"))
    }
    write_json(run_dir / "run_manifest.json", receipt)
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
                files_per_partition=1,
                keep_failed_rows=True,
                run_config=dd.RunConfig(disable_early_shutdown=True),
                row_id_column="pair_id",
                inference_endpoint=getattr(args, "inference_endpoint", None),
                checkpoint_path=str(run_dir / (stage + "-checkpoint")),
                preprocessing_stages=[EnvironmentAuditStage(str(REPO), str(run_dir / "environment"))],
                **options,
            )
            workflow.receipt_dir = run_dir / "receipts" / stage
            workflow.service_endpoints = [item["endpoint"] for item in verified_services]
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
    attempts, attempt_issues = inference_attempts(run_dir)
    seen = {(a["pair_id"], a["column"]) for a in attempts}
    for row in rows:
        for prefix, column, required in (
            ("MAIN", "JudgeStructuredOutput", True),
            ("COVERAGE", "CoverageReview", row.get("coverage_should_run")),
            ("SUBJECT", "SubjectReview", row.get("subject_should_run")),
            ("SUBJECT", "SubjectVerification", row.get("subject_verifier_should_run")),
        ):
            if required and (row["pair_id"], column) not in seen:
                attempt_issues.setdefault(row["pair_id"], []).append(prefix + "_MISSING_INFERENCE_RECEIPT")
    result = metrics(rows, labels, inputs, attempt_issues)
    result["integrity_failures"] = integrity_failures(
        result["issues"], model_errors_fatal=manifest.get("purpose", "acceptance") != "diagnostic"
    )
    write_rows(run_dir / "attempt_audit.jsonl", attempts)
    result["attempt_status_counts"] = dict(Counter(a["status"] for a in attempts))
    result["attempt_retries"] = sum(
        max(0, count - 1) for count in Counter((a["pair_id"], a["column"]) for a in attempts).values()
    )
    expected_ids = {r["pair_id"] for r in inputs}
    result["unmapped_attempts"] = sum(a["pair_id"] not in expected_ids for a in attempts)
    if result["unmapped_attempts"]:
        result.update(gate_passed=False, holdout_passed=False)
    result.update(variant=args.variant, panel=args.panel, repeat=args.repeat, model_revision=receipt["model_revision"])
    private = [
        {**meta.get("source", {}), **meta} for meta in read_json(root / "private_index.json").get(args.panel, [])
    ]
    result["diagnostic_signals"] = diagnostic_signals(rows, private, result["issues"])
    result["language_distribution"] = dict(
        Counter(str(meta.get("language_" + side) or "unknown") for meta in private for side in ("a", "b"))
    )
    lengths = [len(row[field]) for row in inputs for field in ("text_a", "text_b") if isinstance(row[field], str)]
    result["raw_document_length_chars"] = {
        "documents": len(inputs) * 2,
        "missing_documents": len(inputs) * 2 - len(lengths),
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "visible_window_chars": MAX_VISIBLE_CHARS,
        "over_window_documents": sum(length > MAX_VISIBLE_CHARS for length in lengths),
    }
    effects = stage_effects(rows, labels, inputs, attempt_issues)
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
    result["audit_code_sha256"] = digest(Path(__file__).read_bytes())
    result["experiment_manifest_sha256"] = digest((root / "manifest.json").read_bytes())
    result["output_files"] = {str(p): digest(p.read_bytes()) for p in sorted((run_dir / "critics").rglob("*.jsonl"))}
    result["purpose"] = manifest.get("purpose", "acceptance")
    result["missing_source_pairs"] = sum(missing_source(row) for row in inputs)
    result["layer_answer_counts"] = {
        layer: {
            "raw_yes_directions": sum(
                raw_directions(row, layer).get(field) == "yes" for row in rows for field in FIELDS
            ),
            "validated_yes_directions": sum(
                validation(row, layer, result["issues"].get(row["pair_id"], []))[field] == "yes"
                for row in rows
                for field in FIELDS
            ),
        }
        for layer in LAYERS
    }
    result["all_input_validation"] = {
        layer: dict(
            Counter(
                validation(row, layer, result["issues"].get(row["pair_id"], []))["validation_status"] for row in rows
            )
        )
        for layer in LAYERS
    }
    if (root / "adjudications.json").exists() and args.panel == "holdout":
        result["adjudicated_view"] = adjudicated_view(
            rows, labels, inputs, read_json(root / "adjudications.json"), result["issues"]
        )
    pool_attempts = [read_json(path) for path in (run_dir / "receipts").rglob("http/*.json")]
    result["replica_requests"] = dict(
        Counter(a.get("upstream", "unknown") for a in pool_attempts if a.get("method") == "POST")
    )
    result["replica_peak_concurrency"] = max((a.get("replica_active_requests", 0) for a in pool_attempts), default=0)
    result["interpretation"] = "Purpose-selected diagnostics and automated judgments; not corpus-wide error rates."
    write_json(run_dir / "report.json", result)
    with (run_dir / "stage_effects.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    with (run_dir / "decisions.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=["pair_id", "layer", *FIELDS, "validation_status", "validation_errors", "issues"]
        )
        writer.writeheader()
        for row in rows:
            for layer in LAYERS:
                writer.writerow(
                    {
                        "pair_id": row["pair_id"],
                        "layer": layer,
                        **validation(row, layer, result["issues"].get(row["pair_id"], [])),
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
    write_json(root / "acceptance.json", acceptance_state(root))
    return result


def checked_report(root: Path, name: str, *, require_independent: bool = True) -> dict:
    run_dir = root / "runs" / name
    result = read_json(run_dir / "report.json")
    receipt_path = run_dir / "run_manifest.json"
    receipt = read_json(receipt_path)
    if (
        receipt["status"] != "complete"
        or result["execution_manifest_sha256"] != digest(receipt_path.read_bytes())
        or result["experiment_manifest_sha256"] != digest((root / "manifest.json").read_bytes())
        or receipt["experiment_manifest_sha256"] != digest((root / "manifest.json").read_bytes())
        or result["audit_code_sha256"] != digest(Path(__file__).read_bytes())
    ):
        msg = "Report provenance changed; regenerate the audit"
        raise ValueError(msg)
    output_files = {str(p): digest(p.read_bytes()) for p in sorted((run_dir / "critics").rglob("*.jsonl"))}
    if not output_files or result["output_files"] != output_files:
        msg = "Audited outputs changed"
        raise ValueError(msg)
    inputs = read_rows(root / "panels" / result["panel"])
    outputs = read_rows(run_dir / "critics")
    exact_index(outputs, set(exact_index(inputs)))
    if result["input_pairs"] != len(inputs) or result["output_pairs"] != len(outputs):
        msg = "Audited row counts changed"
        raise ValueError(msg)
    if result["panel"] == "holdout":
        source = receipt.get("label_source", "human")
        labels_path = root / ("independent_votes.json" if source == "human" else "agent_votes.json")
        if (require_independent and source != "human") or receipt["labels_sha256"] != digest(labels_path.read_bytes()):
            msg = "Independent human label provenance required"
            raise ValueError(msg)
        validate_review_votes(read_json(labels_path), inputs, source)
    return result


def acceptance_state(root: Path) -> dict:
    """Bind acceptance to the frozen inputs, labels, execution and audited outputs."""
    problems = {}

    def checked(name: str) -> dict:
        run_dir = root / "runs" / name
        if not (run_dir / "report.json").exists():
            problems[name] = "missing_report"
            return {}
        try:
            result = checked_report(root, name)
        except (OSError, ValueError, KeyError) as exc:
            problems[name] = str(exc)
            return {}
        return result

    gates = [checked(f"candidate-gate-{repeat}").get("gate_passed", False) for repeat in (1, 2, 3)]
    diagnostic = checked("candidate-diagnostic-1")
    diagnostic_pass = bool(diagnostic) and (
        not diagnostic["integrity_failures"]
        and not diagnostic.get("unmapped_attempts")
        and all(layer["scored_pairs"] and layer["pair_accuracy"] == 1 for layer in diagnostic["layers"].values())
    )
    hold, base = checked("candidate-holdout-1"), checked("baseline-holdout-1")
    hold_pass = bool(
        hold
        and base
        and hold["holdout_passed"]
        and hold["layers"]["final"]["direction_accuracy"] >= base["layers"]["final"]["direction_accuracy"]
    )
    return {
        "passed": all(gates) and diagnostic_pass and hold_pass,
        "candidate_gate_repeats": gates,
        "candidate_diagnostic": diagnostic_pass,
        "candidate_diagnostic_all_rows_valid": bool(diagnostic) and not diagnostic["issues"],
        "independent_holdout": hold_pass,
        "provenance_problems": problems,
    }


def diagnostic_readiness(root: Path) -> dict:
    """Allow diagnostic scale-up without changing the independent acceptance gate."""
    checks, problems = {}, {}
    verify_files(root, read_json(root / "manifest.json")["files"])
    for name in [
        *(f"candidate-gate-{i}" for i in (1, 2, 3)),
        "candidate-coverage-regression-1",
        "candidate-diagnostic-1",
        "baseline-diagnostic-1",
        "candidate-holdout-1",
        "baseline-holdout-1",
        "candidate-preflight-1",
        "baseline-preflight-1",
    ]:
        try:
            report = checked_report(root, name, require_independent=False)
            good = not report["integrity_failures"] and not report.get("unmapped_attempts")
            if "-gate-" in name:
                good = good and report["gate_passed"]
            if name == "candidate-diagnostic-1":
                good = good and all(
                    layer["scored_pairs"] == len(CONSENSUS) and layer["pair_accuracy"] == 1
                    for layer in report["layers"].values()
                )
            if "-coverage-regression-" in name:
                rows = read_rows(root / "runs" / name / "critics")
                good = good and all(
                    validation(row, layer, report["issues"].get(row["pair_id"], []))[f] == "no"
                    for row in rows
                    if row["pair_id"].startswith("LIST_PRICE_CONFLICT")
                    for layer in LAYERS
                    for f in FIELDS
                )
            if "-preflight-" in name:
                good = (
                    good
                    and len(report["replica_requests"]) == POOL_REPLICAS
                    and report["replica_peak_concurrency"] <= POOL_REPLICAS
                )
            checks[name] = good
        except (OSError, ValueError, KeyError) as error:
            checks[name] = False
            problems[name] = str(error)
    receipt = read_json(root / "engineering_checks.json")
    checks["engineering"] = receipt.get("exit_code") == 0 and all(
        digest(Path(path).read_bytes()) == sha for path, sha in receipt["files"].items()
    )
    return {
        "purpose": "diagnostic",
        "passed": all(checks.values()),
        "checks": checks,
        "problems": problems,
        "independent_human_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--discussion-snapshot", type=Path)
    prep.add_argument("--runs-root", type=Path)
    prep.add_argument("--baseline-repo", type=Path)
    prep.add_argument("--purpose", choices=("acceptance", "diagnostic"), default="acceptance")
    prep.add_argument("--source-experiment", type=Path)
    prep.add_argument("--adjudications", type=Path)
    prep.add_argument("--engineering-receipt", type=Path)
    prep.add_argument("--holdout-candidates", type=Path)
    prep.add_argument("--exclusions", type=Path)
    prep.add_argument("--full-source", type=Path)
    prep.add_argument("--provenance", type=Path)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--variant", choices=("baseline", "candidate"), required=True)
        command.add_argument(
            "--panel",
            choices=("gate", "coverage-regression", "diagnostic", "holdout", "preflight", "full20k"),
            required=True,
        )
        command.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
        command.add_argument("--replicas", type=int, choices=(1, 8), default=1)
        command.add_argument("--label-source", choices=("human", "agent-assisted"), default="human")
        command.add_argument("--inference-endpoint", help="Existing local OpenAI-compatible base URL, including /v1")
        command.add_argument("--service-manifest", type=Path)
        command.add_argument("--service-container", help="Existing container to record for serving provenance")
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
