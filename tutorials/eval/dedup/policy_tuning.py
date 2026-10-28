# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze and compare bounded prompt searches through the native policy evaluator."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from statistics import mean

from policy_cases import aligned, blind_row, digest, mirrored
from policy_eval import (
    FIELDS,
    LAYERS,
    REPO,
    exact_index,
    freeze_variant,
    prepare_from_source,
    raw_directions,
    read_json,
    read_rows,
    validation,
    verify_files,
    write_json,
    write_rows,
)

BASELINE = "6715574e4e553f7b36f263d755f6cebe198e7b11"
LATEST_CASE_COUNT = 9
PERSONAL_PAIR_COUNT = 120
REPEATS = 3
TARGET_DIRECTION_AGREEMENT = Fraction(9, 10)
TARGET_PAIR_AGREEMENT = Fraction(17, 20)
MILESTONE_DIRECTION_AGREEMENT = Fraction(17, 20)
MILESTONE_PAIR_AGREEMENT = Fraction(4, 5)


def controls() -> list[dict]:
    """Policy-derived examples; these are controlled expectations, not human votes."""
    body = "The library opens at 10:00 on Tuesday. Entry costs 8 euros."
    categories = (
        "Essential cookies enable requested communication. Preferences store your settings. "
        "Statistics use anonymous aggregate measurements. Marketing builds advertising profiles."
    )
    examples = [
        (
            "citation_slot",
            "Reuse is allowed with a direct source link to Harbor City's website in the second paragraph.",
            "Reuse is allowed with a direct source link to Meadow City's website in the second paragraph.",
            "yes",
            "yes",
            "R11: unchanged citation terms with only the city-site target changed",
        ),
        (
            "marketing_slot",
            "Carved panels in Harbor. Our decorative woodwork brings style to your home. Contact our team.",
            "Wood trim in Meadow. Our decorative woodwork brings style to your home. Contact our team.",
            "yes",
            "yes",
            "R18: category/location slots without distinct product specifications",
        ),
        (
            "actual_product",
            "Panel model K2. Oak, 2 meters long, price 25 euros.",
            "Panel model M4. Pine, 3 meters long, price 40 euros.",
            "no",
            "no",
            "Existing policy: concrete products and specifications remain material",
        ),
        (
            "submission_endpoint",
            "Send your workshop report to harbor-posts@example.org. Attach pictures; it publishes immediately.",
            "Send your workshop report to meadow-posts@example.org. Attach pictures; it publishes immediately.",
            "no",
            "no",
            "D011: actual submission destinations are operational content",
        ),
        (
            "localized_title",
            "Library information\n" + body,
            "Informations bibliothèque\n" + body,
            "yes",
            "yes",
            "D059: localized heading with identical substantive body",
        ),
        (
            "localized_auth",
            "Email address. A link to set your password will be emailed to you.",
            "Adresse e-mail. Un lien pour définir votre mot de passe vous sera envoyé par e-mail.",
            "yes",
            "yes",
            "D058: equivalent short authentication-interface localization",
        ),
        (
            "body_language",
            body,
            "La bibliothèque ouvre à 10h00 le mardi. L'entrée coûte 8 euros.",
            "no",
            "no",
            "D051: distinct-language factual body prose retains linguistic contribution",
        ),
        (
            "auth_procedure",
            "Enter a new password directly here to activate your account.",
            "A link to choose a new password will be sent by email to activate your account.",
            "no",
            "no",
            "Existing policy: a changed actual authentication procedure remains material",
        ),
        (
            "cookie_intro",
            "We use cookies to improve your experience.\n" + categories,
            "We use device storage to optimize this site. Consent permits browsing IDs; declining may affect functions.\n"
            + categories,
            "yes",
            "yes",
            "D134/D135: generic introductions over fully shared category definitions",
        ),
        (
            "cookie_name",
            "This site uses cookies to provide services, personalize ads and analyze visits. Usage information is shared.",
            "This site uses cookies from Provider Q to provide services, personalize ads and analyze visits. Usage information is shared with Provider Q.",
            "yes",
            "yes",
            "R33: provider-name slot in otherwise equivalent generic cookie messaging",
        ),
        (
            "cookie_repetition",
            categories + "\nAnalytics run only on our servers; no data leaves our control.",
            categories + "\nEssential cookies enable requested communication.",
            "yes",
            "no",
            "R14: repeated generic definition adds no loss; concrete processing restriction does",
        ),
        (
            "cookie_specific_use",
            categories + "\nWe transmit precise GPS coordinates to advertisers for 90 days.",
            categories,
            "yes",
            "no",
            "Existing policy: specific data purpose, recipient behavior and retention period matter",
        ),
        (
            "cookie_restriction",
            categories + "\nAnalytics data never leaves our servers.",
            categories + "\nAnalytics data is sent to external advertising servers.",
            "no",
            "no",
            "Existing policy: incompatible concrete data-handling restrictions remain material",
        ),
    ]
    return [
        {
            "pair_id": "T_" + name,
            "text_a": a,
            "text_b": b,
            FIELDS[0]: ab,
            FIELDS[1]: ba,
            "label_status": "controlled",
            "basis": basis,
        }
        for name, a, b, ab, ba, basis in examples
    ]


def seal(root: Path, manifest: dict) -> None:
    manifest["files"] = {
        str(path.relative_to(root)): digest(path.read_bytes())
        for directory in ("panels", "labels", "input_indexes", "variants")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
    }
    for name in ("evaluation.json", "adjudications.json", "agent_votes.json", "engineering_checks.json"):
        manifest["files"][name] = digest((root / name).read_bytes())
    write_json(root / "manifest.json", manifest)


def prepare(args: argparse.Namespace) -> None:
    session, source = args.session.resolve(), args.source.resolve()
    root = session / "baseline"
    manifest = prepare_from_source(
        argparse.Namespace(
            root=root,
            source_experiment=source,
            adjudications=source / "adjudications.json",
            engineering_receipt=args.engineering_receipt,
            purpose="diagnostic",
        )
    )
    shutil.rmtree(root / "variants/baseline")
    shutil.copytree(source / "variants/candidate", root / "variants/baseline")
    shutil.rmtree(root / "variants/candidate")
    shutil.copytree(root / "variants/baseline", root / "variants/candidate")
    cases = read_json(args.review / "cases.json")["cases"]
    decisions = {row["case_id"]: row for row in read_json(args.review / "adjudications.json")["decisions"]}
    if len(decisions) != LATEST_CASE_COUNT or "D121" in decisions:
        msg = "Expected nine confirmed cases, with D121 still pending"
        raise ValueError(msg)
    selected = []
    for case in cases:
        if case["case_id"] not in decisions:
            continue
        label = decisions[case["case_id"]]
        selected.append(
            {
                "pair_id": case["case_id"],
                "text_a": case["text_a"],
                "text_b": case["text_b"],
                **{field: label[field] for field in FIELDS},
                "label_status": "human_adjudicated",
                "source": label,
            }
        )
    presentations = [item for row in selected + controls() for item in (row, mirrored(row))]
    prepared = aligned([{k: row[k] for k in ("pair_id", "text_a", "text_b")} for row in presentations])
    write_rows(root / "panels/preflight/input.jsonl", [blind_row(row) for row in prepared])
    write_json(
        root / "labels/preflight.json",
        [{k: v for k, v in row.items() if k not in {"text_a", "text_b"}} for row in presentations],
    )
    write_rows(
        root / "input_indexes/preflight.jsonl",
        [
            {
                "pair_id": row["pair_id"],
                **{k + "_sha256": digest(row[k]) for k in ("text_a", "text_b")},
                "evidence_sha256": digest(json.dumps(row["semantic_diff"], sort_keys=True)),
                "truncated": row["truncated"],
            }
            for row in prepared
        ],
    )
    diagnostic = read_json(root / "labels/diagnostic.json")
    inputs = exact_index(read_rows(root / "panels/diagnostic"))
    fixed = []
    latest = {row["pair_id"]: row for row in decisions.values() if row["pair_id"].startswith("D")}
    for row in diagnostic:
        if row["label_status"] != "provisional" or row["pair_id"] == "D121" or inputs[row["pair_id"]]["truncated"]:
            continue
        fixed.append({**row, **{field: latest.get(row["pair_id"], row)[field] for field in FIELDS}})
    if len(fixed) != PERSONAL_PAIR_COUNT:
        msg = "The fixed personal reference must contain 120 complete pairs"
        raise ValueError(msg)
    evaluation = {
        "direction_order": list(FIELDS),
        "fixed_personal_reference": fixed,
        "latest_user_decisions": list(decisions.values()),
        "D121": "pending; not a confirmed label and excluded from fixed reference",
        "consensus_ids": [row["pair_id"] for row in diagnostic if row["label_status"] == "consensus"],
        "discussion_ids": [
            row["pair_id"]
            for row in diagnostic
            if row["label_status"] == "needs_discussion" and not inputs[row["pair_id"]]["truncated"]
        ],
        "truncated_ids": [key for key, row in inputs.items() if row["truncated"]],
        "hard_groups": {
            "original36": {"panel": "gate", "ids": list(exact_index(read_rows(root / "panels/gate")))},
            "extra6": {
                "panel": "coverage-regression",
                "ids": list(exact_index(read_rows(root / "panels/coverage-regression"))),
            },
            "latest18": {
                "panel": "preflight",
                "ids": [row["pair_id"] for row in presentations if not row["pair_id"].startswith("T_")],
            },
            "controlled26": {
                "panel": "preflight",
                "ids": [row["pair_id"] for row in presentations if row["pair_id"].startswith("T_")],
            },
        },
        "reference_kind": "Development reference agreement, not independent human accuracy or corpus deletion risk",
    }
    write_json(root / "evaluation.json", evaluation)
    manifest.update(
        baseline_commit=BASELINE,
        policy_version="retention-policy-v3-overnight",
        semantic_prompt_revision_budget=None,
        panels={**manifest["panels"], "preflight": len(presentations)},
        frozen_at=time.time(),
    )
    seal(root, manifest)
    print(
        json.dumps(
            {
                "root": str(root),
                "hard_groups": {k: len(v["ids"]) for k, v in evaluation["hard_groups"].items()},
                "personal_pairs": len(fixed),
            }
        )
    )


def candidate(args: argparse.Namespace) -> None:
    session = args.session.resolve()
    state = read_json(session / "SESSION_STATUS.json")
    if time.time() >= state["semantic_search_deadline"]:
        msg = "Semantic search deadline reached; reserve the last hour for delivery"
        raise ValueError(msg)
    if args.number < 1:
        msg = "Semantic candidate number must be positive"
        raise ValueError(msg)
    revision = getattr(args, "revision", None)
    if revision and (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]*", revision)
        or args.number not in {item["number"] for item in state["semantic_candidates"]}
    ):
        msg = "An engineering revision needs an existing semantic candidate and a simple unique suffix"
        raise ValueError(msg)
    name = f"candidate-{args.number:02d}" + (f"-{revision}" if revision else "")
    source, root = session / "baseline", session / name
    root.mkdir(exist_ok=False)
    manifest = read_json(source / "manifest.json")
    verify_files(source, manifest["files"])
    for name in ("panels", "labels", "input_indexes"):
        shutil.copytree(source / name, root / name)
    for name in (
        "evaluation.json",
        "adjudications.json",
        "agent_votes.json",
        "engineering_checks.json",
        "private_index.json",
    ):
        shutil.copyfile(source / name, root / name)
    shutil.copytree(source / "variants/baseline", root / "variants/baseline")
    freeze_variant(REPO, root / "variants/candidate")
    diff = subprocess.check_output([shutil.which("git"), "diff", BASELINE], cwd=REPO, text=True)  # noqa: S603
    (root / "candidate.diff").write_text(diff)
    write_json(
        root / "hypothesis.json",
        {"number": args.number, "hypothesis": args.hypothesis, "created": time.time(), "diff_sha256": digest(diff)},
    )
    manifest.update(
        candidate_number=args.number,
        candidate_hypothesis=args.hypothesis,
        engineering_revision=revision,
        frozen_at=time.time(),
    )
    seal(root, manifest)
    declare_verification(root, "candidate")
    records = state.setdefault("engineering_revisions", []) if revision else state["semantic_candidates"]
    records.append({"number": args.number, "revision": revision, "root": str(root), "hypothesis": args.hypothesis})
    write_json(session / "SESSION_STATUS.json", state)
    print(root)


def agreement(rows: dict, labels: list[dict], issues: dict, layer: str, accepted: dict | None = None) -> dict:
    """Keep failures and abstentions in the fixed denominator; alternatives are pair-coherent."""
    correct = pairs = unsafe = blocked = invalid = unresolved = valid = 0
    for label in labels:
        if any(label.get(field) not in {"yes", "no"} for field in FIELDS):
            msg = "Agreement requires explicit binary references; report undecided references separately"
            raise ValueError(msg)
        key = label["pair_id"]
        value = validation(rows[key], layer, issues.get(key, []))
        raw = raw_directions(rows[key], layer)
        choices = (accepted or {}).get(key, [{field: label[field] for field in FIELDS}])
        # Compare raw semantics to one allowed pair; invalid judgments still earn no agreement.
        reference = max(choices, key=lambda c: sum(raw.get(f) == c[f] for f in FIELDS))
        matches = [value[field] == reference[field] for field in FIELDS]
        correct += sum(matches)
        pairs += all(matches)
        unsafe += sum(raw.get(field) == "yes" and reference[field] == "no" for field in FIELDS)
        blocked += sum(raw.get(field) == "no" and reference[field] == "yes" for field in FIELDS)
        invalid += value["validation_status"] == "invalid"
        unresolved += value["validation_status"] == "unresolved"
        valid += value["validation_status"] == "valid"
    n = len(labels)
    return {
        "pairs": n,
        "directions": 2 * n,
        "correct_directions": correct,
        "correct_pairs": pairs,
        "direction_agreement": correct / (2 * n) if n else None,
        "pair_agreement": pairs / n if n else None,
        "reference_wrong_allows": unsafe,
        "reference_wrong_blocks": blocked,
        "invalid_pairs": invalid,
        "unresolved_pairs": unresolved,
        "valid_pairs": valid,
        "valid_judgment_coverage": valid / n if n else None,
    }


def audit_run(args: argparse.Namespace) -> dict:  # noqa: C901, PLR0912, PLR0915
    root = args.root.resolve()
    run_dir = root / "runs" / args.name
    manifest = read_json(root / "manifest.json")
    verify_files(root, manifest["files"])
    report = read_json(run_dir / "report.json")
    execution = read_json(run_dir / "run_manifest.json")
    if report["audit_code_sha256"] != digest((Path(__file__).parent / "policy_eval.py").read_bytes()):
        msg = "Audit code changed since native report"
        raise ValueError(msg)
    for filename, checksum in report["output_files"].items():
        if digest(Path(filename).read_bytes()) != checksum:
            msg = "Native audited output changed"
            raise ValueError(msg)
    rows = exact_index(read_rows(run_dir / "critics"))
    panel = report["panel"]
    exact_index(list(rows.values()), set(exact_index(read_rows(root / "panels" / panel))))
    evaluation = read_json(root / "evaluation.json")
    groups, accepted = {}, {}
    if panel == "diagnostic":
        labels = read_json(root / "labels/diagnostic.json")
        groups["personal120"] = evaluation["fixed_personal_reference"]
        groups["real_consensus6"] = [row for row in labels if row["pair_id"] in evaluation["consensus_ids"]]
    elif panel == "holdout":
        labels = read_json(root / "agent_votes.json")
        resolved_labels = [row for row in labels if all(row.get(f) in {"yes", "no"} for f in FIELDS)]
        groups["development99_original"] = resolved_labels
        revised = {row["pair_id"]: row for row in read_json(root / "adjudications.json")["entries"]}
        groups["development99_reviewed"] = [
            {**row, **{f: revised.get(row["pair_id"], row).get(f, row[f]) for f in FIELDS}} for row in resolved_labels
        ]
        accepted = {k: v["accepted_pair_decisions"] for k, v in revised.items() if "accepted_pair_decisions" in v}
        groups["development97_strict"] = [
            row for row in groups["development99_reviewed"] if row["pair_id"] not in accepted
        ]
    else:
        labels = read_json(root / "labels" / f"{panel}.json")
        for name, group in evaluation["hard_groups"].items():
            if group["panel"] == panel:
                groups[name] = [row for row in labels if row["pair_id"] in group["ids"]]
    result = {
        "run": args.name,
        "root": str(root),
        "panel": panel,
        "groups": {
            name: {
                layer: agreement(
                    rows, labels, report["issues"], layer, accepted if name == "development99_reviewed" else None
                )
                for layer in LAYERS
            }
            for name, labels in groups.items()
        },
        "all_input_validation": report["all_input_validation"],
        "failure_counts": report["failure_counts"],
        "attempt_status_counts": report["attempt_status_counts"],
        "attempt_retries": report["attempt_retries"],
        "stage_direction_changes": report["stage_direction_changes"],
        "issues": report["issues"],
        "integrity_failures": report["integrity_failures"],
        "unmapped_attempts": report["unmapped_attempts"],
        "native_report_sha256": digest((run_dir / "report.json").read_bytes()),
        "evaluation_sha256": digest((root / "evaluation.json").read_bytes()),
        "audit_sha256": digest(Path(__file__).read_bytes()),
        "execution": {
            "started": execution["started"],
            "completed": execution.get("completed"),
            "duration_seconds": execution.get("completed", execution["started"]) - execution["started"],
            "model_revision": execution["model_revision"],
            "judge_config_sha256": execution["judge_config_sha256"],
            "replica_peak_concurrency": report["replica_peak_concurrency"],
        },
    }
    if panel in {"gate", "coverage-regression", "preflight"}:
        result["swap_mismatches"] = {
            layer: [
                key
                for key, row in rows.items()
                if not key.endswith("-swapped")
                and key + "-swapped" in rows
                and any(
                    validation(row, layer, report["issues"].get(key, []))[left]
                    != validation(rows[key + "-swapped"], layer, report["issues"].get(key + "-swapped", []))[right]
                    for left, right in zip(FIELDS, reversed(FIELDS), strict=True)
                )
            ]
            for layer in LAYERS
        }
    records = []
    label_index = {row["pair_id"]: row for row in labels}
    latest = {key: item for item in evaluation["latest_user_decisions"] for key in {item["pair_id"], item["case_id"]}}
    personal = {item["pair_id"]: item for item in evaluation["fixed_personal_reference"]}
    for key, row in rows.items():
        for layer in LAYERS:
            current = personal.get(key, latest.get(key, label_index.get(key, {})))
            records.append(
                {
                    "pair_id": key,
                    "layer": layer,
                    **validation(row, layer, report["issues"].get(key, [])),
                    **{"raw_" + f: raw_directions(row, layer).get(f) for f in FIELDS},
                    **{"frozen_panel_reference_" + f: label_index.get(key, {}).get(f) for f in FIELDS},
                    **{"current_reference_" + f: current.get(f) for f in FIELDS},
                    "accepted_complete_pair_alternatives": json.dumps(accepted.get(key, [])),
                }
            )
    previous = run_dir / "tuning_audit-v2.json"
    if previous.exists() and (old := read_json(previous))["audit_sha256"] != result["audit_sha256"]:
        for suffix in ("json", "csv", "html"):
            path = run_dir / f"tuning_audit-v2.{suffix}"
            if path.exists():
                shutil.copyfile(path, run_dir / f"tuning_audit-{old['audit_sha256'][:12]}.{suffix}")
    write_json(previous, result)
    with (run_dir / "tuning_audit-v2.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (run_dir / "tuning_audit-v2.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Frozen development comparison</title><pre>'
        + html.escape(json.dumps(result, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    return result


def aggregate(
    reports: list[dict], group: str, expected_runs: list[str] | None = None, expected_pairs: int | None = None
) -> dict:
    """Average exact count ratios while preserving every failed hard-regression repeat."""
    selected = [report for report in reports if group in report["groups"]]
    indexed = {report["run"]: report for report in selected}
    if len(indexed) != len(selected):
        msg = "A verification repeat cannot be counted twice"
        raise ValueError(msg)
    names = expected_runs if expected_runs is not None else list(indexed)
    if expected_runs is not None and (len(set(names)) != len(names) or set(indexed) - set(names)):
        msg = "Observed runs must belong to the unique predeclared group"
        raise ValueError(msg)
    missing = [name for name in names if name not in indexed]
    if missing and not expected_pairs:
        msg = "Missing repeats require an explicit frozen pair denominator"
        raise ValueError(msg)
    counts = {
        "direction_agreement": ("correct_directions", "directions"),
        "pair_agreement": ("correct_pairs", "pairs"),
        "valid_judgment_coverage": ("valid_pairs", "pairs"),
    }
    result = {}
    for layer in LAYERS:
        metrics = [
            indexed[name]["groups"][group][layer]
            if name in indexed
            else {
                "pairs": expected_pairs,
                "directions": 2 * expected_pairs,
                "correct_directions": 0,
                "correct_pairs": 0,
                "direction_agreement": 0,
                "pair_agreement": 0,
                "valid_judgment_coverage": 0,
                "valid_pairs": 0,
                "invalid_pairs": 0,
                "unresolved_pairs": 0,
                "incomplete_pairs": expected_pairs,
                "reference_wrong_allows": 0,
                "reference_wrong_blocks": 0,
            }
            for name in names
        ]
        result[layer] = {
            "repeats": names,
            "completed_repeats": len(selected),
            "missing_repeats": missing,
            "denominator_pairs": sorted({item["pairs"] for item in metrics}),
            "each": metrics,
        }
        if metrics:
            for key, (numerator, denominator) in counts.items():
                result[layer][key] = {
                    "each": [item[key] for item in metrics],
                    "mean": float(mean(Fraction(item[numerator], item[denominator]) for item in metrics)),
                    "worst": min(item[key] for item in metrics),
                    "correct_total": sum(item[numerator] for item in metrics),
                    "denominator_total": sum(item[denominator] for item in metrics),
                }
    return result


def declare_verification(root: Path, variant: str) -> None:
    """Commit the three repeat identities before observing any candidate output."""
    write_json(
        root / "verification-plan.json",
        {
            "declared": time.time(),
            "manifest_sha256": digest((root / "manifest.json").read_bytes()),
            "runs": {
                panel: [f"{variant}-{panel}-{repeat}" for repeat in range(1, REPEATS + 1)]
                for panel in ("gate", "coverage-regression", "preflight", "diagnostic", "holdout")
            },
            "replacement_failed_repeats_allowed": False,
        },
    )


def target_status(personal: dict, direction: Fraction, pair: Fraction) -> str:
    """Use the frozen 720/360 denominators, including every invalid judgment."""
    if personal.get("completed_repeats", len(personal.get("repeats", []))) != REPEATS:
        return "NOT_COMPLETED"
    if personal["denominator_pairs"] != [PERSONAL_PAIR_COUNT]:
        msg = "Personal reference denominator changed"
        raise ValueError(msg)
    directions = personal["direction_agreement"]
    pairs = personal["pair_agreement"]
    if (
        directions["denominator_total"] != 2 * PERSONAL_PAIR_COUNT * REPEATS
        or pairs["denominator_total"] != PERSONAL_PAIR_COUNT * REPEATS
    ):
        msg = "Three complete personal-reference repeats are required"
        raise ValueError(msg)
    return (
        "PASS"
        if Fraction(directions["correct_total"], 720) >= direction and Fraction(pairs["correct_total"], 360) >= pair
        else "FAIL"
    )


def nonregression_status(candidate: dict, baseline: dict) -> str:
    if (
        candidate.get("completed_repeats", len(candidate.get("repeats", []))) != REPEATS
        or baseline.get("completed_repeats", len(baseline.get("repeats", []))) != REPEATS
    ):
        return "NOT_COMPLETED"
    a, b = candidate["direction_agreement"], baseline["direction_agreement"]
    return (
        "PASS"
        if Fraction(a["correct_total"], a["denominator_total"]) >= Fraction(b["correct_total"], b["denominator_total"])
        else "FAIL"
    )


def compare(args: argparse.Namespace) -> None:  # noqa: PLR0915
    session = args.session.resolve()
    output = session / args.output
    output.mkdir(exist_ok=False)
    roots = [session / "baseline", session / "start-39714b47", *sorted(session.glob("candidate-*"))]
    reference_files = {
        key: value
        for key, value in read_json(session / "baseline/manifest.json")["files"].items()
        if key.startswith(("panels/", "labels/", "input_indexes/")) or key == "evaluation.json"
    }
    reports, summaries, failed_runs = {}, {}, []
    evaluation = read_json(session / "baseline/evaluation.json")
    group_plans = {
        **{name: (item["panel"], len(item["ids"])) for name, item in evaluation["hard_groups"].items()},
        "real_consensus6": ("diagnostic", len(evaluation["consensus_ids"])),
        "personal120": ("diagnostic", PERSONAL_PAIR_COUNT),
        "development99_original": ("holdout", 99),
        "development99_reviewed": ("holdout", 99),
        "development97_strict": ("holdout", 97),
    }
    for root in roots:
        if not root.is_dir():
            continue
        frozen = read_json(root / "manifest.json")["files"]
        if any(frozen.get(key) != value for key, value in reference_files.items()):
            msg = "Baseline and candidate evaluation artifacts differ"
            raise ValueError(msg)
        for path in sorted((root / "runs").glob("*/run_manifest.json")):
            lifecycle = read_json(path)
            if lifecycle["status"] != "complete":
                failed_runs.append(
                    {
                        "root": root.name,
                        "run": path.parent.name,
                        "status": lifecycle["status"],
                        "expected_input_pairs": read_json(root / "manifest.json")["panels"][lifecycle["panel"]],
                        "main_output_rows": len(read_rows(path.parent / "main"))
                        if (path.parent / "main").exists()
                        else 0,
                        "http_attempts": len(list((path.parent / "receipts").rglob("*.json"))),
                        "error": lifecycle.get("error"),
                        "manifest_sha256": digest(path.read_bytes()),
                    }
                )
        values = [
            audit_run(argparse.Namespace(root=root, name=path.parent.name))
            for path in sorted((root / "runs").glob("*/report.json"))
        ]
        reports[root.name] = values
        declared = (
            read_json(root / "verification-plan.json")["runs"] if (root / "verification-plan.json").exists() else {}
        )
        summaries[root.name] = {
            group: aggregate(values, group, declared.get(panel), pairs)
            for group, (panel, pairs) in group_plans.items()
        }
    best_name = getattr(args, "best_root", None) or f"candidate-{args.best:02d}"
    best = summaries.get(best_name, {})
    baseline = summaries["baseline"]
    hard_groups = {**read_json(session / "baseline/evaluation.json")["hard_groups"], "real_consensus6": {}}
    hard_checks = {
        name: all(
            len(best.get(name, {}).get(layer, {}).get("repeats", [])) == REPEATS
            and best[name][layer]["pair_agreement"]["worst"] == 1
            and best[name][layer]["valid_judgment_coverage"]["worst"] == 1
            for layer in LAYERS
        )
        for name in hard_groups
    }
    personal = best.get("personal120", {}).get("final", {})
    personal_passed = (
        len(personal.get("repeats", [])) == REPEATS
        and personal["direction_agreement"]["worst"] >= TARGET_DIRECTION_AGREEMENT
        and personal["pair_agreement"]["worst"] >= TARGET_PAIR_AGREEMENT
    )
    target = target_status(personal, TARGET_DIRECTION_AGREEMENT, TARGET_PAIR_AGREEMENT)
    milestone = target_status(personal, MILESTONE_DIRECTION_AGREEMENT, MILESTONE_PAIR_AGREEMENT)
    personal_mean_passed = target == "PASS"
    nonregression = {}
    nonregression_states = {}
    for group in ("development99_original", "development99_reviewed", "development97_strict"):
        a, b = best.get(group, {}).get("final", {}), baseline.get(group, {}).get("final", {})
        nonregression_states[group] = nonregression_status(a, b)
        nonregression[group] = nonregression_states[group] == "PASS"
    selected_runs = reports.get(best_name, [])
    plan_path = session / best_name / "verification-plan.json"
    plan = read_json(plan_path) if plan_path.exists() else {"runs": {}}
    expected = {name for names in plan["runs"].values() for name in names}
    observed = {run["run"] for run in selected_runs}
    completion = bool(expected) and expected == observed
    if plan_path.exists() and (
        plan["manifest_sha256"] != digest((session / best_name / "manifest.json").read_bytes())
        or any(run["execution"]["started"] < plan["declared"] for run in selected_runs)
    ):
        msg = "Verification plan must bind the frozen candidate before its first run"
        raise ValueError(msg)
    technical = completion and all(
        not run["integrity_failures"]
        and not run["unmapped_attempts"]
        and all(value == 0 for key, value in run["attempt_status_counts"].items() if key != "valid")
        and all(
            layer.get("invalid", layer.get("invalid_pairs", 0)) == 0 for layer in run["all_input_validation"].values()
        )
        for run in selected_runs
    )
    hard_states = {
        name: "PASS"
        if passed
        else "FAIL"
        if all(best.get(name, {}).get(layer, {}).get("completed_repeats", 0) == REPEATS for layer in LAYERS)
        else "NOT_COMPLETED"
        for name, passed in hard_checks.items()
    }
    result = {
        "baseline_commit": BASELINE,
        "best_candidate": best_name,
        "summary": summaries,
        "runs": reports,
        "failed_runs": failed_runs,
        "requirements": {
            "hard_regressions": hard_checks,
            "personal_targets_each_repeat": personal_passed,
            "personal_targets_mean": personal_mean_passed,
            "development_nonregression_mean": nonregression,
        },
        "requirement_status": {
            "milestone_85_80": milestone,
            "target_90_85": target,
            "hard_regressions": hard_states,
            "technical_validity": "PASS" if technical else "FAIL" if completion else "NOT_COMPLETED",
            "development_nonregression_mean": nonregression_states,
        },
        "acceptance_passed": all(hard_checks.values())
        and personal_mean_passed
        and all(nonregression.values())
        and technical,
        "thresholds": {
            "milestone": {"correct_directions": 612, "directions": 720, "correct_pairs": 288, "pairs": 360},
            "target": {"correct_directions": 648, "directions": 720, "correct_pairs": 306, "pairs": 360},
            "each_repeat_target_required": False,
        },
        "verification_completion": {
            "expected": sorted(expected),
            "observed": sorted(observed),
            "complete": completion,
        },
        "independent_human_accuracy": False,
        "D121": "pending",
        "audit_sha256": digest(Path(__file__).read_bytes()),
        "frozen_evaluation_artifacts": reference_files,
        "created": time.time(),
    }
    write_json(output / "comparison.json", result)
    flat = [
        {"version": version, "run": report["run"], "group": group, "layer": layer, **metrics}
        for version, values in reports.items()
        for report in values
        for group, layers in report["groups"].items()
        for layer, metrics in layers.items()
    ]
    with (output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Qwen retention tuning comparison</title><pre>'
        + html.escape(json.dumps({k: v for k, v in result.items() if k != "runs"}, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "requirements": result["requirements"],
                "acceptance_passed": result["acceptance_passed"],
            }
        )
    )


def execute(args: argparse.Namespace) -> None:
    root, session = args.root.resolve(), args.session.resolve()
    state = read_json(session / "SESSION_STATUS.json")
    deadline = min(state["deadline"], state.get("gpu_stop_at", state["deadline"]))
    service = read_json(session / "service.json")
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + ":" + os.environ["PATH"],
        "PYTHONPATH": str(Path(__file__).parent) + ":" + str(REPO),
        "CUDA_VISIBLE_DEVICES": "0",
        "RAY_OVERRIDE_RESOURCES": '{"CPU":32,"GPU":0}',
    }
    for repeat in args.repeats:
        for panel in args.panels:
            if time.time() + args.estimated_seconds >= min(deadline, state["deadline"] - args.reserve_seconds):
                msg = "Time reserve reached; no new inference run started"
                raise TimeoutError(msg)
            name = f"{args.variant}-{panel}-{repeat}"
            plan_path = root / "verification-plan.json"
            if plan_path.exists() and name not in read_json(plan_path)["runs"].get(panel, []):
                msg = "Run is outside the predeclared verification group"
                raise ValueError(msg)
            if (root / "runs" / name).exists():
                msg = "Run already exists; preserve its evidence and use a new root"
                raise ValueError(msg)
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "policy_eval.py"),
                "run",
                "--root",
                str(root),
                "--variant",
                args.variant,
                "--panel",
                panel,
                "--repeat",
                str(repeat),
                "--inference-endpoint",
                service["endpoint"],
                "--service-container",
                service["container"],
            ]
            if panel == "holdout":
                cmd += ["--label-source", "agent-assisted"]
            write_json(root / "execution_status.json", {"current": name, "started": time.time(), "command": cmd})
            with (root / f"{name}.log").open("w") as log:
                subprocess.run(  # noqa: S603
                    cmd,
                    cwd=REPO,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(1, deadline - time.time()),
                )
            result = audit_run(argparse.Namespace(root=root, name=name))
            print(
                json.dumps({"run": name, "groups": result["groups"], "failures": result["failure_counts"]}), flush=True
            )
            if result["integrity_failures"] or result["unmapped_attempts"]:
                msg = "Native integrity boundary failed"
                raise RuntimeError(msg)
    write_json(root / "execution_status.json", {"current": None, "completed": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--session", type=Path, required=True)
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--review", type=Path, required=True)
    prep.add_argument("--engineering-receipt", type=Path, required=True)
    cand = commands.add_parser("candidate")
    cand.add_argument("--session", type=Path, required=True)
    cand.add_argument("--number", type=int, required=True)
    cand.add_argument("--hypothesis", required=True)
    cand.add_argument(
        "--revision", help="Engineering-only revision of an existing semantic candidate; preserve prior runs"
    )
    audit = commands.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--name", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--session", type=Path, required=True)
    comparison.add_argument("--best", type=int, required=True)
    comparison.add_argument("--best-root", help="Selected engineering revision directory name, when applicable")
    comparison.add_argument("--output", required=True, help="New immutable comparison directory within session")
    run = commands.add_parser("execute")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--session", type=Path, required=True)
    run.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    run.add_argument(
        "--panels",
        nargs="+",
        choices=("gate", "coverage-regression", "preflight", "diagnostic", "holdout"),
        required=True,
    )
    run.add_argument("--repeats", nargs="+", type=int, choices=(1, 2, 3), default=[1])
    run.add_argument("--reserve-seconds", type=int, default=3600)
    run.add_argument("--estimated-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit_run(args), ensure_ascii=False, indent=2))
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
