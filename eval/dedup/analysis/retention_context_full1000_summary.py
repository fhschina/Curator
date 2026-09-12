# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Audit and summarize a fully collected run without making any inference calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from eval.dedup.analysis import retention_context_full1000 as experiment
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic


def record_progress(root: Path, completed: int, population: int) -> None:
    """Append immutable snapshots; never overwrite the first completion record."""
    write_json_atomic(
        root / "progress_snapshots" / f"{completed:04d}.json",
        {"completed": completed, "population": population, "source": "persisted_result_files"},
    )


def replay_valid(row: dict, result: dict, load: Callable[[str], dict]) -> dict:
    p = row["payload"]
    adapter = experiment.pilot.candidate
    raw = load("main") if adapter.input_route(p) == "MODEL_REVIEW" else None
    public = adapter.adapt_main(raw, p)
    require(public == result["components"]["main"], "CONTEXT_SUMMARY_MAIN", "same received main projection")
    if adapter.critic_route(public, p) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
        public, _ = adapter.apply_critic(public, p, load("coverage"))
    require(public == result["components"]["coverage"], "CONTEXT_SUMMARY_COVERAGE", "same received critic projection")
    if adapter.subject_route(public, p) == "REVIEW_BILATERAL_SUBJECTS":
        proposal = load("subject")
        if adapter.subject_proof(public, p, proposal) is not None:
            public, _ = adapter.apply_subject_verification(public, p, proposal, load("verifier"))
    require(public == result["public"], "CONTEXT_SUMMARY_FINAL", "same fresh dependent-stage final projection")
    return public


def audit(root: Path) -> dict:
    manifest = experiment.adjudication.read(root / "manifest.json")
    sources = experiment.pilot.reference.verify_freeze(root / "manifest.json")
    rows = experiment.adjudication.read(root / "panel_private.json")
    require(len(rows) == manifest["population"] == 1000, "CONTEXT_SUMMARY_POPULATION", "original full run only")
    expected = {f"1-candidate-{r['canonical_pair_id']}.json" for r in rows}
    require(
        expected == {p.name for p in (root / "results").glob("*.json")},
        "CONTEXT_SUMMARY_INCOMPLETE",
        "all 1000 persisted results required before scoring",
    )
    requests = experiment._index(experiment.adjudication.read(root / "main_requests.json"), "main requests")
    seen, valid = set(), 0
    for row in rows:
        key = f"1-candidate-{row['canonical_pair_id']}"
        result = experiment.adjudication.read(root / "results" / (key + ".json"))
        require(
            result["canonical_pair_id"] == row["canonical_pair_id"]
            and result["review_id"] == row["review_id"]
            and result["repeat"] == 1
            and result["arm"] == "candidate",
            "CONTEXT_SUMMARY_ID",
            "fresh candidate result bound to exact pair",
        )
        loaded, values = [], {}
        for stage in result["stages"]:
            name = key + "-" + stage["stage"] + ".json"
            seen.add(name)
            request = experiment.adjudication.read(root / "requests" / name)
            receipt = experiment.adjudication.read(root / "responses" / name)
            require(
                receipt["request_sha256"] == request["request_sha256"] == sha256_json(request["body"])
                and stage["response_sha256"] == sha256_file(root / "responses" / name),
                "CONTEXT_SUMMARY_RECEIPT",
                "unaltered bound request and response",
            )
            require(
                all(request["body"][k] == v for k, v in manifest["generation"].items()),
                "CONTEXT_SUMMARY_GENERATION",
                "frozen generation settings",
            )
            if stage["stage"] == "main":
                require(
                    request["body"] == requests[row["canonical_pair_id"]]["body"],
                    "CONTEXT_SUMMARY_REQUEST",
                    "exact frozen main request",
                )
            if receipt["status"] == "RECEIVED":
                choice = receipt["raw_response"]["choices"][0]
                parsed = experiment.pilot.common.strict_json(choice["message"]["content"])
                require(
                    choice["finish_reason"] == "stop" and parsed == receipt["parsed"],
                    "CONTEXT_SUMMARY_RAW",
                    "parsed fields equal original complete raw answer",
                )
                values[stage["stage"]] = parsed

        def load(stage: str, *, selected_values: dict = values, loaded_stages: list = loaded) -> dict:
            loaded_stages.append(stage)
            require(stage in selected_values, "CONTEXT_SUMMARY_MISSING_STAGE", "no borrowing another pair's response")
            return selected_values[stage]

        if result["status"] == "VALID":
            replay_valid(row, result, load)
            require(
                loaded == [s["stage"] for s in result["stages"]], "CONTEXT_SUMMARY_ROUTES", "same exact stage routing"
            )
            valid += 1
        else:
            require(
                set(experiment.adjudication.primary(result["public"]).values()) == {"UNRESOLVED"},
                "CONTEXT_SUMMARY_FAILURE",
                "engineering failure is not a semantic answer",
            )
    require(
        seen
        == {p.name for p in (root / "requests").glob("*.json")}
        == {p.name for p in (root / "responses").glob("*.json")},
        "CONTEXT_SUMMARY_ORPHANS",
        "no unaccounted or borrowed stage responses",
    )
    return {
        "version": experiment.VERSION,
        "population": 1000,
        "valid_pipeline_replays": valid,
        "engineering_failure_placeholders": 1000 - valid,
        "bound_receipts": len(seen),
        "verified_source_files": len(sources),
        "additional_model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root)
    experiment.export(root)
    record_progress(root, 1000, 1000)
    write_json_atomic(root / "post_collection_audit.json", report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
