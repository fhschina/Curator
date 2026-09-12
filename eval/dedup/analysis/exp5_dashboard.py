# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Immutable Pair Explorer snapshots and descriptive Exp5/v0.5 comparison."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
from collections import Counter
from pathlib import Path

from eval.dedup import dashboard
from eval.dedup.analysis import exp5_full20k as experiment
from eval.dedup.analysis.comparison import build_pair_comparisons
from eval.dedup.analysis.metrics import compute_metrics
from eval.dedup.rejudge_comparison import _read_jsonl, _write_jsonl, build_agreement_summary
from eval.dedup.report import _group_size_bucket, _ratio_bucket
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

PRIMARY = ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
V05_RELEASE = Path(
    "/raid/hfang/codex-home/visualizations/2026/09/02/01a063ce-7f9e-7e33-8507-1aa214c561f9/"
    "dedup-eval-v0.5/release_manifest.json"
)


def baseline_root(release_path: Path) -> Path:
    release = experiment.read(release_path)
    require(
        release["version"] == "V0.5"
        and release["version_definition"]["framework"] == "Sarah MinHash judging framework",
        "EXP5_V05_RELEASE",
        "use the published Sarah/Qwen v0.5, not the earlier V0/DeepSeek baseline",
    )
    root = Path(release["result_run_root"])
    require(root.name == release["result_run_id"], "EXP5_V05_ROOT", "published result run identity")
    for artifact in release["artifacts"].values():
        if "sha256" in artifact:
            require(
                sha256_file(Path(artifact["path"])) == artifact["sha256"],
                "EXP5_V05_BINDING",
                "published baseline bindings unchanged",
            )
    return root


def comparison_rows(results: list[dict], baseline: dict[str, dict]) -> list[dict]:
    rows = []
    for result in results:
        key = result["canonical_pair_id"]
        old, new = baseline.get(key), result["public"]
        comparable = old is not None and result["status"] == "VALID"
        rows.append(
            {
                "canonical_pair_id": key,
                "exp5_status": result["status"],
                "v05_status": "VALID" if old is not None else "NO_VALID_OUTPUT",
                "primary_changed": any(old[f] != new[f] for f in PRIMARY) if comparable else None,
                **{f"exp5_{f}": new[f] if result["status"] == "VALID" else "ENGINEERING_FAILURE" for f in PRIMARY},
                **{f"v05_{f}": old[f] if old is not None else "NO_VALID_OUTPUT" for f in PRIMARY},
            }
        )
    return rows


def explorer_html(records: list[dict], contexts: dict, summary: dict) -> str:
    output = dashboard.pair_explorer_html(
        evaluation_run_id=experiment.VERSION, records=records, group_contexts=contexts
    )
    call = "summary(r);sut(r);judge(r);"
    require(
        output.count(call) == 1 and output.count("<script>") == 1,
        "EXP5_EXPLORER_TEMPLATE",
        "known native Pair Explorer integration points",
    )
    function = """
function baselineComparison(r){
 const s=section("v0.5 vs Exp5", "Same pair; version agreement is not accuracy. Directions refer to the original A/B presentation."), g=create("div","facts"), b=r.baseline_v05;
 fact(g,"v0.5 valid output",b?"YES":"NO");
 for(const f of ["same_duplicate_group","a_can_replace_b","b_can_replace_a"]){
  fact(g,`v0.5 ${f}`,b?b[f]:"NO_VALID_OUTPUT");
  fact(g,`Exp5 ${f}`,r.exp5_primary[f]);
 }
 s.appendChild(g);
}
"""
    output = output.replace("<script>", "<script>" + function, 1).replace(call, call + "baselineComparison(r);")
    note = (
        f"Exp5 · {'FINAL' if summary['complete'] else 'PARTIAL SNAPSHOT'} · "
        f"{summary['collected']:,} / {summary['population']:,} collected · "
        f"{summary['pending']:,} pending · {summary['engineering_failures']:,} engineering failures. "
        "Experimental checkpoint, not a release or independent holdout result. "
        "MinHash diagnostics: UNAVAILABLE_MISSING_CONTRACT."
    )
    return output.replace("</header>", '<p class="muted">' + html.escape(note) + "</p></header>", 1)


def calibration(results: list[dict], baseline: dict[str, dict]) -> dict:
    from eval.dedup.analysis import adjudicated_rebenchmark as benchmark
    from eval.dedup.analysis import exp1_reproduction as historical

    labels = benchmark.historical._read_csv(historical.REBENCHMARK / "reference_revised_1000.csv")
    exclusions = set(
        experiment.read(historical.REBENCHMARK / "comparison_exclusions.json")["excluded_canonical_pair_ids"]
    )
    by_id = {r["canonical_pair_id"]: r for r in results}
    keys = {r["canonical_pair_id"] for r in labels}
    if not keys <= by_id.keys():
        return {
            "status": "PENDING_COMPLETE_DEVELOPMENT_COHORT",
            "available": len(keys & by_id.keys()),
            "required": len(keys),
        }
    old = [
        {"canonical_pair_id": key, **baseline[key]}
        if key in baseline
        else {
            "canonical_pair_id": key,
            **experiment.runtime.old.unresolved_judge_output_v3(),
            "metric_only_missing_output": True,
        }
        for key in keys
    ]
    return {
        "status": "AVAILABLE",
        "independent_holdout": False,
        "reference_note": "Mixed development reference: user-confirmed, assistant-applied and inherited labels; not independent human gold.",
        "reference_sha256": sha256_file(historical.REBENCHMARK / "reference_revised_1000.csv"),
        "exclusions_sha256": sha256_file(historical.REBENCHMARK / "comparison_exclusions.json"),
        "v05": benchmark.masked_score(labels, old, exclusions),
        "exp5": benchmark.masked_score(labels, [historical.projection(by_id[key]) for key in keys], exclusions),
    }


def build(root: Path, destination: Path, *, preview: bool = False) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    manifest = experiment.verify(root)
    require(
        preview or (root / "complete.json").is_file(),
        "EXP5_REPORT_INCOMPLETE",
        "final report requires audited completion",
    )
    if not preview:
        for path, digest in experiment.read(root / "complete.json")["artifacts"].items():
            require(sha256_file(Path(path)) == digest, "EXP5_REPORT_BINDING", "completed artifacts unchanged")
    require(not destination.exists(), "EXP5_REPORT_EXISTS", "new immutable snapshot destination")
    index = experiment.read(root / "panel_index.json")
    expected = {r["canonical_pair_id"] for r in index}
    paths = sorted((root / "results").glob("*.json"))
    results = [experiment.read(p) for p in paths]
    keys = {r["canonical_pair_id"] for r in results}
    require(
        len(keys) == len(results) and keys <= expected and (preview or keys == expected),
        "EXP5_REPORT_JOIN",
        "full join, or an explicitly partial snapshot",
    )
    require(bool(keys), "EXP5_REPORT_EMPTY", "wait for at least one collected pair")
    source = Path(manifest["source_run"])
    v05 = baseline_root(V05_RELEASE)
    baseline_rows = _read_jsonl(v05 / "data/judge_results.jsonl")
    baseline = {r["canonical_pair_id"]: r for r in baseline_rows}
    require(
        len(baseline_rows) == len(baseline) and set(baseline) == expected,
        "EXP5_V05_MEMBERSHIP",
        "complete published v0.5 results on exactly the same 20k pairs",
    )
    candidates = pq.read_table(source / "data/candidate_pairs.parquet")
    orientation = lambda r: (r["canonical_pair_id"], r["presented_doc_a"], r["presented_doc_b"])  # noqa: E731
    require(
        {orientation(r) for r in candidates.to_pylist()}
        == {orientation(r) for r in pq.read_table(v05 / "data/candidate_pairs.parquet").to_pylist()},
        "EXP5_V05_ORIENTATION",
        "replacement directions refer to identical A/B presentation",
    )
    selected, flat, errors, payloads = [], [], [], []
    by_id = {r["canonical_pair_id"]: r for r in results}
    for candidate in candidates.to_pylist():
        key = candidate["canonical_pair_id"]
        if key not in keys:
            continue
        result = by_id[key]
        payload = experiment.read(root / "inputs" / (key + ".json"))["payload"]
        digest = sha256_json(payload)
        selected.append({**candidate, "evaluation_run_id": experiment.VERSION, "judge_payload_hash": digest})
        payloads.append({"canonical_pair_id": key, "judge_payload_hash": digest, "payload": payload})
        attempts = len([s for s in result["stages"] if s["stage"].startswith("main-")])
        if result["status"] == "VALID":
            flat.append(
                {
                    **result["public"],
                    "canonical_pair_id": key,
                    "canonical_pair_id_version": candidate["canonical_pair_id_version"],
                    "judge_payload_hash": digest,
                    "attempts": attempts,
                    "prompt_version": experiment.VERSION,
                    "runtime_version": result["version"],
                    "schema_version": "dedup-judge-output-v3",
                }
            )
        else:
            errors.append(
                {"canonical_pair_id": key, "attempts": attempts, "errors": [{"error_type": result["error_code"]}]}
            )
    destination.mkdir(mode=0o700, parents=True)
    for folder in ("data", "logs", "manifests", "reports"):
        (destination / folder).mkdir()
    for relative in (
        "data/pair_provenance.parquet",
        "data/document_outcomes.parquet",
        "manifests/sut_run_manifest.json",
    ):
        (destination / relative).symlink_to((source / relative).resolve())
    pq.write_table(
        pa.Table.from_pylist(selected, schema=candidates.schema), destination / "data/candidate_pairs.parquet"
    )
    _write_jsonl(destination / "data/judge_results.jsonl", flat)
    _write_jsonl(destination / "logs/judge_errors.jsonl", errors)
    _write_jsonl(destination / "data/judge_payloads.jsonl", payloads)
    del payloads
    comparison_path = destination / "data/pair_comparisons.parquet"
    build_pair_comparisons(
        candidate_pairs_path=destination / "data/candidate_pairs.parquet",
        pair_provenance_path=source / "data/pair_provenance.parquet",
        outcomes_path=source / "data/document_outcomes.parquet",
        judge_results_path=destination / "data/judge_results.jsonl",
        judge_errors_path=destination / "logs/judge_errors.jsonl",
        destination=comparison_path,
    )
    comparisons = pq.read_table(comparison_path).to_pylist()
    for row in comparisons:
        row["report_group_size_bucket"] = _group_size_bucket(int(row["predicted_group_size_low"]))
        row["report_ratio_bucket"] = _ratio_bucket(float(row["token_length_ratio"]))
    records = dashboard.build_pair_explorer_records(destination, comparisons)
    for record in records:
        key = record["pair_id"]
        record["baseline_v05"] = {f: baseline[key][f] for f in PRIMARY} if key in baseline else None
        record["exp5_primary"] = {
            f: by_id[key]["public"][f] if by_id[key]["status"] == "VALID" else "ENGINEERING_FAILURE" for f in PRIMARY
        }
    contexts = dashboard.attach_group_context(destination, records)
    agreement, _ = build_agreement_summary([r for key, r in baseline.items() if key in keys], flat)
    rows = comparison_rows(results, baseline)
    summary = {
        "version": experiment.VERSION,
        "baseline": "v0.5",
        "baseline_definition": "Published Sarah MinHash framework + Qwen; earlier DeepSeek is V0, not v0.5.",
        "baseline_run_root": str(v05),
        "baseline_release_sha256": sha256_file(V05_RELEASE),
        "baseline_results_sha256": sha256_file(v05 / "data/judge_results.jsonl"),
        "complete": not preview,
        "population": len(expected),
        "collected": len(results),
        "pending": len(expected) - len(results),
        "valid": len(flat),
        "engineering_failures": len(errors),
        "semantic_unresolved": sum(r["same_duplicate_group"] == "UNRESOLVED" for r in flat),
        "primary_changed_common_valid": sum(r["primary_changed"] is True for r in rows),
        "agreement": agreement,
        "release_eligible": False,
        "scope_note": "Same pair IDs and logical Qwen model; different prompts, semantic contracts and historical serving conditions. Agreement is not accuracy or version superiority. SUT metrics are Judge-conditioned, not independent gold.",
        "minhash_diagnostics": "UNAVAILABLE_MISSING_CONTRACT",
        "exp5_groups": dict(Counter(r["same_duplicate_group"] for r in flat)),
        "v05_groups_same_collected_pairs": dict(
            Counter(r["same_duplicate_group"] for k, r in baseline.items() if k in keys)
        ),
        "development_calibration": calibration(results, baseline),
        "proxy_challenge": {"status": "NOT_USED_FOR_VERSION_SELECTION", "independent_human_gold": False},
    }
    if not preview:
        summary["judge_conditioned_sut_metrics"] = compute_metrics(
            comparison_path,
            requested_judge_pairs=len(expected),
            metrics_destination=destination / "reports/sut_metrics.json",
            slices_destination=destination / "reports/sut_slices.csv",
            accounting_destination=destination / "reports/pipeline_accounting.csv",
            stage_markers=[],
        )
    write_json_atomic(destination / "reports/comparison.json", summary)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(destination / "reports/v05_exp5_pairs.csv", buffer.getvalue())
    write_text_atomic(destination / "reports/pair_explorer_exp5.html", explorer_html(records, contexts, summary))
    write_text_atomic(
        destination / "reports/RESULTS.md",
        "\n".join(
            [
                "# Exp5 vs v0.5",
                "",
                "PARTIAL SNAPSHOT" if preview else "Full 20,000-pair experimental result",
                "",
                f"Collected {len(results):,}/{len(expected):,}; valid {len(flat):,}; engineering failures {len(errors):,}.",
                "",
                f"Primary decisions differ on {summary['primary_changed_common_valid']:,} common-valid pairs. This is not an accuracy score.",
                "",
                "The Pair Explorer retains the native filters, evidence, SUT context and review UI, with a v0.5/Exp5 panel for every pair.",
                "",
                "Development calibration is separate in comparison.json. It uses the same revised reference and five comparison exclusions for both versions; it is not independent holdout validation.",
                "",
                "Experimental checkpoint, not release. Actual SUT MinHash replay remains unavailable without the resolved configuration.",
                "",
            ]
        ),
    )
    write_json_atomic(
        destination / "snapshot.json",
        {
            "source_contract_digest": manifest["contract_digest"],
            "preview": preview,
            "result_bindings": {str(p): sha256_file(p) for p in paths},
            "files": {str(p): sha256_file(p) for p in destination.rglob("*") if p.is_file() and not p.is_symlink()},
        },
    )
    return {
        k: summary[k]
        for k in ("version", "complete", "population", "collected", "pending", "valid", "engineering_failures")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.destination.resolve(), preview=args.preview)))


if __name__ == "__main__":
    main()
