# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Fresh frozen exp3 development run against two immutable historical predictions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.dedup.analysis import adjudicated_rebenchmark as adjudication
from eval.dedup.analysis import retention_context_experiment as pilot
from eval.dedup.analysis.development_diagnostic import _index, classify_primary_error
from eval.dedup.analysis.judge_calibration import _weight
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

VERSION = pilot.runtime.VERSION
HISTORICAL = (adjudication.BASELINE, adjudication.CANDIDATE)
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "retention_context_full1000.md"
PILOT = Path("/raid/hfang/ihb/runs/v06233-exp3-pilot-v1")
REBENCHMARK = pilot.previous.REBENCHMARK


def prepare(root: Path) -> dict:
    root = root.resolve()
    require(not root.exists(), "CONTEXT_FULL_ROOT", "new run root; no cache/output reuse")
    sources = pilot.reference.verify_freeze(PILOT / "manifest.json")
    sources.update(pilot.reference.verify_freeze(PILOT / "audit_manifest.json"))
    sources.update(pilot.reference.verify_freeze(REBENCHMARK / "summary.json"))
    old = adjudication.read(PILOT / "manifest.json")
    complete = adjudication.read(PILOT / "complete.json")
    require(
        old["version"] == VERSION
        and complete["assessment_sha256"] == sha256_file(PILOT / "assessment.json")
        and complete["predeclared_candidate_checks_passed"],
        "CONTEXT_FULL_ADMISSION",
        "completed, reviewed, passed pilot required",
    )
    source_panel = adjudication.SESSION / "v4-subject-v2-full1000/panel_private.json"
    original = adjudication.read(source_panel)
    rows = [{k: r[k] for k in ("canonical_pair_id", "review_id", "payload")} for r in original]
    labels = adjudication.historical._read_csv(REBENCHMARK / "reference_revised_1000.csv")
    require(len(rows) == len(labels) == 1000, "CONTEXT_FULL_POPULATION", "original 1000 pairs required")
    require(_index(rows, "panel").keys() == _index(labels, "reference").keys(), "CONTEXT_FULL_JOIN", "exact join")
    mask = adjudication.read(REBENCHMARK / "comparison_exclusions.json")
    excluded = set(mask["excluded_canonical_pair_ids"])
    require(len(excluded) == 5, "CONTEXT_FULL_MASK", "only the five previously approved exclusions")
    pilot.warmup()
    requests, routes, context_failures = [], Counter(), []
    for row in rows:
        route = pilot.candidate.input_route(row["payload"])
        routes[route] += 1
        if route != "MODEL_REVIEW":
            continue
        body = pilot.request_body(pilot.runtime.messages(row["payload"]), pilot.candidate.response_schema())
        try:
            count = pilot.check_context(body)
        except DedupEvaluationError as exc:
            require(
                exc.issue.code == "RETENTION_PILOT_CONTEXT", "CONTEXT_FULL_PREFLIGHT", "unexpected preflight error"
            )
            count = None
            context_failures.append(row["review_id"])
        requests.append(
            {
                "canonical_pair_id": row["canonical_pair_id"],
                "body": body,
                "request_sha256": sha256_json(body),
                "input_tokens_with_schema": count,
            }
        )
    histories = {}
    historical_scores = adjudication.read(REBENCHMARK / "scores.json")["phases"]["revised_reference_masked995"]
    for version in HISTORICAL:
        predictions = adjudication.read(REBENCHMARK / f"{version}_primary_predictions_1000.json")
        observed = adjudication.masked_score(labels, predictions, excluded)
        require(observed == historical_scores[version], "CONTEXT_FULL_BASELINE", "exact historical score reproduction")
        histories[version] = predictions
    for path in (
        Path(__file__).resolve(),
        PROTOCOL,
        HERE.parents[2] / "tests/eval/dedup/test_retention_context_full1000.py",
        source_panel,
    ):
        sources[str(path)] = sha256_file(path)
    sources.update(pilot.runtime.specification()["sources"])
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "main_requests.json", requests)
    write_json_atomic(root / "historical_predictions.json", histories)
    write_json_atomic(root / "comparison_exclusions.json", mask)
    write_text_atomic(root / "reference_revised_1000.csv", (REBENCHMARK / "reference_revised_1000.csv").read_text())
    write_text_atomic(root / "protocol.md", PROTOCOL.read_text())
    manifest = {
        "version": VERSION,
        "status": "FROZEN_FULL_DEVELOPMENT_DIAGNOSTIC_NOT_RELEASE",
        "model": old["model"],
        "endpoint": old["endpoint"],
        "generation": old["generation"],
        "population": 1000,
        "scored_population": 995,
        "repeat_count": 1,
        "fresh_upstream": True,
        "historical_comparison_only": list(HISTORICAL),
        "reference_changed": False,
        "old_cache_reused": False,
        "semantic_repair_retries": 0,
        "input_routes": dict(routes),
        "main_context_preflight_failures": context_failures,
        "max_logical_calls": 4000,
        "max_external_attempts": 8000,
        "tokenizer_initialization": "SERIAL_BEFORE_WORKERS",
        "known_residuals": [
            "independent_fact_conflict_taxonomy",
            "weak_substantive_witness_selection",
            "inherited_reference_disputes",
            "incomplete_evidence_abstention",
        ],
        "release_eligible": False,
        "full20k_admitted": False,
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def projection(result: dict, stage: str = "final") -> dict:
    public = result["public"] if stage == "final" else result["components"].get(stage)
    missing = (stage == "final" and result["status"] != "VALID") or public is None
    if missing:
        public = pilot.unresolved_judge_output_v4("FULL_DEVELOPMENT_MISSING_STAGE")
    return {
        "canonical_pair_id": result["canonical_pair_id"],
        **public,
        **({"metric_only_missing_output": True} if missing else {}),
    }


def error(label: dict, prediction: dict) -> str:
    return (
        "ENGINEERING_MISSING_OUTPUT"
        if prediction.get("metric_only_missing_output")
        else classify_primary_error(label, prediction)
    )


def transitions(labels: list[dict], before: list[dict], after: list[dict]) -> dict:
    a, b = _index(before, "before"), _index(after, "after")
    require(_index(labels, "reference").keys() == a.keys() == b.keys(), "CONTEXT_FULL_TRANSITION_JOIN", "exact join")
    bins = defaultdict(lambda: {"pairs": 0, "weight": 0.0, "review_ids": []})
    for label in labels:
        pid = label["canonical_pair_id"]
        key = f"{error(label, a[pid])} -> {error(label, b[pid])}"
        bins[key]["pairs"] += 1
        bins[key]["weight"] += _weight(label)
        bins[key]["review_ids"].append(label["review_id"])
    corrected = [v for k, v in bins.items() if not k.startswith("CORRECT ->") and k.endswith("-> CORRECT")]
    regressed = [v for k, v in bins.items() if k.startswith("CORRECT ->") and not k.endswith("-> CORRECT")]
    return {
        "transitions": dict(bins),
        "corrected_pairs": sum(v["pairs"] for v in corrected),
        "regressed_pairs": sum(v["pairs"] for v in regressed),
        "corrected_weight": sum(v["weight"] for v in corrected),
        "regressed_weight": sum(v["weight"] for v in regressed),
    }


def compare(labels: list[dict], histories: dict, results: list[dict], excluded: set[str]) -> dict:
    require(set(histories) == set(HISTORICAL), "CONTEXT_FULL_VERSIONS", "both unchanged historical views required")
    require(
        _index(labels, "reference").keys() == _index(results, "new results").keys(),
        "CONTEXT_FULL_RESULTS",
        "no dropping failures",
    )
    views = {**histories, VERSION: [projection(r) for r in results]}
    scores = {v: adjudication.masked_score(labels, predictions, excluded) for v, predictions in views.items()}
    for score in scores.values():
        score["interpretation"] = (
            "Fresh candidate versus frozen historical outputs on identical mixed development reference; not simultaneous or independent validation."
        )
    selected = [r for r in labels if r["canonical_pair_id"] not in excluded]
    select = lambda rr: [r for r in rr if r["canonical_pair_id"] not in excluded]  # noqa: E731
    paired = {v: transitions(selected, select(views[v]), select(views[VERSION])) for v in HISTORICAL}
    indices = {v: _index(rr, v) for v, rr in views.items()}
    result_index = _index(results, "candidate")
    ledger = []
    for label in labels:
        pid = label["canonical_pair_id"]
        ledger.append(
            {
                "review_id": label["review_id"],
                "canonical_pair_id": pid,
                "weight": _weight(label),
                "scoring_eligibility": "EXCLUDED" if pid in excluded else "INCLUDED",
                "reference_provenance": label.get("adjudication_provenance"),
                "historical_reason_cohort": label.get("human_reason_code"),
                "reference": adjudication.reference.primary(label, "human_"),
                "predictions": {v: adjudication.primary(ix[pid]) for v, ix in indices.items()},
                "errors": {
                    v: "EXCLUDED_NOT_SCORED" if pid in excluded else error(label, ix[pid]) for v, ix in indices.items()
                },
                "candidate_status": result_index[pid]["status"],
                "candidate_engineering_error": result_index[pid].get("error_code"),
                "candidate_relation": indices[VERSION][pid]["relation_type"],
                "candidate_overlap": indices[VERSION][pid]["dominant_overlap_source"],
                "candidate_confidence_tier": indices[VERSION][pid]["confidence_tier"],
            }
        )
    cohorts = {}
    for reason in sorted({r.get("human_reason_code", "UNSPECIFIED") for r in selected}):
        refs = [r for r in selected if r.get("human_reason_code", "UNSPECIFIED") == reason]
        cohorts[reason] = {
            v: adjudication.reference.score(refs, [ix[r["canonical_pair_id"]] for r in refs])
            for v, ix in indices.items()
        }
    confidence = {}
    for tier in ("HIGH", "MEDIUM", "LOW"):
        refs = [r for r in selected if indices[VERSION][r["canonical_pair_id"]]["confidence_tier"] == tier]
        confidence[tier] = (
            adjudication.reference.score(refs, [indices[VERSION][r["canonical_pair_id"]] for r in refs])
            if refs
            else None
        )
    stages = {stage: [projection(r, stage) for r in results] for stage in ("main", "coverage", "final")}
    stage_scores = {stage: adjudication.masked_score(labels, rr, excluded) for stage, rr in stages.items()}
    stage_effects = {
        f"{a}_to_{b}": transitions(selected, select(stages[a]), select(stages[b]))
        for a, b in (("main", "coverage"), ("coverage", "final"), ("main", "final"))
    }
    weighted = scores[VERSION]["weighted"]
    numeric_checks = {
        "weighted_precision_75": weighted["duplicate_precision"] is not None
        and weighted["duplicate_precision"] >= 0.75,
        "weighted_recall_75": weighted["duplicate_recall"] is not None and weighted["duplicate_recall"] >= 0.75,
        "weighted_primary_exact_79": weighted["primary_decision_exact"] >= 0.79,
        "over_group_max66": scores[VERSION]["error_counts"].get("OVER_GROUP", 0) <= 66,
        "protected_cohorts_vs_historical": {
            v: {
                c: cohorts[c][VERSION]["weighted"]["primary_decision_exact"]
                >= cohorts[c][v]["weighted"]["primary_decision_exact"] - 0.03
                for c in ("identity_slot", "meaningful_addition", "translation")
                if c in cohorts
            }
            for v in HISTORICAL
        },
    }
    return {
        "scores": scores,
        "paired_comparisons": paired,
        "ledger": ledger,
        "cohorts": cohorts,
        "confidence_tiers": confidence,
        "stage_scores": stage_scores,
        "stage_effects": stage_effects,
        "development_numeric_checks": numeric_checks,
        "release_eligible": False,
        "full20k_admitted": False,
    }


def export(root: Path) -> dict:
    pilot.reference.verify_freeze(root / "manifest.json")
    labels = adjudication.historical._read_csv(root / "reference_revised_1000.csv")
    rows = adjudication.read(root / "panel_private.json")
    results = [adjudication.read(root / "results" / f"1-candidate-{r['canonical_pair_id']}.json") for r in rows]
    excluded = set(adjudication.read(root / "comparison_exclusions.json")["excluded_canonical_pair_ids"])
    report = compare(labels, adjudication.read(root / "historical_predictions.json"), results, excluded)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    attempts = [e for e in events if e.get("external_request")]
    failures = [r for r in results if r["status"] != "VALID"]
    receipts = [adjudication.read(p) for p in (root / "responses").glob("*.json")]
    retry_hashes = {e["request_hash"] for e in attempts if e.get("upstream_attempt", 1) > 1}
    groups = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        p = row["payload"]
        pilot.read_versioned_output(result["public"], pilot.JUDGE_SCHEMA_V4)
        pilot.candidate.validate_evidence_offsets(result["public"], p)
        a, b = (p[f"document_{s}"]["text"] for s in ("a", "b"))
        if a and b and p["long_document_evidence"]["truncated"] is False:
            groups[sha256_json(sorted((a, b)))].append(
                {"review_id": row["review_id"], "primary": pilot.preflight.normalized_primary(result["public"], a, b)}
            )
    engineering = {
        "population": len(results),
        "valid_pipeline_results": len(results) - len(failures),
        "engineering_failures": len(failures),
        "failure_codes": dict(Counter(r["error_code"] for r in failures)),
        "failures": [{"review_id": r["review_id"], "error_code": r["error_code"]} for r in failures],
        "schema_completion_excluding_failure_sentinels": (len(results) - len(failures)) / len(results),
        "group_unresolved": sum(r["public"]["same_duplicate_group"] == "UNRESOLVED" for r in results),
        "stage_calls": dict(Counter(s["stage"] for r in results for s in r["stages"])),
        "logical_calls": len(receipts),
        "external_attempts": len(attempts),
        "transport_retries": sum(e.get("upstream_attempt", 1) > 1 for e in attempts),
        "retried_request_bodies": len(retry_hashes),
        "semantic_repair_retries": 0,
        "http_statuses": dict(Counter(str(e["http_status"]) for e in attempts)),
        "duplicate_text_groups": sum(len(g) > 1 for g in groups.values()),
        "exact_input_disagreement_groups": [
            g for g in groups.values() if len({sha256_json(r["primary"]) for r in g}) > 1
        ],
        "one_candidate_repeat_only": True,
    }
    report["engineering"] = engineering
    report["input_routes"] = adjudication.read(root / "manifest.json")["input_routes"]
    report["response_artifacts"] = {
        str(p): sha256_file(p)
        for folder in ("requests", "responses", "results")
        for p in (root / folder).glob("*.json")
    }
    ledger = report.pop("ledger")
    write_json_atomic(root / "comparison.json", report)
    write_json_atomic(root / "ledger_1000.json", ledger)
    write_text_atomic(root / "ledger_1000.csv", adjudication.historical._csv_text(ledger))
    for version in HISTORICAL:
        for kind in ("corrected", "regressed"):
            chosen = [
                r
                for r in ledger
                if r["scoring_eligibility"] == "INCLUDED"
                and (
                    (r["errors"][version] != "CORRECT" and r["errors"][VERSION] == "CORRECT")
                    if kind == "corrected"
                    else (r["errors"][version] == "CORRECT" and r["errors"][VERSION] != "CORRECT")
                )
            ]
            write_json_atomic(root / f"{kind}_vs_{version}.json", chosen)
    write_json_atomic(root / "candidate_predictions_1000.json", [projection(r) for r in results])
    text = [
        "# .33-exp3 完整 1,000 条开发集诊断\n",
        "exp3 为本次全链路新运行；.12 和 .33-exp1 为历史输出，不是同期重跑。参考、权重与五条排除不变，995 条计分，1,000 条纳入工程统计。旧 taxonomy 不参与版本主分数；这是混合开发参照，不是独立人工 holdout。\n",
        "| 版本 | 口径 | Precision | Recall | F1 | 主决策完全一致 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for version, score in report["scores"].items():
        for mode, name in (("weighted", "加权"), ("unweighted", "非加权")):
            values = [
                score[mode][k]
                for k in ("duplicate_precision", "duplicate_recall", "duplicate_f1", "primary_decision_exact")
            ]
            text.append(
                f"| {version} | {name} | "
                + " | ".join("N/A" if v is None else f"{100 * v:.2f}%" for v in values)
                + " |"
            )
    text += [
        "\n## 错误分布\n",
        "| 版本 | 误合并 | 明确 NO 的漏合并 | 方向错误 | 未决错误 | 工程缺失 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for version, score in report["scores"].items():
        text.append(
            f"| {version} | "
            + " | ".join(
                str(score["error_counts"].get(k, 0))
                for k in (
                    "OVER_GROUP",
                    "UNDER_GROUP_RESOLVED",
                    "DIRECTION_ONLY",
                    "UNRESOLVED",
                    "ENGINEERING_MISSING_OUTPUT",
                )
            )
            + " |"
        )
    text += ["\n参考为正的未决/缺失仍计入 recall 的漏判；缺失占位符不因碰巧匹配未决参照而得分。\n", "## 逐条得失\n"]
    for version, delta in report["paired_comparisons"].items():
        text.append(
            f"对比 {version}：修好 {delta['corrected_pairs']} 条（权重 {delta['corrected_weight']:.3f}），新增错误 {delta['regressed_pairs']} 条（权重 {delta['regressed_weight']:.3f}）。"
        )
    text += [
        "\n## 工程与限制\n",
        f"最终有效结果 {engineering['valid_pipeline_results']}/1000；工程失败 {len(failures)}；未决 {engineering['group_unresolved']}。外部请求 {len(attempts)}，HTTP 重试 {engineering['transport_retries']}，语义修复重试 0。",
        "不把合法未决算作已完成语义判断，也不把失败占位符算作 schema 成功。截断和 capped packet 沿用原路由；未改 prompt、证据策略或标签。",
        "各历史 reason cohort、confidence tier 的真实一致率，main→coverage→final 的净变化及同文组分歧见 comparison.json；该辅助分层不是新的 gold。",
        "旧基线的工程指标未重新测量。本次候选只有一遍，不声称跨遍稳定；历史对照也不能隔离模型服务随时间的变化。",
        "无 holdout、20k、release、MinHash 推断、commit 或 push。数值检查不是发布授权。",
        "\n逐案账本：ledger_1000.csv / ledger_1000.json；相对各旧版的修好与退化清单另存 corrected_vs_* / regressed_vs_*。",
    ]
    write_text_atomic(root / "REPORT.md", "\n".join(text) + "\n")
    pilot.reference.verify_freeze(root / "manifest.json")
    write_json_atomic(
        root / "complete.json",
        {
            "comparison_sha256": sha256_file(root / "comparison.json"),
            "population": 1000,
            "external_attempts": len(attempts),
            "full20k_admitted": False,
            "artifacts": {str(p): sha256_file(p) for p in root.iterdir() if p.is_file() and p.name != "complete.json"},
        },
    )
    return report


def unused_baseline_renderer(_payload: dict) -> list:
    raise RuntimeError("historical baselines must never be called in the fresh candidate run")


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    root = root.resolve()
    pilot.reference.verify_freeze(root / "manifest.json")
    manifest = adjudication.read(root / "manifest.json")
    require(
        manifest["version"] == VERSION and not (root / "started.json").exists(),
        "CONTEXT_FULL_STARTED",
        "frozen one-pass run only",
    )
    pilot.warmup()
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "CONTEXT_FULL_CREDENTIAL", "existing NVIDIA credential required")
    rows = adjudication.read(root / "panel_private.json")
    requests = _index(adjudication.read(root / "main_requests.json"), "frozen requests")
    write_json_atomic(
        root / "started.json",
        {
            "contract_digest": manifest["contract_digest"],
            "credential_persisted": False,
            "tokenizer_initialization": "SERIAL_BEFORE_WORKERS",
        },
    )
    profile = pilot.TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=180,
        max_external_attempts=8000,
    )
    with pilot.PacedRelayV2(
        profile=profile,
        logical_model="Qwen/Qwen3.8-27B-FP8",
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=160,
        expected_generation_parameters=pilot.GENERATION,
    ) as relay:
        relay.set_context(pilot.RelayContext(VERSION, 0, root / "transport_events.jsonl"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    pilot.execute_case,
                    root,
                    row,
                    "candidate",
                    1,
                    relay.endpoint,
                    requests.get(row["canonical_pair_id"]),
                    unused_baseline_renderer,
                )
                for row in rows
            ]
            failures = 0
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                failures += result["status"] != "VALID"
                progress = {
                    "completed": count,
                    "population": len(rows),
                    "engineering_failures": failures,
                    "last_review_id": result["review_id"],
                    "last_status": result["status"],
                }
                write_json_atomic(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    return export(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    require(
        args.command == "prepare" or args.env_file is not None,
        "CONTEXT_FULL_ARGUMENT",
        "explicit existing credential source",
    )
    result = prepare(args.root) if args.command == "prepare" else run(args.root, args.env_file)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "sources",
                    "artifacts",
                    "response_artifacts",
                    "cohorts",
                    "confidence_tiers",
                    "paired_comparisons",
                    "stage_scores",
                    "stage_effects",
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
