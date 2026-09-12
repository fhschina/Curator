# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Freeze, reproduce and audit exp1 without changing its semantic implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.dedup.analysis import adjudicated_rebenchmark as benchmark
from eval.dedup.analysis import checkpoint_preflight as preflight
from eval.dedup.analysis import exp1_reproduction_runtime as runtime
from eval.dedup.analysis import reference_rebenchmark as reference
from eval.dedup.analysis import retention_context_full1000 as comparison
from eval.dedup.analysis.development_diagnostic import _index
from eval.dedup.analysis.judge_calibration import _weight
from eval.dedup.judging.paced_relay import TransportProfile
from eval.dedup.judging.paced_relay_v2 import PacedRelayV2
from eval.dedup.judging.request_relay import RelayContext
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "exp1_reproduction.md"
REBENCHMARK = Path("/raid/hfang/ihb/runs/v06212-v06233-exp1-adjudicated-rebenchmark-v1")
ORIGIN = benchmark.SESSION / "v4-subject-v2-full1000"
PROOF_ORIGIN = benchmark.SESSION / "subject-proof-v4-full1000"
ORIGINAL_MAIN = benchmark.historical.FULL_ROOT
HISTORICAL = (benchmark.BASELINE, benchmark.CANDIDATE)
FRESH = runtime.VERSION + " [fresh-full1000]"
RATES = ("duplicate_precision", "duplicate_recall", "primary_decision_exact")


def legacy_request_hash(body: dict) -> str:
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def request_identity(rows: list[dict], requests: list[dict]) -> dict:
    panels = _index(rows, "original payloads")
    history = {
        json.loads(line)["request_hash"]
        for path in (ORIGINAL_MAIN / "runtime").rglob("outer-attempt-01.jsonl")
        for line in path.read_text().splitlines()
    }
    require(
        all(legacy_request_hash(r["body"]) in history for r in requests),
        "EXP1_MAIN_HISTORY",
        "exact historical main bodies",
    )
    render = runtime.coverage_renderer()
    counts = {"main": len(requests), "coverage": 0, "subject": 0, "verifier": 0}
    for old in benchmark.read(ORIGIN / "coverage/requests_frozen.json"):
        if old["arm"] != "candidate":
            continue
        payload = panels[old["canonical_pair_id"]]["payload"]
        require(
            runtime.body(render(payload), runtime.coverage.response_schema()) == old["body"],
            "EXP1_COVERAGE_HISTORY",
            "original coverage request",
        )
        counts["coverage"] += 1
    for old in benchmark.read(ORIGIN / "specialist/requests_frozen.json"):
        payload = panels[old["canonical_pair_id"]]["payload"]
        body = runtime.body(
            runtime.subject.messages(
                payload, (runtime.RESOURCES / "v06212_subject_binding_v1_system.txt").read_text()
            ),
            runtime.subject.response_schema(payload),
        )
        require(body == old["body"], "EXP1_SUBJECT_HISTORY", "original payload-specific subject schema")
        counts["subject"] += 1
    proof_rows = _index(benchmark.read(PROOF_ORIGIN / "panel_private.json"), "proof payloads")
    for old in benchmark.read(PROOF_ORIGIN / "requests_frozen.json"):
        payload = proof_rows[old["canonical_pair_id"]]["payload"]
        body = runtime.body(
            runtime.verifier.messages(
                payload, (runtime.RESOURCES / "v06212_subject_proof_verifier_v4_system.txt").read_text()
            ),
            runtime.verifier.response_schema(),
        )
        require(body == old["body"], "EXP1_VERIFIER_HISTORY", "original fixed-proposal verifier request")
        counts["verifier"] += 1
    return counts


def prepare(root: Path) -> dict:
    require(not root.exists(), "EXP1_ROOT", "new root; no historical cache reuse")
    sources = reference.verify_freeze(REBENCHMARK / "summary.json")
    sources.update(reference.verify_freeze(Path("/raid/hfang/ihb/runs/v06233-exp1-preflight-v1/summary.json")))
    original = benchmark.read(ORIGIN / "panel_private.json")
    rows = [{k: r[k] for k in ("canonical_pair_id", "review_id", "payload")} for r in original]
    labels = benchmark.historical._read_csv(REBENCHMARK / "reference_revised_1000.csv")
    require(len(rows) == len(labels) == 1000, "EXP1_POPULATION", "original complete 1000 only")
    require(_index(rows, "panel").keys() == _index(labels, "labels").keys(), "EXP1_JOIN", "exact population join")
    requests = [
        {"canonical_pair_id": r["canonical_pair_id"], "body": runtime.body(runtime.main_messages(r["payload"]))}
        for r in rows
    ]
    for r in requests:
        r["request_sha256"] = sha256_json(r["body"])
        r["legacy_request_hash"] = legacy_request_hash(r["body"])
    matches = request_identity(rows, requests)
    mask = benchmark.read(REBENCHMARK / "comparison_exclusions.json")
    excluded = set(mask["excluded_canonical_pair_ids"])
    require(len(excluded) == 5, "EXP1_MASK", "only five approved comparison exclusions")
    histories = {v: benchmark.read(REBENCHMARK / f"{v}_primary_predictions_1000.json") for v in HISTORICAL}
    old_scores = benchmark.read(REBENCHMARK / "scores.json")["phases"]["revised_reference_masked995"]
    require(
        all(benchmark.masked_score(labels, histories[v], excluded) == old_scores[v] for v in HISTORICAL),
        "EXP1_OLD_SCORE",
        "exact old scores",
    )
    source_paths = [Path(__file__), Path(runtime.__file__), PROTOCOL]
    source_paths += [
        HERE.parents[2] / f"tests/eval/dedup/test_{n}.py" for n in ("exp1_reproduction", "exp1_reproduction_runtime")
    ]
    source_paths += list((ORIGINAL_MAIN / "runtime").rglob("outer-attempt-01.jsonl"))
    source_paths += [ORIGINAL_MAIN / "run_manifest.json", ORIGINAL_MAIN / "execution_freeze.json"]
    source_paths += [
        ORIGIN / "coverage/requests_frozen.json",
        ORIGIN / "specialist/requests_frozen.json",
        PROOF_ORIGIN / "requests_frozen.json",
    ]
    from data_designer.engine.column_generators.utils import prompt_renderer
    from data_designer.engine.models import facade
    from data_designer.engine.models.recipes import response_recipes

    source_paths += [Path(m.__file__) for m in (facade, response_recipes, prompt_renderer)]
    sources.update({str(p.resolve()): sha256_file(p) for p in source_paths})
    write_json_atomic(root / "panel_private.json", rows)
    write_json_atomic(root / "main_requests.json", requests)
    write_json_atomic(root / "historical_predictions.json", histories)
    write_json_atomic(root / "comparison_exclusions.json", mask)
    write_text_atomic(root / "reference_revised_1000.csv", (REBENCHMARK / "reference_revised_1000.csv").read_text())
    write_text_atomic(root / "protocol.md", PROTOCOL.read_text())
    old = benchmark.read(PROOF_ORIGIN / "manifest.json")
    manifest = {
        "version": runtime.VERSION,
        "status": "FROZEN_ORIGINAL_COMPONENTS_FRESH_FULL1000_NOT_RELEASE",
        "population": 1000,
        "scored_population": 995,
        "repeats": 1,
        "model": old["model"],
        "endpoint": old["endpoint"],
        "generation": runtime.GENERATION,
        "historical_request_identity_matches": matches,
        "initial_main_feedback": "PRESERVE_HISTORICAL_DATAFRAME_NAN_RENDERING",
        "main_request_format": "ORIGINAL_CHAT_TEXT_BLOCKS_WITHOUT_RESPONSE_FORMAT",
        "later_request_format": "ORIGINAL_WITHOUT_UNIQUE_ITEMS_WITH_PAYLOAD_SPECIFIC_SUBJECT_ENUMS",
        "main_outer_attempts": 3,
        "main_parser_corrections_per_outer_attempt": 2,
        "critic_semantic_repairs": 0,
        "max_logical_calls": 12000,
        "max_external_attempts": 24000,
        "fresh_main_calls_planned": 1000,
        "deterministic_input_bypass_added": False,
        "transport": {
            "workers": 2,
            "min_interval_seconds": 2,
            "max_attempts": 2,
            "timeout_seconds": 600,
            "deadline_seconds": 640,
        },
        "transport_note": "Bounded low-concurrency collection; original main used concurrency64, later exp1 stages used concurrency2. Model body and semantic logic unchanged; this does not claim identical historical scheduling or server state.",
        "reproduction_point_margin": 0.03,
        "margin_note": "Predeclared diagnostic tolerance, not a statistical noninferiority test or a new release gate.",
        "old_cache_reused": False,
        "reference_changed": False,
        "known_policy_contract_mismatches_preserved": True,
        "independent_validation_admitted": False,
        "release_eligible": False,
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def projection(result: dict, stage: str = "final") -> dict:
    value = result["public"] if stage == "final" else result["components"].get(stage)
    missing = value is None or (stage == "final" and result["status"] != "VALID")
    return {
        "canonical_pair_id": result["canonical_pair_id"],
        **(runtime.unresolved_judge_output_v3() if missing else value),
        **({"metric_only_missing_output": True} if missing else {}),
    }


def assess(labels: list[dict], histories: dict, results: list[dict], excluded: set[str]) -> dict:
    require(set(histories) == set(HISTORICAL), "EXP1_HISTORY_JOIN", "both historical versions required")
    require(
        _index(labels, "labels").keys() == _index(results, "results").keys(), "EXP1_RESULT_JOIN", "no dropped failures"
    )
    views = {**histories, FRESH: [projection(r) for r in results]}
    scores = {v: benchmark.masked_score(labels, p, excluded) for v, p in views.items()}
    indices = {v: _index(p, v) for v, p in views.items()}
    selected = [r for r in labels if r["canonical_pair_id"] not in excluded]
    ledger = []
    for ref in labels:
        pid = ref["canonical_pair_id"]
        ledger.append(
            {
                "canonical_pair_id": pid,
                "review_id": ref["review_id"],
                "weight": _weight(ref),
                "scoring_eligibility": "EXCLUDED" if pid in excluded else "INCLUDED",
                "reference_provenance": ref.get("adjudication_provenance"),
                "historical_reason_cohort": ref.get("human_reason_code"),
                "reference": reference.primary(ref, "human_"),
                "predictions": {v: benchmark.primary(ix[pid]) for v, ix in indices.items()},
                "errors": {
                    v: "EXCLUDED_NOT_SCORED" if pid in excluded else comparison.error(ref, ix[pid])
                    for v, ix in indices.items()
                },
            }
        )
    select = lambda rows: [r for r in rows if r["canonical_pair_id"] not in excluded]  # noqa: E731
    paired = {v: comparison.transitions(selected, select(views[v]), select(views[FRESH])) for v in HISTORICAL}
    cohorts = {}
    for cohort in sorted({r.get("human_reason_code", "UNSPECIFIED") for r in selected}):
        refs = [r for r in selected if r.get("human_reason_code") == cohort]
        cohorts[cohort] = {
            v: reference.score(refs, [ix[r["canonical_pair_id"]] for r in refs]) for v, ix in indices.items()
        }
    stages = {s: [projection(r, s) for r in results] for s in ("main", "coverage", "final")}
    weighted, old = scores[FRESH]["weighted"], scores[benchmark.CANDIDATE]["weighted"]
    checks = {
        "weighted_precision_75": weighted["duplicate_precision"] is not None
        and weighted["duplicate_precision"] >= 0.75,
        "weighted_recall_75": weighted["duplicate_recall"] is not None and weighted["duplicate_recall"] >= 0.75,
        "weighted_primary_exact_79": weighted["primary_decision_exact"] >= 0.79,
        "over_group_max66": scores[FRESH]["error_counts"].get("OVER_GROUP", 0) <= 66,
    }
    stability = {k: weighted[k] is not None and weighted[k] >= old[k] - 0.03 for k in RATES}
    protected = {
        c: cohorts[c][FRESH]["weighted"]["primary_decision_exact"]
        >= cohorts[c][benchmark.CANDIDATE]["weighted"]["primary_decision_exact"] - 0.03
        for c in ("identity_slot", "meaningful_addition", "translation")
        if c in cohorts
    }
    return {
        "scores": scores,
        "paired_comparisons": paired,
        "ledger": ledger,
        "cohorts": cohorts,
        "development_numeric_checks": checks,
        "reproduction_point_checks": stability,
        "protected_cohort_checks": protected,
        "stage_scores": {s: benchmark.masked_score(labels, p, excluded) for s, p in stages.items()},
        "stage_effects": {
            f"{a}_to_{b}": comparison.transitions(selected, select(stages[a]), select(stages[b]))
            for a, b in (("main", "coverage"), ("coverage", "final"), ("main", "final"))
        },
        "release_eligible": False,
        "full20k_admitted": False,
    }


def audit(root: Path, rows: list[dict], results: list[dict]) -> dict:
    reference.verify_freeze(root / "manifest.json")
    requests = _index(benchmark.read(root / "main_requests.json"), "initial requests")
    require(
        {p.stem for p in (root / "results").glob("*.json")} == {r["canonical_pair_id"] for r in rows},
        "EXP1_AUDIT_POPULATION",
        "all results and no extras",
    )
    seen = set()
    valid = 0
    for row, result in zip(rows, results, strict=True):
        pid, payload = row["canonical_pair_id"], row["payload"]
        require(
            result["canonical_pair_id"] == pid and result["review_id"] == row["review_id"],
            "EXP1_AUDIT_ID",
            "exact pair binding",
        )
        raw = {}
        for stage in result["stages"]:
            name = pid + "-" + stage["stage"] + ".json"
            require(name not in seen, "EXP1_DUPLICATE_STAGE", "unique saved calls")
            seen.add(name)
            request, receipt = (benchmark.read(root / folder / name) for folder in ("requests", "responses"))
            require(
                request["request_sha256"] == receipt["request_sha256"] == sha256_json(request["body"])
                and stage["response_sha256"] == sha256_file(root / "responses" / name),
                "EXP1_AUDIT_RECEIPT",
                "bound original received response",
            )
            if stage["stage"] == "main-01-01":
                require(
                    request == {k: requests[pid][k] for k in ("body", "request_sha256")},
                    "EXP1_AUDIT_INITIAL",
                    "exact initial request",
                )
            if receipt["status"] == "RECEIVED":
                raw[stage["stage"]] = receipt["raw_response"]["choices"][0]["message"]["content"]
        if result["status"] != "VALID":
            require(
                set(benchmark.primary(result["public"]).values()) == {"UNRESOLVED"},
                "EXP1_AUDIT_FAILURE",
                "failure stays unresolved",
            )
            continue
        last_main = [s["stage"] for s in result["stages"] if s["stage"].startswith("main-")][-1]
        parsed = runtime.main_toolchain()[1].parse(raw[last_main]).model_dump()
        require(parsed == result["raw_main"], "EXP1_AUDIT_MAIN_RAW", "original native parser only")
        public = runtime.common.critic.main_decision(parsed, payload)
        require(public == result["components"]["main"], "EXP1_AUDIT_MAIN", "same main projection")
        expected = []
        if runtime.common.critic.route(public, payload) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            expected.append("coverage")
            public, _ = runtime.coverage.apply_review(public, payload, runtime.common.strict_json(raw["coverage"]))
        require(public == result["components"]["coverage"], "EXP1_AUDIT_COVERAGE", "same coverage projection")
        if runtime.scope.route(public, payload) == "REVIEW_BILATERAL_SUBJECTS":
            expected.append("subject")
            proposal = runtime.common.strict_json(raw["subject"])
            bound = {**payload, "subject_proposal": proposal}
            if runtime.verifier.route(public, bound) == "VERIFY_FIXED_SUBJECT_VETO":
                expected.append("verifier")
                public, _ = runtime.verifier.apply_review(public, bound, runtime.common.strict_json(raw["verifier"]))
        actual = [s["stage"] for s in result["stages"] if not s["stage"].startswith("main-")]
        require(
            actual == expected and public == result["public"], "EXP1_AUDIT_FINAL", "exact routes and final projection"
        )
        runtime.validate_evidence_offsets(public, payload)
        valid += 1
    require(
        seen
        == {p.name for p in (root / "requests").glob("*.json")}
        == {p.name for p in (root / "responses").glob("*.json")},
        "EXP1_AUDIT_ORPHANS",
        "every receipt accounted for",
    )
    return {
        "valid_pipeline_replays": valid,
        "failure_placeholders": len(results) - valid,
        "bound_calls": len(seen),
        "additional_model_calls": 0,
    }


def export(root: Path) -> dict:
    rows = benchmark.read(root / "panel_private.json")
    results = [benchmark.read(root / "results" / (r["canonical_pair_id"] + ".json")) for r in rows]
    proof = audit(root, rows, results)
    labels = benchmark.historical._read_csv(root / "reference_revised_1000.csv")
    excluded = set(benchmark.read(root / "comparison_exclusions.json")["excluded_canonical_pair_ids"])
    report = assess(labels, benchmark.read(root / "historical_predictions.json"), results, excluded)
    events = [json.loads(line) for line in (root / "transport_events.jsonl").read_text().splitlines()]
    external = [e for e in events if e.get("external_request")]
    retried = [r for r in results if len(r["main_attempts"]) > 1 or any(a["requests"] > 1 for a in r["main_attempts"])]
    failures = [r for r in results if r["status"] != "VALID"]
    groups = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        p = row["payload"]
        a, b = (p[f"document_{s}"]["text"] for s in ("a", "b"))
        if a and b and not p["long_document_evidence"]["truncated"]:
            groups[sha256_json(sorted((a, b)))].append(
                {"review_id": row["review_id"], "primary": preflight.normalized_primary(result["public"], a, b)}
            )
    report["engineering"] = {
        "population": len(results),
        "valid_pipeline_results": len(results) - len(failures),
        "engineering_failures": len(failures),
        "failures": [{"review_id": r["review_id"], "error_code": r["error_code"]} for r in failures],
        "failure_codes": dict(Counter(r["error_code"] for r in failures)),
        "group_unresolved": sum(r["public"]["same_duplicate_group"] == "UNRESOLVED" for r in results),
        "schema_completion": (len(results) - len(failures)) / len(results),
        "stage_calls": dict(Counter(s["stage"].split("-")[0] for r in results for s in r["stages"])),
        "semantic_retried_pairs": len(retried),
        "semantic_retry_rate": len(retried) / len(results),
        "external_attempts": len(external),
        "transport_retries": sum(e.get("upstream_attempt", 1) > 1 for e in external),
        "http_statuses": dict(Counter(str(e.get("http_status")) for e in external)),
        "duplicate_text_groups": sum(len(g) > 1 for g in groups.values()),
        "exact_input_disagreement_groups": [
            g for g in groups.values() if len({sha256_json(r["primary"]) for r in g}) > 1
        ],
    }
    report["offline_audit"] = proof
    checks = [
        *report["development_numeric_checks"].values(),
        *report["reproduction_point_checks"].values(),
        *report["protected_cohort_checks"].values(),
        not failures,
        len(retried) / len(results) <= 0.01,
    ]
    report["reproduction_checks_passed"] = all(checks)
    report["independent_validation_preparation_admitted"] = all(checks)
    report["independent_validation_passed"] = False
    ledger = report.pop("ledger")
    write_json_atomic(root / "comparison.json", report)
    write_json_atomic(root / "ledger_1000.json", ledger)
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
    writer.writeheader()
    writer.writerows(
        {k: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v for k, v in r.items()} for r in ledger
    )
    write_text_atomic(root / "ledger_1000.csv", stream.getvalue())
    write_json_atomic(root / "fresh_predictions_1000.json", [projection(r) for r in results])
    lines = [
        "# .33-exp1 原样全链路复现",
        "",
        "原始 1,000 对、995 条计分；原参考、权重和五条不计分不变。历史输出仅作比较，所有主 Judge 和触发的后续阶段重新调用。不是独立验证或发布。",
        "",
        "| 版本 | 口径 | Precision | Recall | 主决策一致 |",
        "|---|---|---:|---:|---:|",
    ]
    for version, score in report["scores"].items():
        for mode in ("weighted", "unweighted"):
            values = ["N/A" if score[mode][k] is None else f"{100 * score[mode][k]:.2f}%" for k in RATES]
            lines.append(f"| {version} | {mode} | " + " | ".join(values) + " |")
    e = report["engineering"]
    lines += [
        "",
        f"有效管线结果 {e['valid_pipeline_results']}/1000；工程失败 {e['engineering_failures']}；未决 {e['group_unresolved']}；语义重试样本 {e['semantic_retried_pairs']}；传输重试 {e['transport_retries']}。",
        "",
        f"预先声明的复现检查通过：{report['reproduction_checks_passed']}。这只是能否进入独立验证准备的诊断，不是正式统计非劣检验。",
        "",
        "旧 dataframe 缺失反馈渲染为 nan 的初始提示也原样保留；1,000 条初始请求与历史摘要逐条匹配。保留旧判断规则和旧 schema，不引入 exp2/exp3 的规则、输出格式、输入绕过或宽松校验。低并发限速与旧主 Judge 的调度不同，不宣称旧服务状态或跨遍稳定性相同。",
        "",
        "保存结果逐阶段离线重放、证据对齐及请求/响应绑定通过后才生成此报告。失败不从分母删除，失败占位符不得获得未决匹配分。",
        "",
        "独立 holdout 尚未评审；本报告不启动 20k、不发布、不修改参考标签。",
    ]
    write_text_atomic(root / "REPORT.md", "\n".join(lines) + "\n")
    reference.verify_freeze(root / "manifest.json")
    write_json_atomic(
        root / "complete.json",
        {
            "comparison_sha256": sha256_file(root / "comparison.json"),
            "population": len(results),
            "reproduction_checks_passed": report["reproduction_checks_passed"],
            "artifacts": {
                str(p): sha256_file(p)
                for folder in (root, root / "results", root / "requests", root / "responses")
                for p in folder.iterdir()
                if p.is_file()
            },
        },
    )
    return report


def run(root: Path, env_file: Path) -> dict:
    from eval.dedup.cli import _load_repository_env

    reference.verify_freeze(root / "manifest.json")
    manifest = benchmark.read(root / "manifest.json")
    require(
        manifest["version"] == runtime.VERSION and not (root / "started.json").exists(),
        "EXP1_RUN_FROZEN",
        "one fresh run only",
    )
    _load_repository_env(env_file)
    credential = os.environ.get("NVIDIA_API_KEY", "").strip()
    require(bool(credential), "EXP1_CREDENTIAL", "existing NVIDIA credential required")
    runtime.main_toolchain()
    rows = benchmark.read(root / "panel_private.json")
    frozen = _index(benchmark.read(root / "main_requests.json"), "main requests")
    render = runtime.coverage_renderer()
    write_json_atomic(
        root / "started.json",
        {
            "contract_digest": manifest["contract_digest"],
            "credential_source": str(env_file),
            "credential_persisted": False,
        },
    )
    profile = TransportProfile(
        min_interval_seconds=2,
        max_in_flight=2,
        max_attempts=2,
        retry_base_seconds=10,
        retry_cap_seconds=40,
        request_deadline_seconds=640,
        max_external_attempts=manifest["max_external_attempts"],
    )
    with PacedRelayV2(
        profile=profile,
        logical_model=runtime.LOGICAL_MODEL,
        upstream_base_url=manifest["endpoint"],
        upstream_model=manifest["model"],
        upstream_api_key=credential,
        timeout_seconds=600,
        expected_generation_parameters=runtime.GENERATION,
    ) as relay:
        relay.set_context(RelayContext(runtime.VERSION + "-reproduction", 0, root / "transport_events.jsonl"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(runtime.execute_case, root, row, relay.endpoint, frozen[row["canonical_pair_id"]], render)
                for row in rows
            ]
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                progress = {
                    "completed": count,
                    "population": len(rows),
                    "review_id": result["review_id"],
                    "status": result["status"],
                }
                write_json_atomic(root / "progress_snapshots" / f"{count:04d}.json", progress)
                print(json.dumps(progress), flush=True)
    return export(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare(root)
    elif args.command == "summarize":
        result = export(root)
    else:
        require(args.env_file is not None, "EXP1_ENV_FILE", "explicit existing credential source")
        result = run(root, args.env_file)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "version",
                    "population",
                    "historical_request_identity_matches",
                    "reproduction_checks_passed",
                    "engineering",
                )
                if k in result
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
