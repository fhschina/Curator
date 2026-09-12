# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Read-only policy, abstention and exact-input audits of a frozen checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from eval.dedup.analysis import adjudicated_rebenchmark as benchmark
from eval.dedup.analysis.development_diagnostic import _index
from eval.dedup.judging import critic_intervention as main_adapter
from eval.dedup.judging import critic_subject_proof_verifier as verifier
from eval.dedup.judging import local_ndd
from eval.dedup.judging.payload import _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.judging.schema_v3 import validate_judge_output_v3
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

VERSION = "v0.6.2.33-exp1"
RESOURCES = Path(__file__).resolve().parents[1] / "resources/local_ndd"
PROTOCOL = Path(__file__).with_name("checkpoint_preflight_v1.md")
PRIMARY = ("a_can_replace_b", "b_can_replace_a", "same_duplicate_group")


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def primary(value: dict) -> dict:
    return {k: value[k] for k in PRIMARY}


def raw_primary(raw: dict) -> dict:
    a, b = (raw[k]["score"].upper() for k in PRIMARY[:2])
    group = "YES" if "YES" in (a, b) else "NO" if a == b == "NO" else "UNRESOLVED"
    return dict(zip(PRIMARY, (a, b, group), strict=True))


def normalized_primary(value: dict, a: str, b: str) -> dict:
    result = primary(value)
    if a > b:
        result["a_can_replace_b"], result["b_can_replace_a"] = result["b_can_replace_a"], result["a_can_replace_b"]
    return result


def main_trace(row: dict) -> dict:
    raw, payload = row["raw_main"], row["payload"]
    ledger, citations, issue = local_ndd._optional_v3_span_ledger(raw, payload)
    known = {s["span_id"] for s in payload["semantic_diff_evidence"]["spans"]}
    return {
        "raw_primary_not_validated_verdict": raw_primary(raw),
        "adapted_main": primary(row["main_public"]),
        "ledger": ledger,
        "citation_issue": issue,
        "unknown_citations": sorted(set(citations) - known),
        "unresolved_ledger_fields": sorted(k for k, v in ledger.items() if v == "UNRESOLVED"),
        "adapter_rules": [r for r in row["main_public"]["reason_codes"] if r.startswith("SPAN_LEDGER_RULE:")],
    }


def unresolved_inventory(panel: list[dict], final: list[dict], ledger: list[dict]) -> list[dict]:
    packets, predictions, references = (
        _index(rows, name) for rows, name in ((panel, "panel"), (final, "final"), (ledger, "ledger"))
    )
    require(packets.keys() == predictions.keys() == references.keys(), "PREFLIGHT_JOIN", "all original cases required")
    result = []
    for pid, row in packets.items():
        if predictions[pid]["same_duplicate_group"] != "UNRESOLVED":
            continue
        payload, ref = row["payload"], references[pid]
        diff = payload["semantic_diff_evidence"]
        trace = main_trace(row)
        if payload["long_document_evidence"]["truncated"]:
            cause = "TRUNCATED_NO_SPAN_INPUT"
        elif diff["status"] != "COMPLETE":
            cause = "DIFF_PACKET_LIMIT" if diff["status"] == "INCOMPLETE_LIMIT" else "OTHER_INCOMPLETE_PACKET"
        elif trace["citation_issue"]:
            cause = "INVALID_SPAN_CITATION"
        elif trace["unresolved_ledger_fields"]:
            cause = "AUXILIARY_LEDGER_UNRESOLVED"
        else:
            cause = "OTHER_REQUIRES_REVIEW"
        result.append(
            {
                "canonical_pair_id": pid,
                "review_id": row["review_id"],
                "observed_failure_category": cause,
                "truncated": payload["long_document_evidence"]["truncated"],
                "full_text_present_in_saved_payload": all(
                    isinstance(payload[f"document_{s}"]["text"], str) for s in ("a", "b")
                ),
                "diff_status": diff["status"],
                "span_counts_before_cap": diff["span_counts"],
                "rendered_span_count": len(diff["spans"]),
                "available_window_count": len(payload["long_document_evidence"]["windows"]),
                "main_reads_full_text_or_windows": False,
                "weight": float(ref["weight"]),
                "scoring_eligibility": ref["scoring_eligibility"],
                "reference_provenance": ref["reference_provenance"],
                "reference_primary": json.loads(ref["revised_reference"]),
                "reference_error": json.loads(ref["revised_errors"])[VERSION],
                "final_primary": primary(predictions[pid]),
                **trace,
            }
        )
    return result


def unresolved_summary(rows: list[dict], ledger: list[dict]) -> dict:
    scored = [r for r in ledger if r["scoring_eligibility"] == "INCLUDED"]
    denominator = sum(float(r["weight"]) for r in scored)
    positive_weight = sum(
        float(r["weight"]) for r in scored if json.loads(r["revised_reference"])["same_duplicate_group"] == "YES"
    )
    result = {}
    for category in sorted({r["observed_failure_category"] for r in rows}):
        selected = [r for r in rows if r["observed_failure_category"] == category]
        errors = [r for r in selected if r["scoring_eligibility"] == "INCLUDED" and r["reference_error"] != "CORRECT"]
        missed = [r for r in errors if r["reference_primary"]["same_duplicate_group"] == "YES"]
        result[category] = {
            "count": len(selected),
            "reference_errors": len(errors),
            "reference_error_weight": sum(r["weight"] for r in errors),
            "primary_exact_gap_pp": 100 * sum(r["weight"] for r in errors) / denominator,
            "missed_reference_positives": len(missed),
            "recall_gap_pp": 100 * sum(r["weight"] for r in missed) / positive_weight if positive_weight else None,
        }
    return {
        "categories": result,
        "interpretation": "Gap attribution under the frozen development reference, NOT achievable repair gains. Unresolved predictions are not false positives; any repair can change precision.",
    }


def exact_input_groups(panel: list[dict], coverage: list[dict], final: list[dict], template: str) -> list[dict]:
    packets, covered, predictions = (
        _index(rows, name) for rows, name in ((panel, "panel"), (coverage, "coverage"), (final, "final"))
    )
    require(
        packets.keys() == covered.keys() == predictions.keys(),
        "PREFLIGHT_JOIN",
        "full exact-input stage join required",
    )
    # Model prompts are plain text; HTML escaping would change the frozen input.
    render = Environment(undefined=StrictUndefined, autoescape=False).from_string(template)  # noqa: S701
    groups = defaultdict(list)
    for pid, row in packets.items():
        payload = row["payload"]
        a, b = (payload[f"document_{s}"]["text"] for s in ("a", "b"))
        if payload["long_document_evidence"]["truncated"] or not all(isinstance(t, str) and t for t in (a, b)):
            continue
        key = tuple(sorted((a, b)))
        trace = main_trace(row)
        stage_values = {
            "raw_directions": trace["raw_primary_not_validated_verdict"],
            "adapted_main": row["main_public"],
            "coverage": covered[pid]["main_public"],
            "final": predictions[pid],
        }
        groups[key].append(
            {
                "canonical_pair_id": pid,
                "review_id": row["review_id"],
                "payload_sha256": sha256_json(payload),
                "initial_pair_prompt_sha256": sha256_json(render.render(payload=payload, repair_feedback=None)),
                "a_chars": len(a),
                "b_chars": len(b),
                "normalized_stages": {k: normalized_primary(v, a, b) for k, v in stage_values.items()},
                "coverage_changed_primary": primary(row["main_public"]) != primary(covered[pid]["main_public"]),
                "subject_verification_changed_primary": primary(covered[pid]["main_public"])
                != primary(predictions[pid]),
                **trace,
            }
        )
    return [
        {
            "exact_text_pair_sha256": sha256_json(key),
            "case_count": len(items),
            "distinct_oriented_payloads": len({x["payload_sha256"] for x in items}),
            "stage_disagreement": {
                stage: len({sha256_json(x["normalized_stages"][stage]) for x in items}) > 1
                for stage in ("raw_directions", "adapted_main", "coverage", "final")
            },
            "cases": items,
        }
        for key, items in groups.items()
        if len(items) > 1
    ]


def synthetic_payload(a: str, b: str) -> dict:
    return {
        "document_a": {"text": a},
        "document_b": {"text": b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(a, b, truncated=False),
    }


def synthetic_raw(
    payload: dict,
    *,
    profiles: tuple[str, str] = ("substantive_main", "substantive_main"),
    deltas: tuple[str, str] = ("same_record_content_extension", "none"),
    basis: str = "verified_substantive_record",
    conflict: str = "none",
) -> dict:
    scores = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "no",
        "relation_type": "containment",
        "material_difference": "major",
        "primary_material_difference": "main_content_addition_deletion",
        "dominant_overlap_source": "main_content",
        "primary_risk_factor": "containment_asymmetry",
        "confidence_tier": "medium",
        "span_content_profile_a": profiles[0],
        "span_content_profile_b": profiles[1],
        "span_a_delta": deltas[0],
        "span_b_delta": deltas[1],
        "span_shared_basis": basis,
        "span_hard_conflict": conflict,
        "span_translation_status": "not_translation",
    }
    citations = " ".join(s["span_id"] for s in payload["semantic_diff_evidence"]["spans"])
    return {k: {"score": v, "reasoning": "Synthetic ledger fixture: " + citations} for k, v in scores.items()}


def contract_probes() -> list[dict]:
    """Synthetic ledger/validator checks do not claim to measure model comprehension."""
    shared = "Returns are accepted within 30 days."
    result = []
    cases = (
        (
            "policy_plus_product",
            "Product: Field cup\n\n" + shared,
            shared,
            {
                "profiles": ("substantive_main", "non_main_only"),
                "deltas": ("record_identity_role_or_state_change", "none"),
                "basis": "verified_equivalent_non_main_message",
            },
            ("YES", "NO"),
        ),
        (
            "independent_article_addition",
            shared + "\n\nOrchids grow in warm climates.",
            shared,
            {"deltas": ("other_substantive_content", "none")},
            ("YES", "NO"),
        ),
        (
            "proper_subset_with_navigation",
            shared + "\n\nNext page",
            shared,
            {"deltas": ("universal_ui_or_repetition", "none")},
            ("YES", "NO"),
        ),
        ("true_substantive_extension", shared + "\n\nReturn postage is prepaid.", shared, {}, ("YES", "NO")),
        (
            "incompatible_values",
            "Model K costs 10 dollars.",
            "Model K costs 20 dollars.",
            {
                "deltas": ("record_identity_role_or_state_change", "record_identity_role_or_state_change"),
                "conflict": "state_or_version",
            },
            ("NO", "NO"),
        ),
        (
            "two_sided_uncovered",
            shared + "\n\nApples are red.",
            shared + "\n\nPears are green.",
            {"deltas": ("other_substantive_content", "other_substantive_content")},
            ("NO", "NO"),
        ),
    )
    for name, a, b, options, expected in cases:
        payload = synthetic_payload(a, b)
        actual = main_adapter.main_decision(synthetic_raw(payload, **options), payload)
        validate_evidence_offsets(actual, payload)
        result.append(
            {
                "name": name,
                "expected_directions": list(expected),
                "actual_primary": primary(actual),
                "current_policy_compatible": [actual[k] for k in PRIMARY[:2]] == list(expected),
                "legacy_rules": [c for c in actual["reason_codes"] if c.startswith("SPAN_LEDGER_RULE:")],
                "online_calls": 0,
            }
        )
    payload = synthetic_payload(shared + "\n\nNext page", shared)
    value = main_adapter.main_decision(synthetic_raw(payload), payload)
    value.update(material_difference="MINOR", primary_material_difference="OTHER_MATERIAL")
    try:
        validate_judge_output_v3(value)
    except DedupEvaluationError as exc:
        result.append(
            {
                "name": "minor_nonmain_containment_schema",
                "current_policy_compatible": False,
                "error_code": exc.issue.code,
                "error_message": exc.issue.message,
                "online_calls": 0,
            }
        )
    else:
        result.append(
            {"name": "minor_nonmain_containment_schema", "current_policy_compatible": True, "online_calls": 0}
        )
    return result


def replay_final(proof_root: Path, expected: list[dict]) -> tuple[list[dict], dict]:
    rows = read(proof_root / "panel_private.json")
    targets = _index(expected, "expected final")
    require(_index(rows, "proof panel").keys() == targets.keys(), "PREFLIGHT_JOIN", "full final replay join")
    assessment_path = proof_root / "assessment.json"
    assessment = read(assessment_path)
    require(
        read(proof_root / "complete.json")["assessment_sha256"] == sha256_file(assessment_path),
        "PREFLIGHT_COMPLETION",
        "completion binds final assessment",
    )
    requests = {(r["repeat"], r["canonical_pair_id"]): r for r in read(proof_root / "requests_frozen.json")}
    outputs, sources = (
        [],
        {
            str(assessment_path): sha256_file(assessment_path),
            str(proof_root / "complete.json"): sha256_file(proof_root / "complete.json"),
        },
    )
    for row in rows:
        pid = row["canonical_pair_id"]
        value = None
        if verifier.route(row["main_public"], row["payload"]) == "VERIFY_FIXED_SUBJECT_VETO":
            path = proof_root / "responses" / f"1-candidate-{pid}.json"
            receipt = read(path)
            request = requests[(1, pid)]
            require(
                receipt["status"] == "VALID"
                and receipt["canonical_pair_id"] == pid
                and receipt["repeat"] == 1
                and receipt["arm"] == "candidate",
                "PREFLIGHT_RECEIPT",
                "valid exact receipt identity required",
            )
            require(
                receipt["request_sha256"] == request["request_sha256"] == sha256_json(request["body"]),
                "PREFLIGHT_REQUEST",
                "bound original request required",
            )
            require(
                assessment["response_artifacts"][str(path)] == sha256_file(path),
                "PREFLIGHT_RECEIPT_HASH",
                "saved assessment binds receipt",
            )
            choice = receipt["raw_response"]["choices"][0]
            require(
                choice["finish_reason"] == "stop" and choice["message"]["content"] == receipt["assistant_content"],
                "PREFLIGHT_RAW",
                "complete original response required",
            )
            value = json.loads(receipt["assistant_content"])
            sources[str(path)] = sha256_file(path)
        public, _ = verifier.apply_review(row["main_public"], row["payload"], value)
        if value is not None:
            require(public == receipt["public"], "PREFLIGHT_FINAL_REPLAY", "public verifier result changed")
        validate_judge_output_v3(public)
        validate_evidence_offsets(public, verifier.source_payload(row["payload"]))
        require(primary(public) == primary(targets[pid]), "PREFLIGHT_FINAL_REPLAY", "frozen final primary changed")
        outputs.append({"canonical_pair_id": pid, **public})
    return outputs, sources


def run(output: Path, rebenchmark: Path, session: Path) -> dict:
    output, rebenchmark, session = (p.resolve() for p in (output, rebenchmark, session))
    require(not output.exists(), "PREFLIGHT_OUTPUT", "a new, isolated audit root is required")
    baseline_root, proof_root = session / "v4-subject-v2-full1000", session / "subject-proof-v4-full1000"
    sources = {}
    for path in (rebenchmark / "summary.json", baseline_root / "manifest.json", proof_root / "manifest.json"):
        sources.update(benchmark.reference.verify_freeze(path))
    panel = read(baseline_root / "panel_private.json")
    coverage = read(proof_root / "panel_private.json")
    final = read(rebenchmark / f"{VERSION}_primary_predictions_1000.json")
    with (rebenchmark / "ledger_1000.csv").open() as stream:
        ledger = list(csv.DictReader(stream))
    require(
        len(panel) == len(coverage) == len(final) == len(ledger) == 1000,
        "PREFLIGHT_POPULATION",
        "original full 1,000-pair population required",
    )
    for row in panel:
        public = main_adapter.main_decision(row["raw_main"], row["payload"])
        require(public == row["main_public"], "PREFLIGHT_MAIN_REPLAY", "frozen main adapter replay changed")
        validate_judge_output_v3(public)
        validate_evidence_offsets(public, row["payload"])
    full_final, more = replay_final(proof_root, final)
    sources.update(more)
    unresolved = unresolved_inventory(panel, final, ledger)
    unresolved_totals = unresolved_summary(unresolved, ledger)
    template_path = RESOURCES / "hs_v06212_pair.jinja"
    template = template_path.read_text()
    require(
        "document_a" not in template and "document_b" not in template and "windows" not in template,
        "PREFLIGHT_PROMPT_SCOPE",
        "audit assumes this frozen span-only prompt",
    )
    groups = exact_input_groups(panel, coverage, final, template)
    probes = contract_probes()
    frozen_panel_ids = {r["canonical_pair_id"] for r in unresolved}
    frozen_panel_ids.update(
        r["canonical_pair_id"] for g in groups if g["stage_disagreement"]["final"] for r in g["cases"]
    )
    scoped_panel = [
        {
            "canonical_pair_id": r["canonical_pair_id"],
            "review_id": r["review_id"],
            "payload_sha256": sha256_json(r["payload"]),
        }
        for r in panel
        if r["canonical_pair_id"] in frozen_panel_ids
    ]
    result = {
        "schema_version": "dedup-checkpoint-preflight-v1",
        "version": VERSION,
        "status": "BLOCKED_POLICY_CONTRACT_ALIGNMENT"
        if any(not p["current_policy_compatible"] for p in probes)
        else "OFFLINE_CHECKS_COMPLETE_ONLINE_NOT_ASSESSED",
        "population": len(panel),
        "scored_population": sum(r["scoring_eligibility"] == "INCLUDED" for r in ledger),
        "external_model_calls": 0,
        "reference_changed": False,
        "historical_predictions_changed": False,
        "main_replay_exact": len(panel),
        "final_replay_exact": len(full_final),
        "schema_and_evidence_valid_final": len(full_final),
        "semantic_unresolved": len(unresolved),
        "unresolved_reference_errors": sum(
            r["reference_error"] != "CORRECT" and r["scoring_eligibility"] == "INCLUDED" for r in unresolved
        ),
        "unresolved_attribution": unresolved_totals,
        "exact_input_groups": len(groups),
        "exact_input_cases": sum(g["case_count"] for g in groups),
        "exact_input_final_disagreement_groups": sum(g["stage_disagreement"]["final"] for g in groups),
        "diagnostic_panel_population": len(scoped_panel),
        "synthetic_policy_mismatches": [p["name"] for p in probes if not p["current_policy_compatible"]],
        "online_expansion_allowed": False,
        "release_eligible": False,
        "interpretation": "Read-only structural preflight. Existing outputs and labels remain frozen. Schema validity is not policy correctness. Diagnostics are not a benchmark or independent holdout.",
    }
    report = [
        "# .33-exp1 上线前检查",
        "",
        "结论：暂不启动完整 1,000 条新运行或 20k。旧输出结构合法，但旧判断规则与已确认的严格包含口径不一致。",
        "",
        f"离线重放主 Judge {len(panel)}/{len(panel)}、最终组合 {len(full_final)}/{len(full_final)} 与冻结输出一致；全部最终输出通过 v3 schema 与精确证据位置验证。没有调用模型、改参考或改旧输出。",
        "",
        "## 27 条未决的来源",
        "",
        "下列是可观察的触发原因，不是已修复数量，也不是把原始方向恢复后的预期收益。",
        "",
        "| 原因 | 条数 | 参考错误数 | 主决策加权缺口（百分点） | recall 加权缺口（百分点） |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in unresolved_totals["categories"].items():
        report.append(
            f"| {name} | {data['count']} | {data['reference_errors']} | {data['primary_exact_gap_pp']:.4f} | {data['recall_gap_pp']:.4f} |"
        )
    report.extend(
        [
            "",
            "主 Judge 的 pair prompt 只展示片段清单，不展示完整正文或截断窗口。DIFF_PACKET_LIMIT 的完整原文虽然保存在 payload 中，模型看到的仍是达到每类 160 条上限的片段。TRUNCATED_NO_SPAN_INPUT 则没有可展示的片段；不能归因成模型读完长文仍不会判断。",
            "",
            "H0009：方向原本 no/no，但共享记录依据及冲突字段 unresolved，适配器全局未决；日志却只显示 SEMANTIC_DIFF_COMPLETE，原因描述不充分。H0362：引用不存在的 B114。H0679：声称角色冲突却没有双侧独有引用，且原始理由把仅增加标题当冲突；这是引用失败与旧政策偏差叠加，不能只补一个引用。",
            "",
            "## 完全相同输入的一致性",
            "",
            f"全量扫描发现 {len(groups)} 组重复文本输入（共 {sum(g['case_count'] for g in groups)} 条），其中 {sum(g['stage_disagreement']['final'] for g in groups)} 组最终方向在统一 A/B 方位后仍不一致。只按完整原文字节等同性分组，不合并近似文本、空文本或截断输入。",
            "",
            "皮靴案例 H0140 与六个 companion：同方向 payload 和初始 pair prompt 完全一致。四个 companion 的原始方向正确，却被 profile mismatch 规则改为 no/no；另外两个原始方向已经 no/no。七条均未被 coverage/subject 阶段改动。不能靠收窄 critic 修复这些继承的负例，也不应给这些 case ID 写特判。初始 prompt 一致不等于已证明全部历史 HTTP 请求/修复轮次完全相同；本报告不武断归因于随机性。",
            "",
            "## 口径与契约",
            "",
            "合成检查是适配器/契约复现，不是模型能力测验。当前不兼容项：",
            "",
            *[f"- {p['name']}" for p in probes if not p["current_policy_compatible"]],
            "",
            "旧 system、pair、YAML 及适配器仍限定同一具体记录／非正文不包含；普通导航增量仍可双向替代。旧 v3 又强制 containment=MAJOR/MAIN_CONTENT_ADDITION_DELETION。仅增加导航时，不能为了符合 schema 捏造重大主内容变化。",
            "",
            "## 下一步及冻结边界",
            "",
            f"已单独冻结 {len(scoped_panel)} 条机制诊断样本：全部未决＋所有最终方向不一致的同文组。它不是新的有代表性 benchmark；尚不含完整负例保护面板，不能据此选版本。配套修复顺序及在线准入条件见 protocol.md。",
            "",
            "保留 .33-exp1；新口径适配候选单独使用 .33-exp2，不覆盖历史 prompt/schema/cache。先对齐严格包含方向与差异大小的契约，再做同输出离线对照和完整负例保护；输入窗口/片段上限问题另做单变量处理。未完成前禁止由本报告启动在线扩大。",
            "",
            "开发参考仍包含助手规则应用和未复核继承标签，并非独立人工 gold；五条 comparison-only 排除不进入 Judge。400 条独立 holdout、SUT MinHash contract 及 20k 仍未通过准入。",
            "",
        ]
    )
    for path in (
        Path(__file__).resolve(),
        PROTOCOL,
        template_path,
        session / "user-policy-adjudication62-v1/paragraph_subset_policy_v2.md",
        Path(__file__).resolve().parents[3] / "tests/eval/dedup/test_checkpoint_preflight.py",
    ):
        sources[str(path)] = sha256_file(path)
    for filename, value in (
        ("unresolved_ledger.json", unresolved),
        ("exact_input_groups.json", groups),
        ("contract_probes.json", probes),
        (
            "mechanism_panel.json",
            {
                "status": "FROZEN_DIAGNOSTIC_NOT_VERSION_SELECTION_PANEL",
                "cases": scoped_panel,
                "references_in_model_payload": False,
            },
        ),
    ):
        write_json_atomic(output / filename, value)
    write_text_atomic(output / "REPORT.md", "\n".join(report))
    write_text_atomic(output / "protocol.md", PROTOCOL.read_text())
    require(
        all(sha256_file(p) == h for p, h in sources.items()), "PREFLIGHT_SOURCE_CHANGED", "inputs changed during audit"
    )
    result["sources"] = sources
    result["artifacts"] = {str(p): sha256_file(p) for p in output.iterdir()}
    result["contract_digest"] = sha256_json(result)
    write_json_atomic(output / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rebenchmark", type=Path, default=Path("/raid/hfang/ihb/runs/v06212-v06233-exp1-adjudicated-rebenchmark-v1")
    )
    parser.add_argument("--session", type=Path, default=benchmark.SESSION)
    args = parser.parse_args()
    result = run(args.output, args.rebenchmark, args.session)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"sources", "artifacts"}}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
