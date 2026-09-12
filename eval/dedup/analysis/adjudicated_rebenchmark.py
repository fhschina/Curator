# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Replay immutable primary predictions against a provenance-bearing development reference."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import bottleneck_audit as historical
from eval.dedup.analysis import reference_rebenchmark as reference
from eval.dedup.analysis.development_diagnostic import _index, classify_primary_error
from eval.dedup.analysis.judge_calibration import PRIMARY_FIELDS, _weight
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

SESSION = Path("/raid/hfang/ihb/runs/critic-five-hour-20260911T070052Z")
ADJUDICATION = SESSION / "user-policy-adjudication62-v1"
PRIOR_REFERENCE = Path("/raid/hfang/ihb/runs/reference-policy-v2-rebenchmark-v1/reference_policy_v2_partial_draft.csv")
BASELINE = "v0.6.2.12"
CANDIDATE = "v0.6.2.33-exp1"
REFERENCE_VERSION = "paragraph-subset-user-assisted-development-v1"
RATES = (*reference.RATES, "duplicate_f1")


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def primary(row: dict) -> dict:
    return reference.primary(row)


def project_cell(cell: dict) -> list[dict]:
    """Keep missing responses unresolved; do not silently drop deterministic bypasses."""
    _index(cell["pairs"], "saved cell")
    return [
        {
            "canonical_pair_id": p["canonical_pair_id"],
            **primary(p["primary"]),
            **({"metric_only_missing_output": True} if p["status"] == "ENGINEERING_FAILURE" else {}),
        }
        for p in cell["pairs"]
    ]


def revise_reference(labels: list[dict], updates: list[dict], excluded: set[str]) -> tuple[list[dict], list[dict]]:
    refs, applications = _index(labels, "reference"), _index(updates, "applications")
    require(set(applications).isdisjoint(excluded), "ADJUDICATION_OVERLAP", "an excluded case cannot get a label")
    require(set(applications) | excluded <= refs.keys(), "ADJUDICATION_MEMBERSHIP", "unknown reference case")
    result, changes = [], []
    for row in labels:
        pid = row["canonical_pair_id"]
        old = reference.primary(row, "human_")
        updated = deepcopy(row)
        application = applications.get(pid)
        decision = application["decision"] if application else old
        if application:
            require(
                set(decision) == set(PRIMARY_FIELDS)
                and set(decision.values()) <= {"YES", "NO"}
                and (decision["same_duplicate_group"] == "YES")
                == ("YES" in [decision["a_can_replace_b"], decision["b_can_replace_a"]]),
                "ADJUDICATION_DIRECTION",
                "explicit binary, direction-consistent primary decision required",
            )
            require(
                application["provenance"] in {"USER_CONFIRMED", "ASSISTANT_RULE_APPLICATION", "EXACT_INPUT_COMPANION"}
                and application["independent_blind_gold"] is False,
                "ADJUDICATION_PROVENANCE",
                "development labels must retain their actual source",
            )
        updated.update(
            reference_version=REFERENCE_VERSION,
            reference_status="MIXED_USER_ASSISTANT_PARTIAL_DEVELOPMENT_REFERENCE_NOT_INDEPENDENT_GOLD",
            comparison_scoring_eligibility="EXCLUDED" if pid in excluded else "INCLUDED",
            adjudication_provenance="USER_COMPARISON_EXCLUSION"
            if pid in excluded
            else application["provenance"]
            if application
            else "INHERITED_REFERENCE_NOT_REVALIDATED",
            adjudication_source=application["source"] if application else None,
            independent_adjudication=False,
            taxonomy_status="HISTORICAL_TAXONOMY_NOT_REVISED_OR_SCORED",
            prior_primary=old,
        )
        updated.update({f"human_{k}": None if pid in excluded else decision[k] for k in PRIMARY_FIELDS})
        if application and decision != old:
            changes.append(
                {
                    "canonical_pair_id": pid,
                    "review_id": row["review_id"],
                    "weight": _weight(row),
                    "before": old,
                    "after": decision,
                    "provenance": application["provenance"],
                    "source": application["source"],
                }
            )
        require(_weight(updated) == _weight(row), "ADJUDICATION_WEIGHT", "weights must remain unchanged")
        result.append(updated)
    return result, changes


def masked_score(labels: list[dict], predictions: list[dict], excluded: set[str]) -> dict:
    """Validate the original full join before excluding exactly the approved comparison cases."""
    refs, preds = _index(labels, "reference"), _index(predictions, "predictions")
    require(refs.keys() == preds.keys(), "ADJUDICATION_FULL_JOIN", "no missing or extra original predictions")
    require(excluded <= refs.keys(), "ADJUDICATION_MASK", "unknown excluded pair")
    selected = [r for r in labels if r["canonical_pair_id"] not in excluded]
    require(bool(selected), "ADJUDICATION_EMPTY", "empty comparison population")
    require(
        all(r.get("comparison_scoring_eligibility") != "EXCLUDED" for r in selected),
        "ADJUDICATION_MASK_REQUIRED",
        "the revised reference cannot be scored without its matching exclusion mask",
    )
    result = reference.score(selected, [preds[r["canonical_pair_id"]] for r in selected])
    result.update(
        original_rows=len(labels),
        scored_rows=len(selected),
        excluded_rows=len(excluded),
        excluded_weight=sum(_weight(refs[pid]) for pid in excluded),
        full_population_missing_output_pairs=sum(bool(p.get("metric_only_missing_output")) for p in predictions),
        full_population_group_unresolved=sum(p["same_duplicate_group"] == "UNRESOLVED" for p in predictions),
        interpretation="Same saved outputs; exact offline comparison mask; no inference or independent validation.",
    )
    return result


def delta(before: dict, after: dict) -> dict:
    return {
        mode: {
            k: None if before[mode][k] is None or after[mode][k] is None else 100 * (after[mode][k] - before[mode][k])
            for k in RATES
        }
        for mode in ("weighted", "unweighted")
    }


def compare(labels: list[dict], revised: list[dict], views: dict[str, list[dict]], excluded: set[str]) -> dict:
    before_digest = sha256_json(views)
    phases = {}
    for name, refs, mask in (
        ("prior_reference_full1000", labels, set()),
        ("prior_reference_masked995", labels, excluded),
        ("revised_reference_masked995", revised, excluded),
    ):
        phases[name] = {version: masked_score(refs, preds, mask) for version, preds in views.items()}
    effects = {
        version: {
            "exclusion_only_pp": delta(
                phases["prior_reference_full1000"][version], phases["prior_reference_masked995"][version]
            ),
            "reference_only_pp": delta(
                phases["prior_reference_masked995"][version], phases["revised_reference_masked995"][version]
            ),
        }
        for version in views
    }
    require(before_digest == sha256_json(views), "ADJUDICATION_PREDICTION_MUTATION", "predictions are immutable")
    return {"phases": phases, "reference_and_mask_effects": effects}


def companion_updates(checks: list[dict], updates: list[dict], panel: list[dict]) -> list[dict]:
    by_review = {r["review_id"]: r for r in panel}
    require(len(by_review) == len(panel), "ADJUDICATION_REVIEW_ID", "unique review IDs")
    by_pair = _index(updates, "applications")
    result = []
    for check in checks:
        seed, other = (by_review[check[k]] for k in ("in_scope_review_id", "companion_review_id"))
        a, b = (seed["payload"][f"document_{side}"]["text"] for side in ("a", "b"))
        c, d = (other["payload"][f"document_{side}"]["text"] for side in ("a", "b"))
        reverse = check["input_relation"] == "EXACT_A_B_REVERSE"
        require(
            check["input_relation"] in {"EXACT_A_B_REVERSE", "EXACT_SAME_A_B"},
            "ADJUDICATION_COMPANION",
            "exact match only",
        )
        require(
            (a, b) == ((d, c) if reverse else (c, d)), "ADJUDICATION_COMPANION", "complete input strings must match"
        )
        require(
            other["canonical_pair_id"] == check["companion_canonical_pair_id"],
            "ADJUDICATION_COMPANION",
            "bound pair ID",
        )
        require(
            other["canonical_pair_id"] not in by_pair,
            "ADJUDICATION_COMPANION",
            "companion must not overwrite an application",
        )
        decision = deepcopy(by_pair[seed["canonical_pair_id"]]["decision"])
        if reverse:
            decision["a_can_replace_b"], decision["b_can_replace_a"] = (
                decision["b_can_replace_a"],
                decision["a_can_replace_b"],
            )
        require(
            decision == check["companion_proposed_decision"],
            "ADJUDICATION_COMPANION",
            "reviewed orientation must agree",
        )
        result.append(
            {
                "canonical_pair_id": other["canonical_pair_id"],
                "decision": decision,
                "provenance": "EXACT_INPUT_COMPANION",
                "source": str(ADJUDICATION / "reference_consistency_round02_private.json"),
                "source_review_id": seed["review_id"],
                "independent_blind_gold": False,
            }
        )
    return result


def close_exact_companions(updates: list[dict], panel: list[dict], excluded: set[str]) -> list[dict]:
    """Propagate a reviewed decision only to complete, byte-identical document pairs."""
    packets = _index(panel, "panel")
    applications = _index(updates, "applications")
    groups = {}
    for pid, application in applications.items():
        packet = packets[pid]
        a, b = (packet["payload"][f"document_{s}"]["text"] for s in ("a", "b"))
        require(
            isinstance(a, str)
            and isinstance(b, str)
            and a
            and b
            and packet["payload"]["long_document_evidence"]["truncated"] is False,
            "ADJUDICATION_EXACT_SOURCE",
            "only complete, nonempty reviewed inputs can seed exact matching",
        )
        key = tuple(sorted((a, b)))
        normalized = deepcopy(application["decision"])
        if a > b:
            normalized["a_can_replace_b"], normalized["b_can_replace_a"] = (
                normalized["b_can_replace_a"],
                normalized["a_can_replace_b"],
            )
        if key in groups:
            require(groups[key][0] == normalized, "ADJUDICATION_EXACT_SOURCE", "reviewed seeds disagree")
        else:
            groups[key] = (normalized, application, packet["review_id"], (a, b))
    result = []
    for pid, packet in packets.items():
        if pid in applications or pid in excluded:
            continue
        a, b = (packet["payload"][f"document_{s}"]["text"] for s in ("a", "b"))
        if not isinstance(a, str) or not isinstance(b, str) or not a or not b:
            continue
        if packet["payload"].get("long_document_evidence", {}).get("truncated") is not False:
            continue
        key = tuple(sorted((a, b)))
        if key not in groups:
            continue
        normalized, seed, review_id, original = groups[key]
        decision = deepcopy(normalized)
        if a > b:
            decision["a_can_replace_b"], decision["b_can_replace_a"] = (
                decision["b_can_replace_a"],
                decision["a_can_replace_b"],
            )
        result.append(
            {
                "canonical_pair_id": pid,
                "decision": decision,
                "provenance": "EXACT_INPUT_COMPANION",
                "source": seed["source"],
                "source_review_id": review_id,
                "source_canonical_pair_id": seed["canonical_pair_id"],
                "input_relation": "EXACT_SAME_A_B" if original == (a, b) else "EXACT_A_B_REVERSE",
                "normalized_document_pair_sha256": sha256_json(list(key)),
                "independent_blind_gold": False,
            }
        )
    return result


def load_adjudication(panel: list[dict]) -> tuple[list[dict], dict, dict]:
    state = read(ADJUDICATION / "paragraph_subset_status_after_batch06.json")
    require(state["new_boundary_cases_pending"] == 0, "ADJUDICATION_PENDING", "finish declared boundary review first")
    index = {r["case_id"]: r for r in read(ADJUDICATION / "case_index_private.json")}
    decisions = read(ADJUDICATION / "decisions.json")
    event = read(ADJUDICATION / "confirmations/batch06_user_confirmation.json")
    require(
        sha256_file(ADJUDICATION / "decisions.json") == event["resulting_decisions_sha256"],
        "ADJUDICATION_FREEZE",
        "confirmation snapshot",
    )
    mask = read(ADJUDICATION / "comparison_exclusions_v4.json")
    require(
        decisions["active_scoring_exclusions_path"] == str(ADJUDICATION / "comparison_exclusions_v4.json"),
        "ADJUDICATION_MASK",
        "latest confirmed mask",
    )
    require(
        mask["apply_to_versions"] == "ALL_COMPARABLE_VERSIONS_IDENTICALLY",
        "ADJUDICATION_MASK",
        "same mask for both versions",
    )
    require(mask["extend_to_similar_cases_automatically"] is False, "ADJUDICATION_MASK", "no inferred exclusions")
    require(
        mask["runtime_boundary"]["consumer"] == "OFFLINE_VERSION_COMPARISON_REPORTING_ONLY"
        and all(v is False for k, v in mask["runtime_boundary"].items() if k != "consumer"),
        "ADJUDICATION_RUNTIME",
        "exclusions cannot affect Judge execution",
    )
    mask_ids = set(mask["excluded_canonical_pair_ids"])
    excluded_cases, user_cases, pending_cases = set(), set(), set()
    updates = []
    for row in decisions["decisions"]:
        case = row["case_id"]
        if row["status"] == "USER_CONFIRMED":
            user_cases.add(case)
            updates.append(
                {
                    "canonical_pair_id": index[case]["canonical_pair_id"],
                    "decision": primary(row),
                    "provenance": "USER_CONFIRMED",
                    "source": row["user_confirmation"]["record_path"],
                    "independent_blind_gold": False,
                }
            )
        elif row["status"] == "USER_EXCLUDED_FROM_SEMANTIC_SCORING":
            excluded_cases.add(case)
            require(
                all(row[k] is None for k in PRIMARY_FIELDS),
                "ADJUDICATION_EXCLUDED_LABEL",
                "excluded cases have no new label",
            )
        else:
            require(row["status"] == "PENDING_USER_ADJUDICATION", "ADJUDICATION_STATUS", "known status required")
            pending_cases.add(case)
    require(
        mask_ids == {index[c]["canonical_pair_id"] for c in excluded_cases},
        "ADJUDICATION_MASK",
        "exactly confirmed cases",
    )
    applications = []
    packets = _index(panel, "panel")
    for path in state["rule_application_ledgers"]:
        content = read(Path(path))
        require(
            sha256_file(content["policy_path"]) == content["policy_sha256"],
            "ADJUDICATION_POLICY",
            "frozen application policy",
        )
        for row in content["cases"]:
            require(
                row["case_specific_user_confirmation"] is False and row["independent_blind_gold"] is False,
                "ADJUDICATION_PROVENANCE",
                "assistant applications are not user gold",
            )
            require(
                sha256_file(row["original_text_path"]) == row["original_text_sha256"],
                "ADJUDICATION_EVIDENCE",
                "original text changed",
            )
            pid = index[row["case_id"]]["canonical_pair_id"]
            for e in row["evidence"]:
                raw = packets[pid]["payload"][f"document_{e['side'].lower()}"]["text"].encode("utf-8")
                require(
                    raw[e["start_byte"] : e["end_byte"]].decode("utf-8") == e["quote"],
                    "ADJUDICATION_EVIDENCE",
                    "byte-aligned quote",
                )
            applications.append(row["case_id"])
            updates.append(
                {
                    "canonical_pair_id": pid,
                    "decision": primary(row["proposed_decision"]),
                    "provenance": "ASSISTANT_RULE_APPLICATION",
                    "source": path,
                    "independent_blind_gold": False,
                }
            )
    require(
        len(applications) == len(set(applications)) and set(applications) == pending_cases,
        "ADJUDICATION_APPLICATIONS",
        "every pending case has exactly one assistant application",
    )
    require(
        (len(user_cases), len(excluded_cases), len(applications), len(index)) == (11, 5, 46, 62),
        "ADJUDICATION_COUNTS",
        "fixed reviewed population",
    )
    checks = read(ADJUDICATION / "reference_consistency_round02_private.json")
    companions = companion_updates(checks["checks"], updates, panel)
    require(len(companions) == 2, "ADJUDICATION_COMPANION", "only the two previously reviewed exact companions")
    additional = close_exact_companions(updates + companions, panel, mask_ids)
    return (
        updates + companions + additional,
        mask,
        {
            "user_confirmed": 11,
            "assistant_rule_applications": 46,
            "exact_input_companions": len(companions) + len(additional),
            "excluded": 5,
        },
    )


def report(summary: dict, scores: dict, output: Path) -> str:
    rows = [
        "# .12 与 .33-exp1：统一开发参照离线重算\n",
        "两个版本使用完全相同的 995 条计分样本和原始权重。原始输入仍为 1,000 条；仅在比较阶段排除第 9、10、12、14、16 条。没有新增模型调用、修改预测或发布 release。\n",
        "## 新参照下的结果\n",
        "| 版本 | 口径 | Precision | Recall | F1 | 主决策完全一致 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    current = scores["phases"]["revised_reference_masked995"]
    for version in (BASELINE, CANDIDATE):
        for mode, name in (("weighted", "加权"), ("unweighted", "非加权")):
            m = current[version][mode]
            values = [
                m[k] for k in ("duplicate_precision", "duplicate_recall", "duplicate_f1", "primary_decision_exact")
            ]
            rows.append(f"| {version} | {name} | " + " | ".join(f"{100 * v:.2f}%" for v in values) + " |")
    rows += [
        "\n## 参照与排除的影响，与版本变化分开\n",
        "| 版本 | 阶段 | 加权 P | 加权 R | 加权主决策一致 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for phase, name in [
        ("prior_reference_full1000", "旧参照／1000"),
        ("prior_reference_masked995", "旧参照／995，仅排除"),
        ("revised_reference_masked995", "新参照／995"),
    ]:
        for version in (BASELINE, CANDIDATE):
            m = scores["phases"][phase][version]["weighted"]
            rows.append(
                f"| {version} | {name} | {100 * m['duplicate_precision']:.2f}% | {100 * m['duplicate_recall']:.2f}% | {100 * m['primary_decision_exact']:.2f}% |"
            )
    rows += [
        "\n## 剩余主决策错误\n",
        "| 版本 | 误合并 | 明确判 NO 的漏合并 | 方向错误 | 未决错误 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for version in (BASELINE, CANDIDATE):
        errors = current[version]["error_counts"]
        rows.append(
            f"| {version} | "
            + " | ".join(
                str(errors.get(k, 0)) for k in ("OVER_GROUP", "UNDER_GROUP_RESOLVED", "DIRECTION_ONLY", "UNRESOLVED")
            )
            + " |"
        )
    rows += [
        "\n漏合并总数还包括参考为正的未决案例；未决没有从分母删除。错误分类中的未决数与预测为 UNRESOLVED 的总数不同：历史参考也可能为未决。",
        "\n## 参照来源及限制\n",
        f"本次纳入 11 条用户固定标签、46 条助手按确认口径复核、{summary['reference_provenance']['exact_input_companions']} 条同文／反向输入一致性映射；另 5 条不计分。实际改变主标签 {summary['reference_changes']} 条。其余条目保留旧参照，并非全部按新政策重新裁决。",
        "此前两条一致性映射是 H0053 → H0739（A/B 对调）、H0634 → H0177（同向）。全量检查又发现 H0140 的六条同文：H0106、H0185、H0279、H0292、H0414、H0831，统一按用户已确认的皮靴案例映射方向。没有增加样本，也没有将排除推广到类似案例。",
        "这是预测可见、围绕候选误合并集中复核后的混合开发参照，不是独立人工盲审或 holdout。参照与排除导致的涨分不是模型提升；即使本次达到数值门槛，也不代表 release gate 全部通过。",
        "\n## 输出及工程边界\n",
        "v0.6.2.33-exp1 是已有固定证据核验组合的实验别名，不是新主 Judge prompt。固定选取原 full1000 的第一次候选输出；第二次及 fresh-upstream 的主决策逐条一致，未择优挑选。主 Judge 和内容覆盖阶段仍复用原输出，不冒充全链路新运行。",
        "两份完整 1,000 条主决策投影、来源摘要及原文件哈希均保留。没有重验或豁免旧运行的 schema、证据、重试等工程数据，也不将此次读取成功当作新的 schema completion。工程来源及全量未决数量另见 engineering_preserved.json。",
        "只评分主决策和同组 P/R/F1。关系／material taxonomy 保留历史值但不计分；完整翻译的旧 MINOR 等标签不参与此次版本选择。\n",
        f"[逐案比较账本]({output / 'ledger_1000.csv'}) · "
        f"[参照变更]({output / 'reference_changes.json'}) · "
        f"[配对差异]({output / 'paired_comparison.json'}) · "
        f"[实验版本登记]({output / 'experiment_checkpoint.json'})\n",
    ]
    return "\n".join(rows)


def export(output: Path) -> dict:
    output = output.resolve()
    protected = (SESSION, PRIOR_REFERENCE.parent, Path(__file__).resolve().parents[3])
    require(
        all(output != p and output not in p.parents and p not in output.parents for p in protected),
        "ADJUDICATION_OUTPUT",
        "new sibling output root required",
    )
    if (output / "summary.json").exists():
        reference.verify_freeze(output / "summary.json")
        return read(output / "summary.json")
    require(not output.exists() or not any(output.iterdir()), "ADJUDICATION_OUTPUT_EXISTS", "refuse nonempty output")
    sources = {}
    for name in (
        "v4-subject-v2-full1000",
        "subject-proof-v4-full1000",
        "subject-proof-v4-fresh-upstream",
        "subject-proof-v4-fresh-upstream/verification",
    ):
        sources.update(reference.verify_freeze(SESSION / name / "manifest.json"))
        root = SESSION / name
        complete = read(root / "complete.json")
        require(
            complete["assessment_sha256"] == sha256_file(root / "assessment.json"),
            "ADJUDICATION_ASSESSMENT",
            "saved complete assessment",
        )
        for path in (root / "complete.json", root / "assessment.json"):
            sources[str(path)] = sha256_file(path)
        for path, digest in read(root / "assessment.json").get("response_artifacts", {}).items():
            require(sha256_file(path) == digest, "ADJUDICATION_RESPONSE", "saved response changed")
            sources[path] = digest
    for source in read(ADJUDICATION / "manifest.json")["sources"]:
        require(sha256_file(source["path"]) == source["sha256"], "ADJUDICATION_SOURCE", "adjudication source changed")
    for path in ADJUDICATION.rglob("*"):
        if path.is_file():
            sources[str(path)] = sha256_file(path)
    for path in (Path(__file__).resolve(), PRIOR_REFERENCE):
        sources[str(path)] = sha256_file(path)
    panel = read(SESSION / "v4-subject-v2-full1000/panel_private.json")
    labels = historical._read_csv(PRIOR_REFERENCE)
    require(len(panel) == len(labels) == 1000, "ADJUDICATION_POPULATION", "original 1000 pairs")
    require(
        _index(panel, "panel").keys() == _index(labels, "reference").keys(),
        "ADJUDICATION_FULL_JOIN",
        "identical panel membership",
    )
    base_cell = read(SESSION / "v4-subject-v2-full1000/assessment.json")["cells"]["1/saved_12_final"]
    proof = read(SESSION / "subject-proof-v4-full1000/assessment.json")
    candidate_cell = proof["cells"]["1/candidate"]
    views = {BASELINE: project_cell(base_cell), CANDIDATE: project_cell(candidate_cell)}
    require(
        views[BASELINE] == [{"canonical_pair_id": r["canonical_pair_id"], **primary(r["saved_final"])} for r in panel],
        "ADJUDICATION_BASELINE",
        "exact .12 saved final, not main-only or refreshed critic",
    )
    fresh = read(SESSION / "subject-proof-v4-fresh-upstream/verification/assessment.json")
    require(
        fresh["stage"] == "full1000_fresh_subject_conditional_proof_verification",
        "ADJUDICATION_FINAL_STAGE",
        "compare final verified decisions, not intermediate subject proposals",
    )
    for cell in (proof["cells"]["2/candidate"], fresh["cells"]["1/candidate"], fresh["cells"]["2/candidate"]):
        require(
            _index(project_cell(cell), "repeat") == _index(views[CANDIDATE], "candidate"),
            "ADJUDICATION_REPEAT",
            "do not select the best repeat",
        )
    for version, cell in ((BASELINE, base_cell), (CANDIDATE, candidate_cell)):
        observed = masked_score(labels, views[version], set())
        require(
            all(observed[k] == cell["scores"]["partial_draft"][k] for k in ("weighted", "unweighted", "error_counts")),
            "ADJUDICATION_PRIOR_SCORE",
            "reproduce exact prior scores first",
        )
    updates, mask, provenance = load_adjudication(panel)
    excluded = set(mask["excluded_canonical_pair_ids"])
    revised, changes = revise_reference(labels, updates, excluded)
    scored_labels = [r for r in revised if r["canonical_pair_id"] not in excluded]
    scored_payloads = [r for r in panel if r["canonical_pair_id"] not in excluded]
    conflicts = historical.exact_conflicts(scored_labels, scored_payloads)
    touched = {r["canonical_pair_id"] for r in updates}
    require(
        not any(m["canonical_pair_id"] in touched for c in conflicts for m in c["members"]),
        "ADJUDICATION_EXACT_CONFLICT",
        "no revised pair may disagree with an exact input companion",
    )
    scores = compare(labels, revised, views, excluded)
    current = scores["phases"]["revised_reference_masked995"]
    scores["paired_version_delta_pp"] = delta(current[BASELINE], current[CANDIDATE])
    paired = historical.transition_matrix(
        scored_labels,
        *[[r for r in views[v] if r["canonical_pair_id"] not in excluded] for v in (BASELINE, CANDIDATE)],
    )
    paired.pop("main_metrics")
    paired.pop("final_metrics")
    paired["interpretation"] = (
        "Different saved judge configurations, same revised reference and exact comparison mask; not a new online or independent experiment."
    )
    indices = {v: _index(rows, v) for v, rows in views.items()}
    revised_index = _index(revised, "revised reference")
    ledger = []
    for row in labels:
        pid = row["canonical_pair_id"]
        new = revised_index[pid]
        ledger.append(
            {
                "review_id": row["review_id"],
                "canonical_pair_id": pid,
                "weight": _weight(row),
                "scoring_eligibility": new["comparison_scoring_eligibility"],
                "reference_provenance": new["adjudication_provenance"],
                "prior_reference": reference.primary(row, "human_"),
                "revised_reference": reference.primary(new, "human_"),
                "predictions": {v: primary(ix[pid]) for v, ix in indices.items()},
                "prior_errors": {v: classify_primary_error(row, ix[pid]) for v, ix in indices.items()},
                "revised_errors": {
                    v: "EXCLUDED_NOT_SCORED" if pid in excluded else classify_primary_error(new, ix[pid])
                    for v, ix in indices.items()
                },
            }
        )
    engineering = {
        "scope": "ORIGINAL_FULL_1000_UNFILTERED_NOT_REVALIDATED_OR_RERUN",
        "online_calls_this_rebenchmark": 0,
        "historical_engineering_records_unchanged": True,
        "metrics_not_remeasured": mask["engineering_checks_retained"],
        "views": {
            v: {
                "original_rows": len(p),
                "group_unresolved": sum(r["same_duplicate_group"] == "UNRESOLVED" for r in p),
                "missing_output_sentinels": sum(bool(r.get("metric_only_missing_output")) for r in p),
            }
            for v, p in views.items()
        },
        "warning": "No new component failures is not evidence that inherited main-Judge engineering errors disappeared.",
    }
    checkpoint = {
        "version": CANDIDATE,
        "status": "EXPERIMENTAL_CHECKPOINT_NOT_RELEASE",
        "baseline_version": BASELINE,
        "fixed_main": "v0.6.2.12 original raw replay v6-route",
        "components": [
            "coverage critic v4",
            "subject-v1 proposals allowed by scope-v2",
            "dedup-critic-subject-proof-verifier-v4",
        ],
        "canonical_output_source": str(SESSION / "subject-proof-v4-full1000/assessment.json"),
        "canonical_cell": "1/candidate",
        "prediction_sha256": sha256_json(views[CANDIDATE]),
        "source_manifest_digest": read(SESSION / "subject-proof-v4-full1000/manifest.json")["contract_digest"],
        "prompt_version_registered": False,
        "runtime_defaults_changed": False,
        "note": "User-approved name for the already evaluated combination; not a new call, cache migration or end-to-end run.",
    }
    summary = {
        "schema_version": "adjudicated-rebenchmark-v1",
        "versions": [BASELINE, CANDIDATE],
        "reference_version": REFERENCE_VERSION,
        "reference_status": "PARTIAL_MIXED_DEVELOPMENT_NOT_INDEPENDENT_GOLD",
        "population": 1000,
        "scored_population": 995,
        "excluded_population": 5,
        "excluded_weight": sum(_weight(r) for r in labels if r["canonical_pair_id"] in excluded),
        "reference_provenance": provenance,
        "reference_changes": len(changes),
        "reference_changed_weight": sum(r["weight"] for r in changes),
        "historical_predictions_modified": False,
        "historical_scores_overwritten": False,
        "external_model_calls": 0,
        "release_eligible": False,
        "remaining_exact_reference_conflict_groups": len(conflicts),
        "prediction_digests": {v: sha256_json(p) for v, p in views.items()},
        "development_numeric_checks": {
            v: {
                "precision_75": current[v]["weighted"]["duplicate_precision"] >= 0.75,
                "recall_75": current[v]["weighted"]["duplicate_recall"] >= 0.75,
                "primary_exact_79": current[v]["weighted"]["primary_decision_exact"] >= 0.79,
            }
            for v in views
        },
    }
    require(
        summary["excluded_population"] == len(excluded) and summary["scored_population"] == len(scored_labels),
        "ADJUDICATION_POPULATION",
        "exact 1000/995/5 counts",
    )
    artifacts = {
        "scores.json": scores,
        "reference_changes.json": changes,
        "reference_applications.json": updates,
        "comparison_exclusions.json": mask,
        "paired_comparison.json": paired,
        "engineering_preserved.json": engineering,
        "experiment_checkpoint.json": checkpoint,
        "remaining_exact_reference_conflicts.json": conflicts,
    }
    for name, value in artifacts.items():
        write_json_atomic(output / name, value)
    for version, rows in views.items():
        write_json_atomic(output / f"{version}_primary_predictions_1000.json", rows)
    write_text_atomic(output / "reference_prior.csv", PRIOR_REFERENCE.read_text())
    write_text_atomic(output / "reference_revised_1000.csv", historical._csv_text(revised))
    write_text_atomic(output / "ledger_1000.csv", historical._csv_text(ledger))
    write_text_atomic(output / "REPORT.md", report(summary, scores, output))
    require(
        all(sha256_file(p) == h for p, h in sources.items()),
        "ADJUDICATION_SOURCE_MUTATION",
        "source files changed during replay",
    )
    summary.update(sources=sources, artifacts={str(p): sha256_file(p) for p in output.iterdir() if p.is_file()})
    summary["contract_digest"] = sha256_json(summary)
    write_json_atomic(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    result = export(parser.parse_args().output)
    print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "artifacts"}}, indent=2))
