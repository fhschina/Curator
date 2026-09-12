# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Same-response proof-validation replay; absent dependent responses never become predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import retention_v4_experiment as pilot
from eval.dedup.judging import retention_v4_proof as candidate
from eval.dedup.validation import (
    DedupEvaluationError,
    require,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)

VERSION = "v0.6.2.33-exp2-proof1-offline"


def replay_case(row: dict, repeat: int, old: dict, load_stage: Callable[[str], dict]) -> dict:
    payload = row["payload"]
    result = deepcopy(old)
    result.update(status="VALID", components={}, replay_version=VERSION)
    result.pop("error_code", None)
    try:
        value = load_stage("main") if candidate.input_route(payload) == "MODEL_REVIEW" else None
        public = candidate.adapt_main(value, payload)
        result["components"]["main"] = deepcopy(public)
        if candidate.critic_route(public, payload) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            public, _ = candidate.apply_critic(public, payload, load_stage("coverage"))
        result["components"]["coverage"] = deepcopy(public)
        if candidate.subject_route(public, payload) == "REVIEW_BILATERAL_SUBJECTS":
            proposal = load_stage("subject")
            if candidate.subject_proof(public, payload, proposal) is not None:
                public, _ = candidate.apply_subject_verification(public, payload, proposal, load_stage("verifier"))
        candidate.previous.validate_judge_output_v4(public)
        candidate.previous.validate_evidence_offsets(public, payload)
        result["public"] = public
    except DedupEvaluationError as exc:
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=exc.issue.code,
            public=pilot.unresolved_judge_output_v4("OFFLINE_REPLAY_MISSING_OR_INVALID_PROOF"),
        )
    require(
        result["canonical_pair_id"] == row["canonical_pair_id"] and result["repeat"] == repeat,
        "RETENTION_PROOF_REPLAY_ID",
        "same original pair and repeat required",
    )
    return result


def replay(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    require(not output.exists(), "RETENTION_PROOF_REPLAY_ROOT", "new offline root required")
    sources = pilot.reference.verify_freeze(source / "manifest.json")
    completion = pilot.preflight.read(source / "complete.json")
    assessment_path = source / "assessment.json"
    require(
        completion["assessment_sha256"] == sha256_file(assessment_path),
        "RETENTION_PROOF_REPLAY_COMPLETION",
        "completed original pilot must be hash-bound",
    )
    old_report = pilot.preflight.read(assessment_path)
    require(
        all(sha256_file(p) == h for p, h in old_report["response_artifacts"].items()),
        "RETENTION_PROOF_REPLAY_RECEIPTS",
        "original request/response/result artifacts changed",
    )
    sources.update(old_report["response_artifacts"])
    rows = pilot.preflight.read(source / "panel_private.json")
    results, changes = [], []
    for repeat in (1, 2):
        for row in rows:
            pid = row["canonical_pair_id"]
            results.append(pilot.preflight.read(source / "results" / f"{repeat}-baseline-{pid}.json"))
            old = pilot.preflight.read(source / "results" / f"{repeat}-candidate-{pid}.json")

            def load_stage(stage: str, *, selected_repeat: int = repeat, selected_pid: str = pid) -> dict:
                key = f"{selected_repeat}-candidate-{selected_pid}-{stage}.json"
                path = source / "responses" / key
                require(
                    path.exists(),
                    "RETENTION_PROOF_REPLAY_MISSING_STAGE",
                    "no inventing or borrowing a missing dependent stage",
                )
                receipt = pilot.preflight.read(path)
                request = pilot.preflight.read(source / "requests" / key)
                require(
                    receipt["status"] == "RECEIVED"
                    and receipt["request_sha256"] == request["request_sha256"] == sha256_json(request["body"]),
                    "RETENTION_PROOF_REPLAY_REQUEST",
                    "exact bound original response required",
                )
                choice = receipt["raw_response"]["choices"][0]
                require(
                    choice["finish_reason"] == "stop", "RETENTION_PROOF_REPLAY_FINISH", "complete response required"
                )
                value = pilot.common.strict_json(choice["message"]["content"])
                require(
                    value == receipt["parsed"],
                    "RETENTION_PROOF_REPLAY_RAW",
                    "parsed fields must match original response",
                )
                return value

            updated = replay_case(row, repeat, old, load_stage)
            results.append(updated)
            if (
                pilot.preflight.primary(old["public"]) != pilot.preflight.primary(updated["public"])
                or old["status"] != updated["status"]
            ):
                changes.append(
                    {
                        "review_id": row["review_id"],
                        "repeat": repeat,
                        "old_status": old["status"],
                        "new_status": updated["status"],
                        "old_primary": pilot.preflight.primary(old["public"]),
                        "new_primary": pilot.preflight.primary(updated["public"]),
                    }
                )
    report = pilot.assessment(rows, results)
    report.update(
        version=VERSION,
        scope="SAME_SAVED_RESPONSES_PROOF_VALIDATION_ONLY_NOT_FRESH_ONLINE_RESULT",
        external_model_calls=0,
        source_pilot=str(source),
        changes=changes,
        startup_failure_not_filled=True,
        reference_changed=False,
    )
    write_json_atomic(output / "results.json", results)
    write_json_atomic(output / "assessment.json", report)
    text = """# .33-exp2 实现及小实验复核

已实现新的输出契约、主 Judge、critic、收窄后的 subject/verifier 适配和独立版本注册。
旧 v0–v3 读取路径、.33-exp1 原始输出及历史评分保持不变。

## 已完成的在线实验

48 条（36 条机制诊断＋12 条合成保护），两个版本、各两遍；145 次模型调用，
全部 HTTP 200，0 网络重试。需要模型判断的主阶段实际调用 91 次；另一次在
发请求之前因分词器并发首次加载失败。两侧均明确绕过原有的 24 条不完整输入。
这不是完整 1,000 条评估，合成样本也不是独立人工 gold。

新候选对 7 条“皮靴标题＋完整相同政策”在两遍中均给出正确单向替代，并将
增量如实标为 MINOR；基线每遍仅 3/7 正确。两个已发现的同文组在候选两遍
中都保持组内一致。普通导航、独立文章、字段补充和完整翻译的对应合成用例
也得到预期方向。

## 问题与本轮修复

1. 首次并发加载分词器可复现 ImportError。新增独立执行入口，在工作线程
   启动前串行预热；冷进程并发测试通过。原失败未删除，也没有补入一次
   回答冒充原本成功的结果。本轮未执行修复后的整轮新调用。
2. 9 个候选结果被过严的证据位置规则拒绝：独有 SKU／主体／否定前缀已在
   loss 字段提供，context 字段选了共享命题，却被要求在 context 再出现一次
   独有值。独立 proof1 修订接受原回答中已提供的独有值和双侧上下文组合；
   不换 ID、不生成新引文、不读参照改结果，也不放宽冲突必须有真实独有证据
   的要求。完整重放同一批保存回答后，这 9 个结果均恢复为有证据的 no/no。
3. S09（否定引文）仍有真实语义问题。第一遍主 Judge 错判，critic 识别了否定，
   其原证据经上述验证修正后可以恢复正确 no/no；第二遍主 Judge 和 critic 的
   结构化字段仍输出无冲突、双向替代。第二遍 critic 的说明文字又承认否定
   关系，并纠结 B 没有独有片段。不能用说明文字偷偷覆盖其正式字段。

## 结果与准入

proof1 离线重放中，7 条标题方向保护两遍均 7/7；12 条合成保护分别 12/12、
11/12。尚有启动失败造成的一条缺失，以及第二遍否定引文误判。未通过全部
预先声明的准入条件，未启动新的完整 1,000 条或 20k，也没有发布新总体分数。

下一步应单独处理“共享命题被上下文否定”的表达：独有增量证据与上下文冲突
证据不能互相绑死，不能因 B 无独有片段而被迫放弃冲突。保留本次全部输出和
对照结果，先固定这一项修订及非冲突引文保护，再做新的小批配对验证。不要
同时调整长文输入、模型或评分口径。H0130/H0344 的继承标签争议仍单列，
没有新增比较排除项。
"""
    write_text_atomic(output / "REPORT.md", text)
    for path in (
        Path(__file__).resolve(),
        Path(candidate.__file__).resolve(),
        Path(__file__).resolve().parents[3] / "tests/eval/dedup/test_retention_v4_proof_replay.py",
        Path(__file__).resolve().parents[3] / "tests/eval/dedup/test_retention_v4_proof.py",
        source / "manifest.json",
        source / "assessment.json",
        source / "complete.json",
    ):
        sources[str(path)] = sha256_file(path)
    manifest = {
        "version": VERSION,
        "external_model_calls": 0,
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in output.iterdir()},
        "release_eligible": False,
        "full1000_admitted": False,
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(output / "manifest.json", manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = replay(args.source, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
