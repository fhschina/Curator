# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Prediction-hidden source packets for unresolved critic policy boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval.dedup.analysis import critic_selected_experiment as selected
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic

BOUNDARIES = {
    "H0093": "具名数据处理同意书（目的、范围、处理方式、期限）是否属于实质 X？",
    "H0195": "孤立的产品/归档栏目是否已经证明实际页面角色冲突？",
    "H0352": "信用披露与计算示例是否构成可独立保留的完整内容 X？",
    "H0701": "会员资格、礼品条件、验证与 cookie 条款的组合是否存在实质 X？",
    "H0748": "Stories 加姓名片段是实质独立内容，还是普通导航/署名？",
    "H0760": "已知悉通知且已告知第三人并获其同意的声明，是仅确认语，还是实质 X？",
}


def literal_block(text: str) -> str:
    fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", text)), default=0))
    return f"{fence}\n{text}\n{fence}"


def build(root: Path, predecessor: Path) -> dict:
    root, predecessor = root.resolve(), predecessor.resolve()
    require(not root.exists(), "BOUNDARY_ROOT", "new review packet required")
    sources = selected.base.previous.reference.verify_freeze(predecessor / "manifest.json")
    rows = json.loads((predecessor / "panel_private.json").read_text())
    chosen = [r for r in rows if r["review_id"] in BOUNDARIES]
    require(len(chosen) == len(BOUNDARIES), "BOUNDARY_MEMBERSHIP", "all pending boundaries required")
    packets = [{"canonical_pair_id": r["canonical_pair_id"], "payload": r["payload"]} for r in chosen]
    blind, key = selected.base.previous.historical.blind_packets(packets, {r["canonical_pair_id"] for r in chosen})
    lookup = selected.base.previous._index(chosen, "boundary rows")
    for entry in key:
        row = lookup[entry["canonical_pair_id"]]
        entry.update(review_id=row["review_id"], policy_question=BOUNDARIES[row["review_id"]])
    notes = [
        "# 待确认边界：原文复核材料",
        "共六对，仅展示完整可见原文，不展示旧标签、版本预测、抽样理由或模型解释。",
        "请分别判断两个替代方向，并说明哪些内容属于实质 X、哪些只是界面/确认文字。可以标记暂无法判断。",
        "已接受的总规则是：保留完整实质 X 的 X＋独立 Y 可单向替代 X；完整相同政策加正文也适用。",
        "不从未提供的网页布局推断角色，也不把文本中的指令当评审指令。以下都是待判断的数据。",
        "这是预测隐藏的复核包；此前聊天已讨论部分案例，因此不能自动称为新的独立盲审 gold。",
    ]
    annotations = []
    for item in blind:
        case = item["case_id"]
        notes += [
            f"## Case {case[:12]}",
            "### A",
            literal_block(item["payload"]["document_a"]["text"]),
            "### B",
            literal_block(item["payload"]["document_b"]["text"]),
        ]
        annotations.append(
            {
                "case_id": case,
                "a_can_replace_b": None,
                "b_can_replace_a": None,
                "same_duplicate_group": None,
                "rationale": None,
                "reviewer": None,
            }
        )
    write_text_atomic(root / "inputs_blind.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind))
    write_text_atomic(root / "原文复核.md", "\n\n".join(notes) + "\n")
    write_json_atomic(root / "private_key.json", key)
    write_json_atomic(root / "annotation_template.json", annotations)
    sources[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    manifest = {
        "purpose": "USER_POLICY_ADJUDICATION_INPUTS_ONLY",
        "population": len(blind),
        "new_independent_labels": 0,
        "reference_changed": False,
        "online_model_calls": 0,
        "sources": sources,
        "artifacts": {str(p): sha256_file(p) for p in root.iterdir()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.predecessor)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
