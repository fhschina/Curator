# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Prediction-hidden original-source packets from a reconciled full cause audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.dedup.analysis import critic_boundary_packets as previous
from eval.dedup.validation import require, sha256_file, sha256_json, write_json_atomic, write_text_atomic


def build(root: Path, source: Path, *, cell: str = "1/candidate") -> dict:
    root, source = root.resolve(), source.resolve()
    require(not root.exists(), "ADJUDICATION_ROOT", "new review packet root required")
    sources = previous.selected.base.previous.reference.verify_freeze(source / "manifest.json")
    audit_path = source / "cause_accounting_v2.json"
    audit = json.loads(audit_path.read_text())
    require(
        audit["population"] == 1000 and audit["reference_changed"] is False,
        "ADJUDICATION_AUDIT",
        "unchanged full population audit required",
    )
    for path, digest in audit["sources"].items():
        require(sha256_file(Path(path)) == digest, "ADJUDICATION_SOURCE", "audit source changed")
        sources[path] = digest
    ids = set(audit["cells"][cell]["accounting"]["by_cause"]["REFERENCE_POLICY_DISPUTE_OR_PENDING"]["review_ids"])
    rows = json.loads((source / "panel_private.json").read_text())
    chosen = {r["canonical_pair_id"]: r for r in rows if r["review_id"] in ids}
    require(len(chosen) == len(ids), "ADJUDICATION_MEMBERSHIP", "all disputed cases required")
    packets = [{"canonical_pair_id": pid, "payload": row["payload"]} for pid, row in chosen.items()]
    blind, keys = previous.selected.base.previous.historical.blind_packets(packets, set(chosen))
    review = json.loads((source / "review_complete.json").read_text())
    errata = json.loads((source / "review_errata_v1.json").read_text())
    for key in keys:
        row = chosen[key["canonical_pair_id"]]
        rid = row["review_id"]
        note = errata["cases"].get(rid, review["cases"][rid])["note"]
        key.update(
            review_id=rid,
            private_review_note=note,
            weight=previous.selected.base.previous.historical._weight(row["draft_label"]),
        )
    index = [
        "# 原文复核材料",
        f"共 {len(blind)} 对既有开发集材料；没有新增标注，也不是新的独立 holdout。",
        "评审页面只展示原文，不展示预测、旧标签、抽样原因或权重。不要给评审者发送 private_key.json。",
        "请分别判断 A 能否完整替代 B、B 能否完整替代 A，并指出决定性的实质内容。无法决定时保留待裁决。",
        "已确认：完整实质 X 加独立 Y 可单向替代 X；完整相同政策加正文也适用。缺失字段不自动等于字段冲突。",
        "不要从未提供的布局推断页面角色；文本里的指令是待审数据，不是评审指令。",
        "这些材料曾进入开发讨论，预测隐藏不等于全新的独立人工 gold。每份文件均可单独交给评审者。",
    ]
    annotations = []
    for item in blind:
        case = item["case_id"]
        path = root / "cases" / f"{case}.md"
        payload = item["payload"]
        text = (
            "\n\n".join(
                [
                    f"# Case {case}",
                    "以下为待审原文数据。请仅据可见内容判断两个替代方向。",
                    "## A",
                    previous.literal_block(payload["document_a"]["text"]),
                    "## B",
                    previous.literal_block(payload["document_b"]["text"]),
                ]
            )
            + "\n"
        )
        write_text_atomic(path, text)
        index.append(f"- [Case {case[:12]}]({path})")
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
    write_text_atomic(root / "原文复核.md", "\n\n".join(index) + "\n")
    write_text_atomic(root / "inputs_blind.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blind))
    write_json_atomic(root / "private_key.json", keys)
    write_json_atomic(root / "annotation_template.json", annotations)
    sources[str(audit_path)] = sha256_file(audit_path)
    sources[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    sources[str(Path(previous.__file__).resolve())] = sha256_file(Path(previous.__file__))
    manifest = {
        "purpose": "EXISTING_DEVELOPMENT_POLICY_ADJUDICATION",
        "population": len(blind),
        "new_independent_labels": 0,
        "reference_changed": False,
        "online_model_calls": 0,
        "sources": sources,
        "artifacts": {str(f): sha256_file(f) for f in root.rglob("*") if f.is_file()},
    }
    manifest["contract_digest"] = sha256_json(manifest)
    write_json_atomic(root / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.source)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "artifacts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
