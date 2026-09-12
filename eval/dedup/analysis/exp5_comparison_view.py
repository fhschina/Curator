# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# ruff: noqa: RUF001

"""Presentation-only comparison views; no Judge or reference-label mutation."""

from __future__ import annotations

import html

from eval.dedup.dashboard import _script_json


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def table(headers: list[str], rows: list[list]) -> str:
    def cells(tag: str, values: list) -> str:
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)

    return (
        "<table><thead><tr>"
        + cells("th", headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + cells("td", row) + "</tr>" for row in rows)
        + "</tbody></table>"
    )


def calibration_table(summary: dict) -> str:
    calibration = summary["development_calibration"]
    if calibration["status"] != "AVAILABLE":
        return (
            f"<p>等待完整开发集：{calibration['available']} / {calibration['required']} 条已采集。"
            "不把当前部分样本的分数当作完整开发集结果。</p>"
        )
    rows = []
    for key, label in (("weighted", "加权"), ("unweighted", "非加权")):
        for metric, name in (
            ("duplicate_precision", "Duplicate precision"),
            ("duplicate_recall", "Duplicate recall"),
            ("primary_decision_exact", "主决策完全一致"),
        ):
            a, b = (calibration[v][key][metric] for v in ("v05", "exp5"))
            delta = "—" if a is None or b is None else f"{100 * (b - a):+.2f} pp"
            rows.append([label, name, percentage(a), percentage(b), delta])
    return table(["口径", "指标", "v0.5", "Exp5", "变化"], rows)


def overview_html(summary: dict, pairs: list[dict]) -> str:
    n, total = summary["collected"], summary["population"]
    state = "完整实验结果" if summary["complete"] else "运行中 · 部分结果快照"
    groups = summary["exp5_groups"], summary["v05_groups_same_collected_pairs"]
    distribution = table(
        ["判断", "v0.5", "Exp5"],
        [
            [label, groups[1].get(key, 0), groups[0].get(key, 0)]
            for key, label in (("YES", "同组"), ("NO", "不同组"), ("UNRESOLVED", "语义未决"))
        ],
    )
    agreement = summary["agreement"]
    matrix = agreement["same_duplicate_group_matrix"]
    transition = table(
        ["v0.5 → Exp5", "同组", "不同组", "语义未决"],
        [
            [label, *[matrix.get(key, {}).get(column, 0) for column in ("YES", "NO", "UNRESOLVED")]]
            for key, label in (("YES", "同组"), ("NO", "不同组"), ("UNRESOLVED", "语义未决"))
        ],
    )
    sut = "<p>完整采集及校验完成后生成。这里的指标评价 SUT 在不同 Judge 口径下的结果，不是 Judge 自身的 precision / recall。</p>"
    if "baseline_judge_conditioned_sut_metrics" in summary:
        rows = []
        for frame, key, label in (
            ("track_5a_removal_frame", "removal_precision", "5a：Judge 认为可安全删除的比例"),
            ("track_5a_removal_frame", "wrong_removal_rate", "5a：Judge 认为误删的比例"),
            ("track_5b_candidate_pool", "positive_yield", "5b：跨组候选中 Judge 判同组的比例"),
        ):
            rows.append(
                [
                    label,
                    percentage(summary["baseline_judge_conditioned_sut_metrics"][frame][key]),
                    percentage(summary["judge_conditioned_sut_metrics"][frame][key]),
                ]
            )
        sut = table(["指标", "v0.5 口径", "Exp5 口径"], rows)
    tiers = ""
    calibration = summary["development_calibration"]
    if calibration["status"] == "AVAILABLE":
        tiers = (
            "<h3>Exp5 置信档位的实际正确率</h3>"
            + table(
                ["档位", "计分样本", "非加权主决策一致", "加权主决策一致"],
                [
                    [
                        tier,
                        value["rows"],
                        percentage(value["unweighted_primary_exact"]),
                        percentage(value["weighted_primary_exact"]),
                    ]
                    for tier, value in calibration["exp5_empirical_confidence_tiers"].items()
                ],
            )
            + "<p>仅是本开发集上的经验正确率，不是已校准概率；v0.5 的旧数值 confidence 不与档位直接比较。</p>"
        )
    template = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';script-src 'unsafe-inline';connect-src 'self';base-uri 'none';form-action 'none'">
<title>Exp5 vs v0.5 · 20K 实验对比</title><style>
:root{font-family:system-ui,sans-serif;color:#172033;background:#f4f6f9;font-size:15px;line-height:1.5}*{box-sizing:border-box}body{margin:0}header{background:white;border-bottom:1px solid #d8dee9;padding:24px max(24px,calc((100vw - 1360px)/2))}h1{margin:0 0 8px;font-size:28px}h2{font-size:19px;margin:0 0 14px}h3{font-size:16px}p,.muted{color:#647087}nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#2457d6}main{max-width:1408px;margin:auto;padding:24px;display:grid;gap:20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.card,section{background:white;border:1px solid #d8dee9;border-radius:8px;padding:20px}.card span{display:block;color:#647087}.card strong{font-size:28px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #e5eaf0;vertical-align:top}th{color:#647087}tbody tr:hover{background:#f8fafc}.badge{display:inline-block;background:#edf3ff;color:#2457d6;border-radius:5px;padding:4px 9px}.controls{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap}input,select,button{font:inherit;padding:8px;border:1px solid #ccd5e2;border-radius:5px;background:white}input{min-width:320px}button{cursor:pointer}button:disabled{opacity:.5}.scroll{overflow:auto}.pair-id{font:12px monospace;max-width:360px;word-break:break-all}.warning{border-left:4px solid #bf8700;background:#fff9e9;padding:12px;color:#604a14}.links{display:flex;gap:16px;margin-top:12px}#live{font-size:13px;color:#647087}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.columns{grid-template-columns:1fr}main{padding:16px}input{min-width:0;flex:1}}
</style></head><body><header><span class="badge">__STATE__</span><h1>Exp5 vs v0.5</h1>
<p>同一批 __TOTAL__ 对 · Exp5 = Exp1 + 证据小修复 · v0.5 = 已发布的 Sarah 框架 + Qwen</p>
<nav><a href="pair_explorer_exp5.html">打开 Pair Explorer</a><a href="v05_exp5_pairs.csv">下载逐对对照 CSV</a><a href="comparison.json">下载完整统计</a><a href="RESULTS.md">展示摘要</a></nav>
<p id="live">快照采集时间：__TIME__。快照 __N__ / __TOTAL__；后台可能已继续推进。</p></header>
<main><div class="cards"><div class="card"><span>已采集 / 总量</span><strong>__N__ / __TOTAL__</strong></div><div class="card"><span>有效结果</span><strong>__VALID__</strong></div><div class="card"><span>终态工程失败</span><strong>__FAILURES__</strong></div><div class="card"><span>主决策不同的 pair</span><strong>__CHANGED__</strong></div></div>
<div class="warning">这是实验检查点，不是正式 release，也不是独立 holdout 验证。版本间一致率不等于正确率；不依据 proxy 或 DeepSeek agreement 宣布版本胜出。</div>
<div class="columns"><section><h2>同一已采集范围内的判断分布</h2>__DISTRIBUTION__<p>Exp5 工程失败单独保留，不伪装成语义未决或不同组。</p></section>
<section><h2>同组判断转移</h2>__MATRIX__<p>仅统计双方有效结果：__COMMON__ 对。反映判断变化，不是对错混淆矩阵。</p></section></div>
<section id="calibration"><h2>开发集校准：与同一份参考比较</h2><p>完整开发集 1,000 条，沿用五条比较排除；加权与非加权分别展示。参照包含用户确认、助手规则应用和继承标签，不冒充独立人工金标。</p>__CALIBRATION____TIERS__</section>
<section><h2>SUT 诊断：分别由两个 Judge 评价</h2>__SUT__<p>5a 与 5b 的抽样范围不同，不合并为全语料 precision / recall。实际 SUT MinHash 配置缺失：UNAVAILABLE_MISSING_CONTRACT。</p></section>
<section><h2>逐对版本对照</h2><p>替代方向始终使用原始 A/B 展示顺序；点击 pair 在原 Pair Explorer 中查看正文、证据和 SUT 上下文。</p>
<div class="controls"><input id="search" placeholder="搜索 pair ID"><select id="change"><option value="all">全部已采集 pair</option><option value="changed">仅主决策变化</option><option value="same">仅主决策一致</option><option value="failure">工程失败 / 不可比较</option></select><button id="previous">上一页</button><button id="next">下一页</button><span id="page"></span></div>
<div class="scroll"><table><thead><tr><th>Pair</th><th>v0.5：同组 · A替B · B替A</th><th>Exp5：同组 · A替B · B替A</th><th>状态</th></tr></thead><tbody id="pairs"></tbody></table></div></section>
</main><script>
const DATA=__PAIRS__,LIMIT=50;let page=0,filtered=[];
const fieldKeys=["same_duplicate_group","a_can_replace_b","b_can_replace_a"];
function cell(tr,text){const td=document.createElement("td");td.textContent=text;tr.appendChild(td);return td}
function render(){const body=document.getElementById("pairs");body.replaceChildren();const count=Math.max(1,Math.ceil(filtered.length/LIMIT));page=Math.min(page,count-1);
 for(const r of filtered.slice(page*LIMIT,(page+1)*LIMIT)){const tr=document.createElement("tr"),td=cell(tr,""),link=document.createElement("a");link.textContent=r.canonical_pair_id;link.href="pair_explorer_exp5.html#pair="+encodeURIComponent(r.canonical_pair_id);td.className="pair-id";td.appendChild(link);
  cell(tr,fieldKeys.map(k=>r["v05_"+k]).join(" · "));cell(tr,fieldKeys.map(k=>r["exp5_"+k]).join(" · "));cell(tr,r.primary_changed===null?"不可比较 / 工程失败":r.primary_changed?"变化":"一致");body.appendChild(tr)}
 document.getElementById("page").textContent=`${filtered.length} 对 · ${page+1} / ${count}`;document.getElementById("previous").disabled=page===0;document.getElementById("next").disabled=page+1>=count;}
function filter(){const query=document.getElementById("search").value.toLowerCase(),mode=document.getElementById("change").value;filtered=DATA.filter(r=>r.canonical_pair_id.toLowerCase().includes(query)&&(mode==="all"||mode==="changed"&&r.primary_changed===true||mode==="same"&&r.primary_changed===false||mode==="failure"&&r.primary_changed===null));page=0;render()}
document.getElementById("search").oninput=filter;document.getElementById("change").onchange=filter;document.getElementById("previous").onclick=()=>{page--;render()};document.getElementById("next").onclick=()=>{page++;render()};filter();
if(location.protocol.startsWith("http")){async function progress(){try{const r=await fetch("progress.json",{cache:"no-store"});if(!r.ok)return;const s=await r.json();document.getElementById("live").textContent=`快照 __N__ / __TOTAL__ · 后台 ${s.completed} / ${s.population} · ${s.complete?"采集校验完成":s.running?"运行中":"已停止，请检查运行状态"}`;}catch{}}progress();setInterval(progress,30000)}
</script></body></html>"""
    values = {
        "STATE": html.escape(state),
        "N": f"{n:,}",
        "TOTAL": f"{total:,}",
        "VALID": f"{summary['valid']:,}",
        "FAILURES": f"{summary['engineering_failures']:,}",
        "CHANGED": f"{summary['primary_changed_common_valid']:,}",
        "TIME": html.escape(summary.get("snapshot_at_utc", "—")),
        "COMMON": f"{agreement['common_valid']:,}",
        "DISTRIBUTION": distribution,
        "MATRIX": transition,
        "CALIBRATION": calibration_table(summary),
        "TIERS": tiers,
        "SUT": sut,
        "PAIRS": _script_json(pairs),
    }
    for key, value in values.items():
        template = template.replace("__" + key + "__", value)
    return template


def results_markdown(summary: dict) -> str:
    lines = [
        "# Exp5 vs v0.5",
        "",
        "完整实验结果" if summary["complete"] else "部分结果快照",
        "",
        f"已采集 {summary['collected']:,}/{summary['population']:,}；有效 {summary['valid']:,}；工程失败 {summary['engineering_failures']:,}；语义未决 {summary['semantic_unresolved']:,}。",
        "",
        "Exp5 保留 Exp1，仅增加一次受约束的冲突证据修复；没有接入 Exp4。v0.5 指发布记录中的 Sarah 框架＋Qwen，不是更早的 V0／DeepSeek。",
        "",
        f"双方有效范围内，{summary['primary_changed_common_valid']:,} 对的主决策不同。这不是准确率，也不据此宣布哪个版本更好。",
        "",
    ]
    calibration = summary["development_calibration"]
    if calibration["status"] == "AVAILABLE":
        lines += [
            "## 同一开发参照下的校准",
            "",
            "| 口径 | 版本 | Precision | Recall | 主决策完全一致 |",
            "|---|---|---:|---:|---:|",
        ]
        for key, label in (("weighted", "加权"), ("unweighted", "非加权")):
            for version in ("v05", "exp5"):
                values = calibration[version][key]
                metrics = " | ".join(
                    percentage(values[m])
                    for m in ("duplicate_precision", "duplicate_recall", "primary_decision_exact")
                )
                lines.append(f"| {label} | {'v0.5' if version == 'v05' else 'Exp5'} | {metrics} |")
        lines += [
            "",
            "使用相同的 1,000 条开发参照和五条比较排除。参照包含用户确认、助手规则应用及继承标签；不是独立人工金标，也不是 holdout 验证。",
            "",
        ]
    else:
        lines += ["完整开发集尚未采集齐，不报告部分开发样本分数。", ""]
    lines += [
        "## 展示入口",
        "",
        "- [版本总览](comparison.html)",
        "- [Pair Explorer](pair_explorer_exp5.html)",
        "- [逐对对照 CSV](v05_exp5_pairs.csv)",
        "- [完整统计 JSON](comparison.json)",
        "",
        "SUT 诊断按 5a／5b 分开呈现，不能解释为 Judge 在全语料上的 precision／recall。缺少实际 SUT 配置，MinHash 重放保持 UNAVAILABLE_MISSING_CONTRACT。",
        "",
        "该版本仍是实验检查点，未标记正式 release；不使用 proxy 或 DeepSeek agreement 决定版本优劣。",
        "",
    ]
    return "\n".join(lines)
