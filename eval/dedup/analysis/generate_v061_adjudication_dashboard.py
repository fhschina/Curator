# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the V0.5/V0.6/V0.6.1 adjudicated comparison CSV and dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

PRIMARY_FIELDS = ("same_duplicate_group", "a_can_replace_b", "b_can_replace_a")
TAXONOMY_FIELDS = ("relation_type", "material_difference")
COMPARABLE_FIELDS = PRIMARY_FIELDS + TAXONOMY_FIELDS
V061_FIELDS = (
    *COMPARABLE_FIELDS,
    "primary_material_difference",
    "expected_minhash_action",
    "surface_evidence_sufficiency",
    "expected_minhash_outcome",
    "dominant_overlap_source",
    "primary_risk_factor",
    "evidence_quality",
    "confidence",
)
DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = DEFAULT_ANALYSIS_ROOT / "v05_v06_codex_adjudication_14336.csv"
DEFAULT_RUN_ROOT = Path("/raid/hfang/ihb/runs/v0.6.1")
DEFAULT_SOURCE_RESULTS = Path(
    "/raid/hfang/dedup_eval_runs/dedup-full-20260813T220949Z-d4c37bb483/v0_run/data/judge_results.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(value)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _index(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row.get("canonical_pair_id", ""))
        if not pair_id or pair_id in result:
            msg = f"{label} contains a missing or duplicate canonical_pair_id: {pair_id!r}"
            raise ValueError(msg)
        result[pair_id] = row
    return result


def _distance(row: dict[str, Any], prefix: str, fields: tuple[str, ...]) -> int:
    return sum(str(row.get(f"{prefix}_{field}", "")) != str(row.get(f"adjudicated_{field}", "")) for field in fields)


def _support(v05_distance: int, v061_distance: int) -> str:
    if v05_distance == v061_distance == 0:
        return "BOTH_ACCEPTABLE"
    if v05_distance < v061_distance:
        return "V0.5"
    if v061_distance < v05_distance:
        return "V0.6.1"
    return "NEITHER"


def _change(previous: int, current: int) -> str:
    if current < previous:
        return "IMPROVED"
    if current > previous:
        return "REGRESSED"
    return "UNCHANGED"


def _v061_values(result: dict[str, Any]) -> dict[str, Any]:
    values = {f"v061_{field}": result.get(field, "") for field in V061_FIELDS}
    values["v061_reason_codes"] = ";".join(str(item) for item in result.get("reason_codes", []))
    values["v061_evidence_count"] = len(result.get("evidence", []))
    values["v061_attempts"] = result.get("attempts", "")
    values["v061_retried"] = result.get("retried", "")
    return values


def _benchmark_rows(
    original: list[dict[str, str]],
    v061_by_pair: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for source in original:
        pair_id = source["canonical_pair_id"]
        if pair_id not in v061_by_pair:
            msg = f"V0.6.1 result missing from adjudicated benchmark: {pair_id}"
            raise ValueError(msg)
        row: dict[str, Any] = {**source, **_v061_values(v061_by_pair[pair_id])}
        for prefix in ("v05", "v06", "v061"):
            primary_distance = _distance(row, prefix, PRIMARY_FIELDS)
            taxonomy_distance = _distance(row, prefix, TAXONOMY_FIELDS)
            row[f"{prefix}_primary_distance"] = primary_distance
            row[f"{prefix}_primary_exact"] = primary_distance == 0
            row[f"{prefix}_taxonomy_distance"] = taxonomy_distance
            row[f"{prefix}_taxonomy_exact"] = taxonomy_distance == 0
        row["v05_v061_support"] = _support(row["v05_primary_distance"], row["v061_primary_distance"])
        row["v05_v061_support_reason"] = (
            "Primary decision tuple distance to the existing adjudication: "
            f"V0.5={row['v05_primary_distance']}, V0.6.1={row['v061_primary_distance']}. "
            "Descriptive taxonomy and MinHash diagnostics do not decide the winner."
        )
        row["v061_change_vs_v06"] = _change(row["v06_primary_distance"], row["v061_primary_distance"])
        row["changed_fields_v05_v061"] = ";".join(
            field for field in COMPARABLE_FIELDS if str(row[f"v05_{field}"]) != str(row[f"v061_{field}"])
        )
        output.append(row)
    return output


def _presented_metadata(candidate: dict[str, Any], side: str) -> dict[str, Any]:
    doc_id = int(candidate[f"presented_doc_{side.lower()}"])
    suffix = "low" if doc_id == int(candidate["doc_id_low"]) else "high"
    return {
        "doc_id": doc_id,
        "url": candidate.get(f"url_{suffix}", ""),
        "hostname": candidate.get(f"hostname_{suffix}", ""),
        "language": candidate.get(f"language_{suffix}", ""),
        "tokens": candidate.get(f"token_count_{suffix}", ""),
    }


def _excerpt(payload: dict[str, Any], side: str, limit: int = 900) -> str:
    document = payload[f"document_{side.lower()}"]
    text = document.get("text")
    if isinstance(text, str):
        return text if len(text) <= limit else text[:limit].rstrip() + "…"
    windows = [
        str(item.get("text", ""))
        for item in payload.get("long_document_evidence", {}).get("windows", [])
        if item.get("side") == side
    ]
    joined = "\n…\n".join(windows)
    return joined if len(joined) <= limit else joined[:limit].rstrip() + "…"


def _full_conflict_rows(
    *,
    candidates: list[dict[str, Any]],
    payload_by_pair: dict[str, dict[str, Any]],
    v05_by_pair: dict[str, dict[str, Any]],
    v061_by_pair: dict[str, dict[str, Any]],
    adjudication_by_pair: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for candidate in candidates:
        pair_id = str(candidate["canonical_pair_id"])
        if pair_id not in v05_by_pair:
            continue
        v05 = v05_by_pair[pair_id]
        v061 = v061_by_pair[pair_id]
        changed = [field for field in COMPARABLE_FIELDS if v05.get(field) != v061.get(field)]
        if not changed:
            continue
        payload = payload_by_pair[pair_id]
        side_a = _presented_metadata(candidate, "A")
        side_b = _presented_metadata(candidate, "B")
        prior = adjudication_by_pair.get(pair_id)
        row: dict[str, Any] = {
            "canonical_pair_id": pair_id,
            "changed_fields": ";".join(changed),
            **{f"document_a_{key}": value for key, value in side_a.items()},
            **{f"document_b_{key}": value for key, value in side_b.items()},
            "document_a_excerpt": _excerpt(payload, "A"),
            "document_b_excerpt": _excerpt(payload, "B"),
            **{f"v05_{field}": v05.get(field, "") for field in COMPARABLE_FIELDS},
            "v05_fuzzy_scope": v05.get("fuzzy_scope", ""),
            **_v061_values(v061),
            "prior_adjudication_available": prior is not None,
        }
        if prior is not None:
            row.update({f"adjudicated_{field}": prior.get(f"adjudicated_{field}", "") for field in COMPARABLE_FIELDS})
            row["adjudication_source"] = prior.get("adjudication_source", "")
            row["adjudication_confidence"] = prior.get("adjudication_confidence", "")
            v05_distance = _distance(row, "v05", PRIMARY_FIELDS)
            v061_distance = _distance(row, "v061", PRIMARY_FIELDS)
            row["v05_v061_support"] = _support(v05_distance, v061_distance)
        else:
            row["v05_v061_support"] = "NOT_ADJUDICATED"
        output.append(row)
    return output


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _dashboard_data(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append(
            {
                "id": row["review_id"],
                "pair": row["canonical_pair_id"],
                "stratum": row["sample_stratum"],
                "changed": row["changed_fields_v05_v061"],
                "support": row["v05_v061_support"],
                "change": row["v061_change_vs_v06"],
                "source": row["adjudication_source"],
                "band": row["confidence_band"],
                "adjudication_confidence": float(row["adjudication_confidence"]),
                "reason": row["adjudication_reason"],
                "lesson": row["hs_policy_lesson"],
                "reason_code": row["adjudication_reason_code"],
                "human": [row[f"adjudicated_{field}"] for field in COMPARABLE_FIELDS],
                "v05": [row[f"v05_{field}"] for field in COMPARABLE_FIELDS]
                + [row["v05_fuzzy_scope"], int(row["v05_consistency_violations"])],
                "v06": [row[f"v06_{field}"] for field in COMPARABLE_FIELDS]
                + [row["v06_fuzzy_scope"], int(row["v06_consistency_violations"])],
                "v061": [row[f"v061_{field}"] for field in COMPARABLE_FIELDS],
                "diagnostics": [
                    row["v061_primary_material_difference"],
                    row["v061_expected_minhash_action"],
                    row["v061_surface_evidence_sufficiency"],
                    row["v061_expected_minhash_outcome"],
                    row["v061_dominant_overlap_source"],
                    row["v061_primary_risk_factor"],
                    row["v061_evidence_quality"],
                    float(row["v061_confidence"]),
                    row["v061_reason_codes"],
                    int(row["v061_evidence_count"]),
                    int(row["v061_attempts"]),
                ],
                "distances": [
                    int(row["v05_primary_distance"]),
                    int(row["v06_primary_distance"]),
                    int(row["v061_primary_distance"]),
                ],
                "a": {
                    "doc": row["presented_doc_a"],
                    "url": row["document_a_url"],
                    "host": row["document_a_hostname"],
                    "lang": row["document_a_language"],
                    "tokens": int(row["document_a_tokens"]),
                    "text": row["document_a_excerpt"],
                    "evidence": row["document_a_evidence"],
                },
                "b": {
                    "doc": row["presented_doc_b"],
                    "url": row["document_b_url"],
                    "host": row["document_b_hostname"],
                    "lang": row["document_b_language"],
                    "tokens": int(row["document_b_tokens"]),
                    "text": row["document_b_excerpt"],
                    "evidence": row["document_b_evidence"],
                },
            }
        )
    return result


def _dashboard_html(rows: list[dict[str, Any]], *, full_conflict_count: int, common_valid_count: int) -> str:
    data = _script_json(_dashboard_data(rows))
    source_counts = Counter(row["adjudication_source"] for row in rows)
    subtitle = (
        f"The same {len(rows):,} V0.5/V0.6 conflict pairs are used as a frozen adjudicated benchmark: "
        f"{source_counts.get('BLIND_SEED_REVIEW', 0) + source_counts.get('SEED_REVIEW', 0):,} prior blind seed reviews and "
        f"{sum(value for key, value in source_counts.items() if 'PROXY' in key):,} calibrated Codex-proxy labels. "
        f"Across {common_valid_count:,} common-valid pairs in the complete workload, V0.5 and V0.6.1 differ on "
        f"{full_conflict_count:,} pairs in comparable fields."
    )
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';script-src 'unsafe-inline';base-uri 'none';form-action 'none'">
<title>V0.5 vs V0.6.1 Adjudication Dashboard</title>
<style>
:root{--bg:#f5f7f6;--panel:#fff;--ink:#17251f;--muted:#68756e;--line:#dce4df;--green:#087f5b;--green2:#dff6ec;--blue:#2563eb;--blue2:#eaf0ff;--amber:#b45309;--amber2:#fff3dc;--red:#b42318;--red2:#fff0ee;--purple:#7c3aed;--purple2:#f1eaff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}header{padding:24px 30px 18px;background:linear-gradient(135deg,#073b31,#0b6b55);color:#fff}h1{margin:0 0 7px;font-size:27px}h2{margin:0;font-size:18px}h3{margin:0 0 8px;font-size:15px}.subtitle{max-width:1100px;color:#d7eee6}.notice{margin-top:11px;padding:8px 11px;border:1px solid #5b9886;border-radius:7px;background:#ffffff12;font-size:12px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;padding:18px 30px 0}.stat,.panel,.chart{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:0 2px 8px #18251f0b}.stat{padding:13px 15px}.stat span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.stat strong{display:block;margin-top:3px;font-size:22px}.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:11px;padding:12px 30px}.chart{padding:14px}.chart h3{margin-bottom:10px}.bar-row{display:grid;grid-template-columns:minmax(120px,.7fr) 1fr auto;align-items:center;gap:8px;margin:6px 0;font-size:12px}.track{height:8px;background:#eef1ef;border-radius:999px;overflow:hidden}.fill{display:block;height:100%;min-width:2px;background:var(--green);border-radius:inherit}.controls{display:flex;flex-wrap:wrap;gap:8px;padding:2px 30px 13px}input,select,button{min-height:36px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);padding:7px 9px;font:inherit}input{min-width:290px;flex:1}button{cursor:pointer}.workspace{display:grid;grid-template-columns:minmax(610px,.95fr) minmax(580px,1.05fr);gap:13px;padding:0 30px 30px}.panel{min-width:0;overflow:hidden}.head{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--line)}.pager{display:flex;align-items:center;gap:8px}.table-wrap{max-height:calc(100vh - 345px);overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{position:sticky;top:0;background:#fbfcfb;color:var(--muted);font-size:11px}tr.item{cursor:pointer}tr.item:hover,tr.selected{background:#eef8f4}.mono{font:11px ui-monospace,monospace;max-width:145px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.badge{display:inline-block;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:700;white-space:nowrap}.v05{background:var(--blue2);color:var(--blue)}.v061,.improved{background:var(--green2);color:var(--green)}.both,.unchanged{background:var(--purple2);color:var(--purple)}.neither,.regressed{background:var(--red2);color:var(--red)}.detail{padding:15px;max-height:calc(100vh - 280px);overflow:auto}.detail>section{margin-top:16px;padding-top:15px;border-top:1px solid var(--line)}.detail>section:first-of-type{margin-top:11px}.tuple-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.version{padding:10px;border:1px solid var(--line);border-top:4px solid var(--muted);border-radius:8px;background:#fbfcfb}.version.human{border-top-color:var(--purple)}.version.old{border-top-color:var(--blue)}.version.mid{border-top-color:var(--amber)}.version.new{border-top-color:var(--green)}.field{margin:6px 0}.field span,.fact span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.field strong,.fact strong{overflow-wrap:anywhere}.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 13px}.docs{display:grid;grid-template-columns:1fr 1fr;gap:10px}.doc{padding:11px;border:1px solid var(--line);border-radius:8px;background:#fbfcfb;min-width:0}.meta{color:var(--muted);font-size:11px;overflow-wrap:anywhere}.doc pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f2;padding:9px;border-radius:6px;font:11px/1.5 ui-monospace,monospace}.evidence{padding:8px;border-left:3px solid var(--purple);background:var(--purple2);white-space:pre-wrap;overflow-wrap:anywhere}.empty{padding:35px;text-align:center;color:var(--muted)}a{color:var(--blue)}@media(max-width:1250px){.workspace{grid-template-columns:1fr}.table-wrap,.detail{max-height:none}}@media(max-width:780px){header,.stats,.charts,.controls,.workspace{padding-left:14px;padding-right:14px}.stats,.charts{grid-template-columns:1fr 1fr}.tuple-grid,.facts,.docs{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Adjudication · V0.5 vs V0.6.1</h1><div class="subtitle">__SUBTITLE__</div><div class="notice">Winner/support labels use only the three primary decisions: same duplicate group and both replacement directions. Relation/material are descriptive; V0.6.1 MinHash fields are diagnostics and never decide the semantic winner.</div></header>
<section class="stats"><div class="stat"><span>Filtered benchmark</span><strong id="n">0</strong></div><div class="stat"><span>V0.5 primary exact</span><strong id="a5">0</strong></div><div class="stat"><span>V0.6 primary exact</span><strong id="a6">0</strong></div><div class="stat"><span>V0.6.1 primary exact</span><strong id="a61">0</strong></div><div class="stat"><span>V0.6.1 improves on V0.6</span><strong id="imp">0</strong></div></section>
<section class="charts"><div class="chart"><h3>V0.5 vs V0.6.1 support</h3><div id="support-chart"></div></div><div class="chart"><h3>V0.6.1 change vs V0.6</h3><div id="change-chart"></div></div><div class="chart"><h3>Expected MinHash outcome</h3><div id="minhash-chart"></div></div></section>
<section class="controls"><input id="search" type="search" placeholder="Search pair, host, text, reason…"><select id="support"><option value="">All support outcomes</option></select><select id="change"><option value="">Any change vs V0.6</option></select><select id="stratum"><option value="">All original strata</option></select><select id="minhash"><option value="">Any MinHash outcome</option></select><select id="risk"><option value="">Any V0.6.1 risk</option></select><button id="reset">Reset</button></section>
<main class="workspace"><section class="panel"><div class="head"><h2>Pairs</h2><div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div></div><div class="table-wrap"><table><thead><tr><th>ID</th><th>Support</th><th>vs V0.6</th><th>Human</th><th>V0.5</th><th>V0.6.1</th><th>Risk</th></tr></thead><tbody id="rows"></tbody></table><div id="none" class="empty" hidden>No matching pairs.</div></div></section><section class="panel"><div class="head"><h2>Pair detail</h2><button id="copy" disabled>Copy pair ID</button></div><div id="detail" class="detail"><div class="empty">Select a pair.</div></div></section></main>
<script>
const DATA=__DATA__,FIELDS=["Same duplicate group","A can replace B","B can replace A","Relation","Material difference"],DIAG=["Primary material difference","Expected MinHash action","Surface evidence sufficiency","Expected MinHash outcome","Dominant overlap source","Primary risk factor","Evidence quality","Confidence","Reason codes","Evidence count","Attempts"],PAGE=100,els=Object.fromEntries(["search","support","change","stratum","minhash","risk"].map(x=>[x,document.getElementById(x)])),state={items:DATA,page:0,selected:null};
const txt=(v)=>v===undefined||v===null||v===""?"—":String(v),node=(tag,cls,value)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(value!==undefined)n.textContent=txt(value);return n},pct=(n,d)=>d?`${(n/d*100).toFixed(1)}%`:`0.0%`,tone=v=>v==="V0.5"?"v05":v==="V0.6.1"?"v061":v==="BOTH_ACCEPTABLE"?"both":v==="IMPROVED"?"improved":v==="REGRESSED"?"regressed":v==="UNCHANGED"?"unchanged":"neither";
function option(id,key){for(const v of[...new Set(DATA.map(r=>key(r)).filter(Boolean))].sort()){const o=node("option","",v);o.value=v;els[id].appendChild(o)}}option("support",r=>r.support);option("change",r=>r.change);option("stratum",r=>r.stratum);option("minhash",r=>r.diagnostics[3]);option("risk",r=>r.diagnostics[5]);
function matches(r){const q=els.search.value.trim().toLowerCase(),hay=[r.id,r.pair,r.stratum,r.changed,r.reason_code,r.reason,r.lesson,r.a.host,r.b.host,r.a.text,r.b.text,...r.diagnostics].join(" ").toLowerCase();return(!q||hay.includes(q))&&(!els.support.value||r.support===els.support.value)&&(!els.change.value||r.change===els.change.value)&&(!els.stratum.value||r.stratum===els.stratum.value)&&(!els.minhash.value||r.diagnostics[3]===els.minhash.value)&&(!els.risk.value||r.diagnostics[5]===els.risk.value)}
function bars(id,key){const c=document.getElementById(id),counts=new Map();for(const r of state.items){const v=key(r);counts.set(v,(counts.get(v)||0)+1)}const values=[...counts].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])),max=values[0]?.[1]||1;c.replaceChildren();for(const[v,n]of values){const row=node("div","bar-row"),label=node("span","",v),track=node("span","track"),fill=node("span","fill"),amount=node("strong","",n.toLocaleString());fill.style.width=`${n/max*100}%`;track.appendChild(fill);row.append(label,track,amount);c.appendChild(row)}}
function stats(){const n=state.items.length;document.getElementById("n").textContent=n.toLocaleString();document.getElementById("a5").textContent=pct(state.items.filter(r=>r.distances[0]===0).length,n);document.getElementById("a6").textContent=pct(state.items.filter(r=>r.distances[1]===0).length,n);document.getElementById("a61").textContent=pct(state.items.filter(r=>r.distances[2]===0).length,n);document.getElementById("imp").textContent=pct(state.items.filter(r=>r.change==="IMPROVED").length,n);bars("support-chart",r=>r.support);bars("change-chart",r=>r.change);bars("minhash-chart",r=>r.diagnostics[3])}
function badge(v){return node("span",`badge ${tone(v)}`,v.replaceAll("_"," "))}function renderRows(){const body=document.getElementById("rows"),pages=Math.max(1,Math.ceil(state.items.length/PAGE));state.page=Math.min(state.page,pages-1);body.replaceChildren();for(const r of state.items.slice(state.page*PAGE,(state.page+1)*PAGE)){const tr=node("tr",`item${state.selected===r.pair?" selected":""}`);tr.append(node("td","mono",r.id),(()=>{const x=node("td");x.appendChild(badge(r.support));return x})(),(()=>{const x=node("td");x.appendChild(badge(r.change));return x})(),node("td","",r.human[0]),node("td","",r.v05[0]),node("td","",r.v061[0]),node("td","",r.diagnostics[5]));tr.onclick=()=>select(r.pair);body.appendChild(tr)}document.getElementById("page").textContent=`${state.page+1} / ${pages}`;document.getElementById("prev").disabled=state.page===0;document.getElementById("next").disabled=state.page+1>=pages;document.getElementById("none").hidden=state.items.length!==0}
function field(c,label,value){const d=node("div","field");d.append(node("span","",label),node("strong","",value));c.appendChild(d)}function fact(c,label,value){const d=node("div","fact");d.append(node("span","",label),node("strong","",value));c.appendChild(d)}function section(root,title){const s=node("section"),h=node("h2","",title);s.appendChild(h);root.appendChild(s);return s}function version(c,title,values,cls,distance){const d=node("div",`version ${cls}`);d.appendChild(node("h3","",`${title} · primary distance ${distance}`));FIELDS.forEach((f,i)=>field(d,f,values[i]));c.appendChild(d)}
function safeLink(c,url){try{const u=new URL(url);if(["http:","https:"].includes(u.protocol)){const a=node("a","",url);a.href=u.href;a.target="_blank";a.rel="noopener noreferrer";c.appendChild(a);return}}catch(_){}c.appendChild(node("span","meta",url))}function doc(c,label,d){const x=node("div","doc");x.append(node("h3","",`${label} · doc ${d.doc}`),node("div","meta",`${d.host} · ${d.lang} · ${d.tokens.toLocaleString()} tokens`));safeLink(x,d.url);x.appendChild(node("pre","",d.text));if(d.evidence)x.appendChild(node("div","evidence",d.evidence));c.appendChild(x)}
function select(id){const r=DATA.find(x=>x.pair===id);if(!r)return;state.selected=id;const root=document.getElementById("detail");root.replaceChildren(node("h2","mono",r.pair),node("div","meta",`${r.id} · ${r.stratum} · changed: ${r.changed||"none"}`));const s=section(root,"Primary decision and taxonomy"),grid=node("div","tuple-grid");version(grid,"Adjudication",r.human,"human",0);version(grid,"V0.5",r.v05,"old",r.distances[0]);version(grid,"V0.6",r.v06,"mid",r.distances[1]);version(grid,"V0.6.1",r.v061,"new",r.distances[2]);s.appendChild(grid);const ds=section(root,"V0.6.1 MinHash diagnostics"),facts=node("div","facts");DIAG.forEach((f,i)=>fact(facts,f,r.diagnostics[i]));ds.appendChild(facts);const as=section(root,"Adjudication context"),af=node("div","facts");for(const x of[["Support",r.support],["Change vs V0.6",r.change],["Source",r.source],["Confidence band",r.band],["Confidence",r.adjudication_confidence],["Reason code",r.reason_code]])fact(af,...x);as.append(af,node("p","",r.reason),node("p","meta",r.lesson));const docs=section(root,"Documents"),dg=node("div","docs");doc(dg,"Document A",r.a);doc(dg,"Document B",r.b);docs.appendChild(dg);document.getElementById("copy").disabled=false;history.replaceState(null,"",`#pair=${encodeURIComponent(id)}`);renderRows()}
function apply(){state.page=0;state.items=DATA.filter(matches).sort((a,b)=>a.support.localeCompare(b.support)||a.pair.localeCompare(b.pair));stats();renderRows()}for(const e of Object.values(els))e.addEventListener(e.id==="search"?"input":"change",apply);document.getElementById("reset").onclick=()=>{for(const e of Object.values(els))e.value="";apply()};document.getElementById("prev").onclick=()=>{if(state.page){state.page--;renderRows()}};document.getElementById("next").onclick=()=>{state.page++;renderRows()};document.getElementById("copy").onclick=()=>navigator.clipboard.writeText(state.selected);apply();const initial=location.hash.match(/^#pair=(.+)$/);if(initial)select(decodeURIComponent(initial[1]));
</script></body></html>"""
    return template.replace("__SUBTITLE__", subtitle).replace("__DATA__", data)


def generate(
    *,
    benchmark_csv: Path,
    run_root: Path,
    source_results: Path,
    benchmark_output: Path,
    conflicts_output: Path,
    dashboard_output: Path,
) -> dict[str, Any]:
    original = _read_csv(benchmark_csv)
    adjudication_by_pair = _index(original, label="benchmark")
    v061_rows = _read_jsonl(run_root / "data" / "judge_results.jsonl")
    if any(row.get("record_type") != "result" for row in v061_rows):
        raise ValueError("V0.6.1 results contain non-result records")
    v061_by_pair = _index(v061_rows, label="V0.6.1 results")
    v05_by_pair = _index(_read_jsonl(source_results), label="V0.5 results")
    import pyarrow.parquet as pq

    candidates = pq.read_table(run_root / "data" / "candidate_pairs.parquet").to_pylist()
    payload_rows = _read_jsonl(run_root / "data" / "judge_payloads.jsonl")
    payload_by_pair = {str(row["canonical_pair_id"]): row["payload"] for row in payload_rows}
    expected_ids = {str(row["canonical_pair_id"]) for row in candidates}
    if (
        set(v061_by_pair) != expected_ids
        or set(payload_by_pair) != expected_ids
        or not set(v05_by_pair) <= expected_ids
    ):
        raise ValueError("full-workload pair membership differs across V0.5, V0.6.1, candidates, or payloads")

    benchmark = _benchmark_rows(original, v061_by_pair)
    conflicts = _full_conflict_rows(
        candidates=candidates,
        payload_by_pair=payload_by_pair,
        v05_by_pair=v05_by_pair,
        v061_by_pair=v061_by_pair,
        adjudication_by_pair=adjudication_by_pair,
    )
    benchmark_fields = [
        *original[0],
        *(f"v061_{field}" for field in V061_FIELDS),
        "v061_reason_codes",
        "v061_evidence_count",
        "v061_attempts",
        "v061_retried",
        *(
            f"{prefix}_{metric}"
            for prefix in ("v05", "v06", "v061")
            for metric in ("primary_distance", "primary_exact", "taxonomy_distance", "taxonomy_exact")
        ),
        "v05_v061_support",
        "v05_v061_support_reason",
        "v061_change_vs_v06",
        "changed_fields_v05_v061",
    ]
    conflict_fields = list(conflicts[0]) if conflicts else ["canonical_pair_id"]
    _write_csv(benchmark_output, benchmark, benchmark_fields)
    _write_csv(conflicts_output, conflicts, conflict_fields)
    _write_text(
        dashboard_output,
        _dashboard_html(benchmark, full_conflict_count=len(conflicts), common_valid_count=len(v05_by_pair)),
    )
    return {
        "benchmark_pairs": len(benchmark),
        "full_common_valid_pairs": len(v05_by_pair),
        "full_comparable_conflicts": len(conflicts),
        "support": dict(Counter(row["v05_v061_support"] for row in benchmark)),
        "v061_change_vs_v06": dict(Counter(row["v061_change_vs_v06"] for row in benchmark)),
        "primary_exact": {
            prefix: sum(bool(row[f"{prefix}_primary_exact"]) for row in benchmark) for prefix in ("v05", "v06", "v061")
        },
        "benchmark_csv": str(benchmark_output),
        "conflicts_csv": str(conflicts_output),
        "dashboard_html": str(dashboard_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-csv", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "v05_v06_v061_adjudicated_benchmark_14336.csv",
    )
    parser.add_argument(
        "--conflicts-output",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "v05_v061_comparable_conflicts.csv",
    )
    parser.add_argument(
        "--dashboard-output",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / "v05_v061_adjudication_dashboard.html",
    )
    args = parser.parse_args()
    summary = generate(
        benchmark_csv=args.benchmark_csv.resolve(),
        run_root=args.run_root.resolve(),
        source_results=args.source_results.resolve(),
        benchmark_output=args.benchmark_output.resolve(),
        conflicts_output=args.conflicts_output.resolve(),
        dashboard_output=args.dashboard_output.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
