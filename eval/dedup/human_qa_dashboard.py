# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reviewer-blind, self-contained Human QA dashboard."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

from eval.dedup.contracts import REASON_CODES, canonical_json_bytes
from eval.dedup.validation import require, write_text_atomic

HUMAN_QA_DASHBOARD_VERSION = "dedup-human-qa-dashboard-v1"
PACKET_DISPLAY_NAMES = {
    "human_qa_blind": "Blind sample",
    "human_qa_diagnostic": "Diagnostic set",
}

REASON_CODE_GUIDANCE = {
    "NUMBER_CHANGE": "A number changes the meaning or factual value.",
    "DATE_TIME_CHANGE": "A date or time differs materially.",
    "PRODUCT_VERSION_CHANGE": "The documents describe different product versions.",
    "URL_CHANGE": "A URL difference is material to the content.",
    "NAMED_ENTITY_CHANGE": "A person, place, organization, or other named entity differs.",
    "NEGATION_CHANGE": "Negation reverses or materially changes the meaning.",
    "CODE_LITERAL_CHANGE": "Source-code literals or commands differ materially.",
    "CODE_OUTPUT_CHANGE": "Code output or execution results differ materially.",
    "INSERTION_DELETION": "One document adds or removes meaningful content.",
    "BOILERPLATE": "Overlap is primarily navigation, template, or boilerplate text.",
    "PARSER_NOISE": "The difference appears to come from extraction or parser noise.",
    "LANGUAGE_MISMATCH": "The documents use different languages in a material way.",
    "TOPIC_ONLY": "The documents share a topic but not replaceable content.",
    "INSUFFICIENT_EVIDENCE": "The visible evidence is insufficient to decide safely.",
    "OTHER_MATERIAL": "Another material difference is present; explain it in notes.",
}


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _read_packet(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), "QA_PACKET_NOT_AVAILABLE", "Human QA packet does not exist", path=str(path))
    with path.open("r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    require(rows, "QA_PACKET_EMPTY", "Human QA packet is empty", path=str(path))
    qa_ids = [str(row.get("qa_pair_id", "")) for row in rows]
    require(all(qa_ids), "QA_PACKET_INVALID", "Human QA packet contains a missing pair ID", path=str(path))
    require(len(qa_ids) == len(set(qa_ids)), "QA_PACKET_INVALID", "Human QA pair IDs are not unique", path=str(path))
    for row in rows:
        payload = row.get("visible_payload")
        require(isinstance(payload, dict), "QA_PACKET_INVALID", "Human QA row has no visible payload")
        windows = payload.get("long_document_evidence", {}).get("windows", [])
        for key, side in (("document_a", "A"), ("document_b", "B")):
            document = payload.get(key)
            require(isinstance(document, dict), "QA_PACKET_INVALID", "Human QA row has a missing document", side=key)
            has_windows = any(item.get("side") == side and isinstance(item.get("text"), str) for item in windows)
            require(
                isinstance(document.get("text"), str) or has_windows,
                "QA_PACKET_INVALID",
                "Human QA document has no visible text",
                side=side,
            )
    return rows


def _visible_document(payload: dict[str, Any], *, key: str, side: str) -> dict[str, Any]:
    document = payload[key]
    text = document.get("text")
    if not isinstance(text, str):
        windows = [
            item
            for item in payload.get("long_document_evidence", {}).get("windows", [])
            if item.get("side") == side and isinstance(item.get("text"), str)
        ]
        text = "\n\n".join(
            f"[Visible window {index}: chars {item.get('start_char', '?')}-{item.get('end_char', '?')}]\n{item['text']}"
            for index, item in enumerate(windows, start=1)
        )
    return {"metadata": document.get("metadata", {}), "text": text}


def _dashboard_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, row in enumerate(rows, start=1):
        payload = row["visible_payload"]
        records.append(
            {
                "qa_pair_id": row["qa_pair_id"],
                "position": index,
                "document_a": _visible_document(payload, key="document_a", side="A"),
                "document_b": _visible_document(payload, key="document_b", side="B"),
            }
        )
    return records


def human_qa_dashboard_html(
    *,
    evaluation_run_id: str,
    packets: list[dict[str, Any]],
) -> str:
    """Render a self-contained dashboard without Judge or SUT conclusions."""

    packet_data = []
    for packet in packets:
        records = _dashboard_records(packet["rows"])
        packet_data.append(
            {
                "label": packet["label"],
                "title": PACKET_DISPLAY_NAMES.get(packet["label"], packet["label"].replace("_", " ").title()),
                "digest": hashlib.sha256(canonical_json_bytes(records)).hexdigest()[:16],
                "records": records,
            }
        )
    reason_guidance = [
        {"code": code, "description": REASON_CODE_GUIDANCE[code]}
        for code in REASON_CODE_GUIDANCE
        if code in REASON_CODES
    ]
    template = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none';style-src 'unsafe-inline';script-src 'unsafe-inline';base-uri 'none';form-action 'none'">
<title>Human QA Review</title>
<style>
:root{color-scheme:light;--canvas:#eef1f5;--paper:#fff;--ink:#17212b;--muted:#667281;--line:#d7dde5;--soft:#f6f8fa;--accent:#176b5b;--accent-soft:#e8f4f0;--warn:#9a6700;--warn-soft:#fff3cd;--danger:#b42318;--danger-soft:#fff1f0;--focus:#005fcc;--shadow:0 8px 24px rgba(20,31,43,.08)}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select,textarea{font:inherit;color:inherit}button{cursor:pointer}.topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:14px 22px;background:var(--paper);border-bottom:1px solid var(--line)}h1{margin:0;font-size:20px;letter-spacing:-.02em}.subtitle{margin-top:2px;color:var(--muted);font-size:12px}.actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.button{min-height:36px;padding:7px 12px;border:1px solid var(--line);border-radius:7px;background:var(--paper);font-weight:650}.button:hover{border-color:var(--accent);color:var(--accent)}.button.primary{border-color:var(--accent);background:var(--accent);color:#fff}.button.danger{color:var(--danger)}.hidden{display:none!important}.stats{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:10px;padding:14px 22px 0}.stat{padding:10px 12px;background:var(--paper);border:1px solid var(--line);border-radius:8px}.stat span{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}.stat strong{display:block;margin-top:2px;font-size:21px;font-variant-numeric:tabular-nums}.progress{height:5px;margin:10px 22px 0;background:#dfe5eb;border-radius:999px;overflow:hidden}.progress span{display:block;height:100%;width:0;background:var(--accent)}.workspace{display:grid;grid-template-columns:290px minmax(0,1fr);gap:14px;padding:14px 22px 24px}.queue-panel,.detail-panel{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:9px;box-shadow:var(--shadow)}.queue-panel{position:sticky;top:80px;align-self:start;max-height:calc(100vh - 104px);display:flex;flex-direction:column;overflow:hidden}.queue-tools{padding:12px;border-bottom:1px solid var(--line)}.queue-tools input,.queue-tools select{width:100%;min-height:36px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--paper)}.queue-tools select{margin-top:7px}.queue-list{overflow:auto}.queue-item{display:grid;grid-template-columns:30px 1fr auto;gap:8px;align-items:center;width:100%;padding:10px 12px;border:0;border-bottom:1px solid var(--line);background:var(--paper);text-align:left}.queue-item:hover,.queue-item.selected{background:var(--accent-soft)}.queue-number{color:var(--muted);font-variant-numeric:tabular-nums}.queue-id{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace}.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:#edf0f3;color:var(--muted);font-size:10px;font-weight:750}.badge.labeled{background:var(--accent-soft);color:var(--accent)}.badge.ambiguous{background:var(--warn-soft);color:var(--warn)}.badge.draft{background:#eaf2ff;color:#175cd3}.queue-empty{padding:28px 14px;color:var(--muted);text-align:center}.detail-panel{padding:18px}.pair-head{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;padding-bottom:14px;border-bottom:1px solid var(--line)}h2{margin:0;font-size:18px}.pair-id{margin-top:4px;color:var(--muted);font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}.blind-note{max-width:390px;padding:8px 10px;border-radius:7px;background:var(--soft);color:var(--muted);font-size:12px}.documents{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.document{min-width:0;border:1px solid var(--line);border-radius:8px;background:var(--soft);overflow:hidden}.document-head{padding:11px 12px;border-bottom:1px solid var(--line);background:var(--paper)}h3{margin:0;font-size:15px}.metadata{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px;color:var(--muted);font-size:11px}.url{display:block;margin-top:6px;color:var(--accent);overflow-wrap:anywhere}.text{height:390px;margin:0;padding:13px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;background:var(--soft);font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.review{margin-top:14px;padding:15px;border:1px solid var(--line);border-radius:8px}.review-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}.review-head p{margin:3px 0 0;color:var(--muted);font-size:12px}.field-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:14px}.field-card{min-width:0;padding:11px;border:1px solid var(--line);border-radius:7px}.field-card legend,.field-card>label{display:block;margin-bottom:8px;font-weight:700}.field-help{display:block;margin-top:-5px;margin-bottom:8px;color:var(--muted);font-size:11px}.segments{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.segments label{position:relative}.segments input{position:absolute;opacity:0;pointer-events:none}.segments span{display:block;padding:7px 5px;border:1px solid var(--line);border-radius:6px;background:var(--paper);text-align:center;font-size:12px}.segments input:checked+span{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:750}.field-card select{width:100%;min-height:36px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--paper)}.reason-block{margin-top:14px}.reason-block h3{font-size:14px}.reason-block p{margin:3px 0 9px;color:var(--muted);font-size:12px}.reasons{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.reason{position:relative;display:block;padding:8px 9px 8px 31px;border:1px solid var(--line);border-radius:7px;background:var(--paper);cursor:pointer}.reason:hover{border-color:var(--accent)}.reason input{position:absolute;left:9px;top:10px}.reason strong{display:block;font:11px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace}.reason small{display:block;margin-top:3px;color:var(--muted);font-size:10px;line-height:1.3}.notes-label{display:block;margin-top:14px;font-weight:700}.notes{width:100%;min-height:80px;margin-top:6px;padding:9px;border:1px solid var(--line);border-radius:7px;resize:vertical;background:var(--paper)}.review-footer{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;padding-top:12px;border-top:1px solid var(--line)}.review-status{color:var(--muted);font-size:12px}.review-actions{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.error{margin-top:10px;padding:8px 10px;border-radius:6px;background:var(--danger-soft);color:var(--danger);font-weight:650}.empty-detail{padding:80px 20px;color:var(--muted);text-align:center}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:3px solid color-mix(in srgb,var(--focus) 35%,transparent);outline-offset:2px}
.packet-select{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:12px;font-weight:700}.packet-select select{min-height:36px;padding:7px 30px 7px 9px;border:1px solid var(--accent);border-radius:7px;background:var(--paper);color:var(--ink);font-weight:700}.required{color:var(--danger)}
@media(max-width:980px){.workspace{grid-template-columns:1fr}.queue-panel{position:static;max-height:280px}.documents{grid-template-columns:1fr}.text{height:310px}.reasons{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){.topbar{align-items:flex-start;flex-direction:column}.stats{grid-template-columns:1fr 1fr}.workspace,.stats{padding-left:12px;padding-right:12px}.progress{margin-left:12px;margin-right:12px}.detail-panel{padding:12px}.field-grid,.reasons{grid-template-columns:1fr}.review-footer{align-items:flex-start;flex-direction:column}.review-actions{justify-content:flex-start}}
</style></head><body>
<header class="topbar"><div><h1>Human QA Review</h1><div class="subtitle">Run __RUN_ID__ · reviewer-blind packet · __VERSION__</div></div><div class="actions"><label class="packet-select">Review set<select id="packet-selector" aria-label="Review set"></select></label><button class="button" id="import-button" type="button">Import labels CSV</button><input class="hidden" id="import-file" type="file" accept=".csv,text/csv"><button class="button" id="export-csv" type="button">Export labels CSV</button><button class="button primary" id="export-json" type="button">Export packet JSON</button></div></header>
<section class="stats" aria-label="Review progress"><div class="stat"><span>Reviewed</span><strong id="stat-reviewed">0 / 0</strong></div><div class="stat"><span>Labeled</span><strong id="stat-labeled">0</strong></div><div class="stat"><span>Ambiguous</span><strong id="stat-ambiguous">0</strong></div><div class="stat"><span>Remaining</span><strong id="stat-remaining">0</strong></div></section><div class="progress" aria-hidden="true"><span id="progress-bar"></span></div>
<main class="workspace"><aside class="queue-panel" aria-label="Pair queue"><div class="queue-tools"><input id="search" type="search" placeholder="Search pair ID, URL, or text"><select id="status-filter"><option value="">All review states</option><option value="PENDING">Pending</option><option value="DRAFT">Draft</option><option value="LABELED">Labeled</option><option value="AMBIGUOUS">Ambiguous</option></select></div><div class="queue-list" id="queue-list"></div><div class="queue-empty hidden" id="queue-empty">No pairs match this filter.</div></aside><section class="detail-panel" id="detail"><div class="empty-detail">Choose a pair to start reviewing.</div></section></main>
<script>
const PACKETS=__PACKET_DATA__,RUN_ID=__RUN_ID_JSON__,VERSION=__VERSION_JSON__,REASONS=__REASON_DATA__;
const FIELDS=["same_duplicate_group","a_can_replace_b","b_can_replace_a","relation_type","material_difference","fuzzy_scope"],REQUIRED_FIELDS=FIELDS.slice(0,3),LABEL_FIELDS=["qa_pair_id",...FIELDS,"reason_codes","reviewer_status","notes"],CSV_FIELDS=LABEL_FIELDS;
const ANSWERS=["YES","NO","UNRESOLVED"],OPTIONS={relation_type:["EXACT","CANONICAL_EXACT","NEAR_SURFACE","CONTAINMENT","VERSION_RELATED","RELATED_NON_DUPLICATE","UNRELATED","UNRESOLVED"],material_difference:["NONE","MINOR","MAJOR","UNRESOLVED"],fuzzy_scope:["IN_SCOPE","BORDERLINE","OUT_OF_SCOPE","UNRESOLVED"]};
const DEFAULT=()=>({same_duplicate_group:"",a_can_replace_b:"",b_can_replace_a:"",relation_type:"",material_difference:"",fuzzy_scope:"",reason_codes:[],reviewer_status:"PENDING",notes:"",updated_at:""});
let packet=null,PAIRS=[],PAIR_IDS=new Set(),STORE="",reviews={},selected="";const queue=document.getElementById("queue-list"),detail=document.getElementById("detail"),search=document.getElementById("search"),statusFilter=document.getElementById("status-filter"),packetSelector=document.getElementById("packet-selector");
for(const item of PACKETS)packetSelector.appendChild(new Option(`${item.title} (${item.records.length})`,item.label));
function create(tag,cls,text){const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=String(text);return node}function review(id){return {...DEFAULT(),...(reviews[id]||{})}}function load(){try{const parsed=JSON.parse(localStorage.getItem(STORE)||"{}");return parsed&&typeof parsed==="object"?parsed:{}}catch(_){return{}}}function persist(message="Saved in this browser."){try{localStorage.setItem(STORE,JSON.stringify(reviews));setStatus(message)}catch(_){setStatus("Browser storage unavailable. Export before closing.",true)}}function setStatus(message,isError=false){const node=document.getElementById("review-message");if(node){node.textContent=message;node.style.color=isError?"var(--danger)":""}}
function activatePacket(label,pairId="",updateHash=true){packet=PACKETS.find(item=>item.label===label)||PACKETS[0];PAIRS=packet.records;PAIR_IDS=new Set(PAIRS.map(item=>item.qa_pair_id));STORE=`dedup-human-qa:${RUN_ID}:${packet.digest}:${VERSION}`;reviews=load();selected=PAIR_IDS.has(pairId)?pairId:PAIRS[0]?.qa_pair_id||"";packetSelector.value=packet.label;document.getElementById("export-json").textContent=`Export ${packet.title} JSON`;search.value="";statusFilter.value="";renderQueue();renderDetail();if(updateHash)updateLocation()}
function stateFor(id){const r=review(id);if(r.reviewer_status==="LABELED"||r.reviewer_status==="AMBIGUOUS")return r.reviewer_status;return FIELDS.some(f=>r[f])||r.reason_codes.length||r.notes?"DRAFT":"PENDING"}function filtered(){const q=search.value.trim().toLowerCase(),status=statusFilter.value;return PAIRS.filter(p=>{const hay=[p.qa_pair_id,p.document_a?.text,p.document_b?.text,...Object.values(p.document_a?.metadata||{}),...Object.values(p.document_b?.metadata||{})].join(" ").toLowerCase();return(!q||hay.includes(q))&&(!status||stateFor(p.qa_pair_id)===status)})}
function stats(){const states=PAIRS.map(p=>stateFor(p.qa_pair_id)),labeled=states.filter(x=>x==="LABELED").length,ambiguous=states.filter(x=>x==="AMBIGUOUS").length,reviewed=labeled+ambiguous;document.getElementById("stat-reviewed").textContent=`${reviewed.toLocaleString()} / ${PAIRS.length.toLocaleString()}`;document.getElementById("stat-labeled").textContent=labeled.toLocaleString();document.getElementById("stat-ambiguous").textContent=ambiguous.toLocaleString();document.getElementById("stat-remaining").textContent=(PAIRS.length-reviewed).toLocaleString();document.getElementById("progress-bar").style.width=`${PAIRS.length?reviewed/PAIRS.length*100:0}%`}
function renderQueue(){const rows=filtered();queue.replaceChildren();for(const pair of rows){const state=stateFor(pair.qa_pair_id),button=create("button",`queue-item${selected===pair.qa_pair_id?" selected":""}`);button.type="button";button.append(create("span","queue-number",String(pair.position).padStart(3,"0")),create("span","queue-id",pair.qa_pair_id),create("span",`badge ${state.toLowerCase()}`,state));button.onclick=()=>selectPair(pair.qa_pair_id);queue.appendChild(button)}document.getElementById("queue-empty").classList.toggle("hidden",rows.length!==0);stats()}
function safeLink(parent,url){if(!url){parent.appendChild(create("span","url","No URL"));return}try{const parsed=new URL(url);if(["http:","https:"].includes(parsed.protocol)){const link=create("a","url",url);link.href=parsed.href;link.target="_blank";link.rel="noopener noreferrer";parent.appendChild(link);return}}catch(_){}parent.appendChild(create("span","url",url))}
function documentCard(side,document){const card=create("article","document"),head=create("div","document-head"),meta=document.metadata||{},tags=create("div","metadata");head.appendChild(create("h3","",`Document ${side}`));for(const value of[meta.language,meta.character_count!==undefined?`${Number(meta.character_count).toLocaleString()} chars`:"",meta.crawl_timestamp].filter(Boolean))tags.appendChild(create("span","",value));head.appendChild(tags);safeLink(head,meta.url);card.append(head,create("pre","text",document.text||"No visible text."));return card}
function radioField(name,title,help,rv){const field=create("fieldset","field-card"),legend=create("legend","",title),hint=create("span","field-help",help),segments=create("div","segments");field.append(legend,hint);for(const value of ANSWERS){const label=create("label"),input=create("input"),text=create("span","",value);input.type="radio";input.name=name;input.value=value;input.checked=rv[name]===value;input.onchange=saveDraft;label.append(input,text);segments.appendChild(label)}field.appendChild(segments);return field}
function selectField(name,title,help,rv){const card=create("div","field-card"),label=create("label","",title),hint=create("span","field-help",help),select=create("select");select.id=name;select.appendChild(new Option("Choose…",""));for(const value of OPTIONS[name])select.appendChild(new Option(value,value));select.value=rv[name];select.onchange=saveDraft;card.append(label,hint,select);return card}
function formValue(){const current=review(selected),value={...current};for(const name of FIELDS.slice(0,3)){value[name]=detail.querySelector(`input[name="${name}"]:checked`)?.value||""}for(const name of FIELDS.slice(3))value[name]=document.getElementById(name)?.value||"";value.reason_codes=[...detail.querySelectorAll('input[name="reason_codes"]:checked')].map(x=>x.value);value.notes=document.getElementById("notes")?.value||"";return value}
function saveDraft(){if(!selected)return;const value=formValue();value.reviewer_status=value.reviewer_status==="AMBIGUOUS"||value.reviewer_status==="LABELED"?"PENDING":value.reviewer_status;value.updated_at=new Date().toISOString();reviews[selected]=value;if(stateFor(selected)==="PENDING")delete reviews[selected];persist();renderQueue()}
function validate(value){const missing=REQUIRED_FIELDS.filter(field=>!value[field]);return missing.length?`Complete the three required decisions before saving (${missing.join(", ")}).`:""}
function saveLabeled(){const value=formValue(),error=validate(value);if(error){showError(error);return}value.reviewer_status="LABELED";value.updated_at=new Date().toISOString();reviews[selected]=value;persist("Label saved.");showError("");renderQueue();nextPair()}
function markAmbiguous(){const value=formValue();for(const field of REQUIRED_FIELDS)value[field]="AMBIGUOUS";value.reviewer_status="AMBIGUOUS";value.updated_at=new Date().toISOString();reviews[selected]=value;persist("Marked ambiguous.");renderQueue();nextPair()}
function clearReview(){delete reviews[selected];persist("Review cleared.");renderQueue();renderDetail()}
function showError(message){const node=document.getElementById("form-error");if(!node)return;node.textContent=message;node.classList.toggle("hidden",!message)}function nextPair(){const index=PAIRS.findIndex(p=>p.qa_pair_id===selected),next=[...PAIRS.slice(index+1),...PAIRS.slice(0,index)].find(p=>!["LABELED","AMBIGUOUS"].includes(stateFor(p.qa_pair_id)));if(next)selectPair(next.qa_pair_id);else renderDetail()}
function renderDetail(){const pair=PAIRS.find(p=>p.qa_pair_id===selected);detail.replaceChildren();if(!pair){detail.appendChild(create("div","empty-detail","Choose a pair to start reviewing."));return}const rv=review(selected),head=create("div","pair-head"),title=create("div");title.append(create("h2","",`Pair ${pair.position} of ${PAIRS.length}`),create("div","pair-id",pair.qa_pair_id));head.append(title,create("div","blind-note","Reviewer-blind: no LLM Judge verdict, fuzzy-dedup outcome, or sampling stratum is shown."));const docs=create("div","documents");docs.append(documentCard("A",pair.document_a),documentCard("B",pair.document_b));const form=create("section","review"),formHead=create("div","review-head"),formTitle=create("div");formTitle.append(create("h2","","Human decision"),create("p","","The first three decisions are required; relation, difference, scope, reason codes, and notes are optional."));formHead.append(formTitle,create("span",`badge ${stateFor(selected).toLowerCase()}`,stateFor(selected)));const grid=create("div","field-grid");grid.append(radioField("same_duplicate_group","Same duplicate group? *","Required · Would you group these as the same duplicate content?",rv),radioField("a_can_replace_b","Can A replace B? *","Required · Could Document A safely stand in for Document B?",rv),radioField("b_can_replace_a","Can B replace A? *","Required · Could Document B safely stand in for Document A?",rv),selectField("relation_type","Relation type","Optional · Choose the closest content relationship.",rv),selectField("material_difference","Material difference","Optional · How consequential are the differences?",rv),selectField("fuzzy_scope","Fuzzy-dedup scope","Optional · Should fuzzy deduplication cover this pair?",rv));const reasonBlock=create("div","reason-block");reasonBlock.append(create("h3","","Reason codes · Optional"),create("p","","Select any legacy human-review code that helps explain the decision. These optional labels are independent of the versioned automated Judge reasons."));const reasons=create("div","reasons");for(const item of REASONS){const label=create("label","reason"),input=create("input"),body=create("span");input.type="checkbox";input.name="reason_codes";input.value=item.code;input.checked=rv.reason_codes.includes(item.code);input.onchange=saveDraft;body.append(create("strong","",item.code),create("small","",item.description));label.append(input,body);reasons.appendChild(label)}reasonBlock.appendChild(reasons);const notesLabel=create("label","notes-label","Notes · Optional"),notes=create("textarea","notes");notes.id="notes";notes.placeholder="Explain OTHER_MATERIAL, uncertainty, or any decision detail…";notes.value=rv.notes;notes.onchange=saveDraft;const error=create("div","error hidden");error.id="form-error";const footer=create("div","review-footer"),message=create("div","review-status",rv.updated_at?`Last saved ${rv.updated_at}`:"Not saved yet"),actions=create("div","review-actions");message.id="review-message";for(const [label,cls,handler]of[["Clear","button danger",clearReview],["Mark ambiguous","button",markAmbiguous],["Save & next","button primary",saveLabeled]]){const button=create("button",cls,label);button.type="button";button.onclick=handler;actions.appendChild(button)}footer.append(message,actions);form.append(formHead,grid,reasonBlock,notesLabel,notes,error,footer);detail.append(head,docs,form)}
function updateLocation(){const params=new URLSearchParams({packet:packet.label,pair:selected});history.replaceState(null,"",`#${params}`)}function locationState(){const params=new URLSearchParams(location.hash.slice(1));return{packet:params.get("packet")||PACKETS[0]?.label,pair:params.get("pair")||""}}
function selectPair(id){if(!PAIR_IDS.has(id))return;selected=id;renderQueue();renderDetail();updateLocation();document.querySelector(".detail-panel")?.scrollIntoView({block:"start"})}
function csvEscape(value){const text=String(value??"");return /[",\n\r]/.test(text)?`"${text.replaceAll('"','""')}"`:text}function labelRow(pair){const rv=review(pair.qa_pair_id),state=stateFor(pair.qa_pair_id),row={qa_pair_id:pair.qa_pair_id};for(const field of FIELDS)row[field]=state==="PENDING"||state==="DRAFT"?rv[field]||"":rv[field];row.reason_codes=rv.reason_codes.length?JSON.stringify(rv.reason_codes):"";row.reviewer_status=state==="DRAFT"?"PENDING":state;row.notes=rv.notes||"";return row}function exportRows(){return PAIRS.map(labelRow)}function shareRows(){return PAIRS.map(pair=>{const{qa_pair_id,...review}=labelRow(pair);review.reason_codes=review.reason_codes?JSON.parse(review.reason_codes):[];return{qa_pair_id,position:pair.position,review,document_a:pair.document_a,document_b:pair.document_b}})}function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type})),link=create("a");link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),0)}function exportCsv(){const lines=[CSV_FIELDS.join(","),...exportRows().map(row=>CSV_FIELDS.map(field=>csvEscape(row[field])).join(","))];download(`${packet.label}_labels.csv`,"text/csv;charset=utf-8",lines.join("\r\n")+"\r\n")}function exportPacketJson(){download(`${packet.label}_review_packet.json`,"application/json",JSON.stringify({schema_version:"dedup-human-qa-share-v1",dashboard_version:VERSION,run_id:RUN_ID,packet:{label:packet.label,title:packet.title,digest:packet.digest},pairs:shareRows()},null,2))}
function parseCsv(text){const rows=[];let row=[],cell="",quoted=false;for(let i=0;i<text.length;i++){const char=text[i];if(quoted){if(char==='"'&&text[i+1]==='"'){cell+='"';i++}else if(char==='"')quoted=false;else cell+=char}else if(char==='"')quoted=true;else if(char===','){row.push(cell);cell=""}else if(char==='\n'){row.push(cell.replace(/\r$/, ""));rows.push(row);row=[];cell=""}else cell+=char}if(cell||row.length){row.push(cell);rows.push(row)}const headers=rows.shift()||[];return rows.filter(x=>x.some(Boolean)).map(values=>Object.fromEntries(headers.map((header,index)=>[header,values[index]||""])))}function parseReasons(value){if(!value)return[];try{const parsed=JSON.parse(value);if(Array.isArray(parsed))return parsed.filter(code=>REASONS.some(x=>x.code===code))}catch(_){}return value.split("|").map(x=>x.trim()).filter(code=>REASONS.some(x=>x.code===code))}
async function importCsv(file){try{const rows=parseCsv(await file.text());if(!rows.length)throw new Error("CSV has no rows");let imported=0;for(const row of rows){if(!PAIR_IDS.has(row.qa_pair_id))continue;const value=DEFAULT();for(const field of FIELDS)value[field]=row[field]||"";value.reason_codes=parseReasons(row.reason_codes);value.reviewer_status=["LABELED","AMBIGUOUS"].includes(row.reviewer_status)?row.reviewer_status:"PENDING";value.notes=row.notes||"";value.updated_at=new Date().toISOString();reviews[row.qa_pair_id]=value;imported++}persist(`Imported ${imported} matching rows.`);renderQueue();renderDetail()}catch(error){setStatus(`Import failed: ${error.message}`,true)}}
search.oninput=renderQueue;statusFilter.onchange=renderQueue;packetSelector.onchange=()=>activatePacket(packetSelector.value);document.getElementById("export-csv").onclick=exportCsv;document.getElementById("export-json").onclick=exportPacketJson;document.getElementById("import-button").onclick=()=>document.getElementById("import-file").click();document.getElementById("import-file").onchange=event=>{if(event.target.files[0])importCsv(event.target.files[0]);event.target.value=""};document.addEventListener("keydown",event=>{if((event.ctrlKey||event.metaKey)&&event.key==="Enter"){event.preventDefault();saveLabeled()}});window.onhashchange=()=>{const state=locationState();activatePacket(state.packet,state.pair,false)};const initial=locationState();activatePacket(initial.packet,initial.pair,false);updateLocation();
</script></body></html>"""
    return (
        template.replace("__RUN_ID__", html.escape(evaluation_run_id))
        .replace("__PACKET_DATA__", _script_json(packet_data))
        .replace("__RUN_ID_JSON__", _script_json(evaluation_run_id))
        .replace("__VERSION_JSON__", _script_json(HUMAN_QA_DASHBOARD_VERSION))
        .replace("__VERSION__", HUMAN_QA_DASHBOARD_VERSION)
        .replace("__REASON_DATA__", _script_json(reason_guidance))
    )


def publish_human_qa_dashboard(
    *,
    packet_path: Path | None = None,
    destination: Path,
    evaluation_run_id: str,
    packet_label: str | None = None,
    packet_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate a QA packet and publish its reviewer-blind dashboard."""

    require(
        packet_paths is not None or (packet_path is not None and packet_label is not None),
        "QA_DASHBOARD_PACKETS_MISSING",
        "At least one labeled Human QA packet is required",
    )
    sources = packet_paths or {str(packet_label): packet_path}
    packets = [{"label": label, "rows": _read_packet(path)} for label, path in sources.items() if path is not None]
    dashboard = human_qa_dashboard_html(
        evaluation_run_id=evaluation_run_id,
        packets=packets,
    )
    write_text_atomic(destination, dashboard)
    return {
        "dashboard_version": HUMAN_QA_DASHBOARD_VERSION,
        "qa_pairs": sum(len(packet["rows"]) for packet in packets),
        "packets": {packet["label"]: len(packet["rows"]) for packet in packets},
        "destination": str(destination),
    }
