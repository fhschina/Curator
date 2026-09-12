# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in exp1 checkpoint differing only at the unsupported-conflict evidence failure."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from eval.dedup.analysis import exp1_reproduction_runtime as old
from eval.dedup.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic

VERSION = old.VERSION + "+conflict-evidence-fix1"
EDITABLE_FIELDS = frozenset(("a_context_span_id", "b_context_span_id", "explanation"))


def repair_messages(payload: dict, raw: dict, render_coverage: Callable) -> list[dict]:
    return [
        *render_coverage(payload),
        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
        {
            "role": "user",
            "content": (
                "Local validation failed: CRITIC_SCOPE_CONFLICT. The conflict has only shared "
                "context witnesses, with no unique differing value selected. Return the same "
                "eight-field judgment. Change ONLY a_context_span_id, b_context_span_id and "
                "explanation. Preserve conflict, both loss IDs, overlap_basis and "
                "shared_anchor_ids exactly. Select existing own-side context spans identifying "
                "the actual conflicting subjects/values, including at least one unique delta. "
                "Do not invent evidence or infer hidden context. If the existing claim cannot "
                "be supported, leave the witnesses unchanged and explain the insufficiency; "
                "the result will remain a validation failure."
            ),
        },
    ]


def execute_case(root: Path, row: dict, endpoint: str, frozen: dict, render_coverage: Callable) -> dict:
    key, payload = row["canonical_pair_id"], row["payload"]
    result = {
        "canonical_pair_id": key,
        "review_id": row["review_id"],
        "version": VERSION,
        "status": "VALID",
        "components": {},
        "stages": [],
        "main_attempts": [],
    }

    def call(stage: str, request: dict) -> dict:
        require(all(request[k] == v for k, v in old.GENERATION.items()), "EXP1_GENERATION", "frozen generation only")
        receipt = old.collect(root, key + "-" + stage, request, endpoint)
        result["stages"].append(
            {"stage": stage, "response_sha256": sha256_file(root / "responses" / (key + "-" + stage + ".json"))}
        )
        require(receipt["status"] == "RECEIVED", receipt.get("error_code", "EXP1_RESPONSE"), "fresh response required")
        return receipt["raw_response"]

    def review(stage: str, messages: list[dict], schema: dict) -> dict:
        response = call(stage, old.body(messages, schema))
        choice = response["choices"][0]
        require(
            choice["finish_reason"] == "stop", "EXP1_STAGE_FINISH", "historical critic requires a complete response"
        )
        return old.common.strict_json(choice["message"]["content"])

    try:
        require(frozen["request_sha256"] == sha256_json(frozen["body"]), "EXP1_MAIN_BINDING", "frozen initial request")
        feedback = None
        for outer in range(1, 4):
            inner = 0
            messages = frozen["body"]["messages"] if outer == 1 else old.main_messages(payload, feedback)

            def main_call(request: dict, *, attempt: int = outer) -> dict:
                nonlocal inner
                inner += 1
                if attempt == inner == 1:
                    require(request == frozen["body"], "EXP1_MAIN_INITIAL", "exact frozen initial body")
                return call(f"main-{attempt:02d}-{inner:02d}", request)

            try:
                raw = old.generate_main(messages, main_call)
                public = old.common.critic.main_decision(raw, payload)
                result["raw_main"] = raw
                result["main_attempts"].append({"outer": outer, "requests": inner, "status": "VALID"})
                break
            except Exception as exc:
                feedback = old.local_ndd._safe_retry_feedback(exc)
                result["main_attempts"].append(
                    {"outer": outer, "requests": inner, "status": "FAILURE", "feedback": feedback}
                )
                if outer == 3:
                    raise
        result["components"]["main"] = deepcopy(public)
        if old.common.critic.route(public, payload) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            raw = review("coverage", render_coverage(payload), old.coverage.response_schema())
            try:
                public, rule = old.coverage.apply_review(public, payload, raw)
            except DedupEvaluationError as exc:
                if exc.issue.code != "CRITIC_SCOPE_CONFLICT":
                    raise
                result["coverage_evidence_repair"] = {
                    "trigger": exc.issue.code,
                    "original_review": deepcopy(raw),
                    "attempts": 1,
                }
                repaired = review(
                    "coverage-evidence-repair",
                    repair_messages(payload, raw, render_coverage),
                    old.coverage.response_schema(),
                )
                result["coverage_evidence_repair"]["repaired_review"] = deepcopy(repaired)
                require(
                    set(repaired) == set(raw)
                    and all(repaired[k] == v for k, v in raw.items() if k not in EDITABLE_FIELDS),
                    "EXP1_EVIDENCE_REPAIR_SCOPE",
                    "only context witnesses and explanation may change",
                )
                public, rule = old.coverage.apply_review(public, payload, repaired)
                raw = repaired
            result["coverage_review"], result["coverage_rule"] = raw, rule
        result["components"]["coverage"] = deepcopy(public)
        if old.scope.route(public, payload) == "REVIEW_BILATERAL_SUBJECTS":
            raw = review(
                "subject",
                old.subject.messages(payload, (old.RESOURCES / "v06212_subject_binding_v1_system.txt").read_text()),
                old.subject.response_schema(payload),
            )
            result["subject_proposal"] = raw
            bound = {**payload, "subject_proposal": raw}
            if old.verifier.route(public, bound) == "VERIFY_FIXED_SUBJECT_VETO":
                verification = review(
                    "verifier",
                    old.verifier.messages(
                        bound, (old.RESOURCES / "v06212_subject_proof_verifier_v4_system.txt").read_text()
                    ),
                    old.verifier.response_schema(),
                )
                public, rule = old.verifier.apply_review(public, bound, verification)
                result["verifier_review"], result["verifier_rule"] = verification, rule
        old.validate_judge_output_v3(public)
        old.validate_evidence_offsets(public, payload)
        result["public"] = public
    except Exception as exc:  # noqa: BLE001 - failures remain in the full evaluation denominator
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=exc.issue.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__,
            public=old.unresolved_judge_output_v3(),
        )
    write_json_atomic(root / "results" / (key + ".json"), result)
    return result
