# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Selected Judge v0.7 execution with deterministic recovery paths."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from eval.dedup.core.validation import DedupEvaluationError, require, sha256_file, sha256_json, write_json_atomic
from eval.dedup.judging import anchor_ids, coverage_format_recovery
from eval.dedup.judging import repetition_recovery as repetition
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION, contract

VERSION = JUDGE_CONTRACT_VERSION
EDITABLE_FIELDS = frozenset(("a_context_span_id", "b_context_span_id", "explanation"))


def repair_messages(payload: dict, raw: dict, render_coverage: Callable) -> list[dict]:
    """Request one evidence-only correction without changing the decision fields."""
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


def execute_case(
    root: Path,
    row: dict,
    endpoint: str,
    frozen: dict,
    render_coverage: Callable,
    *,
    version: str = VERSION,
) -> dict:
    """Execute one immutable contract row; ``version`` supports legacy replay."""
    key, payload = row["canonical_pair_id"], row["payload"]
    result = {
        "canonical_pair_id": key,
        "review_id": row["review_id"],
        "version": version,
        "status": "VALID",
        "components": {},
        "stages": [],
        "main_attempts": [],
    }

    def call(stage: str, request: dict) -> dict:
        require(
            all(request[k] == v for k, v in contract.GENERATION.items()), "EXP1_GENERATION", "frozen generation only"
        )
        receipt = contract.collect(root, key + "-" + stage, request, endpoint)
        result["stages"].append(
            {"stage": stage, "response_sha256": sha256_file(root / "responses" / (key + "-" + stage + ".json"))}
        )
        require(receipt["status"] == "RECEIVED", receipt.get("error_code", "EXP1_RESPONSE"), "fresh response required")
        return receipt["raw_response"]

    def review(stage: str, messages: list[dict], schema: dict) -> dict:
        request = contract.body(messages, schema)
        response = call(stage, request)
        finding = coverage_format_recovery.detect(response) if stage == "coverage" else None
        if finding is not None:
            result["coverage_format_repair"] = {
                "trigger_stage": stage,
                "diagnostic": finding,
                "repair_stage": "coverage-format-repair",
                "attempts": 1,
            }
            response = call("coverage-format-repair", coverage_format_recovery.repair_request(request))
        choice = response["choices"][0]
        require(
            choice["finish_reason"] == "stop", "EXP1_STAGE_FINISH", "historical critic requires a complete response"
        )
        return contract.strict_json(choice["message"]["content"])

    try:
        require(frozen["request_sha256"] == sha256_json(frozen["body"]), "EXP1_MAIN_BINDING", "frozen initial request")
        feedback = None
        for outer in range(1, 4):
            inner = 0
            messages = frozen["body"]["messages"] if outer == 1 else contract.main_messages(payload, feedback)

            def main_call(request: dict, *, attempt: int = outer) -> dict:
                nonlocal inner
                inner += 1
                if attempt == inner == 1:
                    require(request == frozen["body"], "EXP1_MAIN_INITIAL", "exact frozen initial body")
                stage = f"main-{attempt:02d}-{inner:02d}"
                response = call(stage, request)
                finding = repetition.detect(response)
                if finding is not None:
                    raise repetition.TruncatedRepetition(finding, stage)  # noqa: TRY301 - escape native SDK correction
                return response

            try:
                raw = contract.generate_main(messages, main_call)
                public = contract.critic.main_decision(raw, payload)
                result["raw_main"] = raw
                result["main_attempts"].append({"outer": outer, "requests": inner, "status": "VALID"})
                break
            except repetition.TruncatedRepetition as exc:
                result["main_attempts"].append(
                    {
                        "outer": outer,
                        "requests": inner,
                        "status": "FAILURE",
                        "feedback": {
                            "code": "MAIN_TRUNCATED_REPETITION",
                            "message": "repetitive incomplete output omitted",
                        },
                    }
                )
                result["main_repetition_repair"] = {
                    "trigger_stage": exc.stage,
                    "diagnostic": exc.diagnostic,
                    "repair_stage": "main-repetition-repair",
                    "attempts": 1,
                }
                attempt_record = {
                    "outer": outer,
                    "requests": 1,
                    "stage": "main-repetition-repair",
                    "status": "FAILURE",
                }
                result["main_attempts"].append(attempt_record)
                response = call("main-repetition-repair", repetition.repair_request(frozen["body"]))
                choice = response["choices"][0]
                require(
                    choice["finish_reason"] == "stop",
                    "MAIN_REPETITION_REPAIR_INCOMPLETE",
                    "one complete format-only repair required; no further main retries",
                )
                raw = contract.main_toolchain()[1].parse(choice["message"]["content"]).model_dump()
                public = contract.critic.main_decision(raw, payload)
                contract.validate_judge_output(public)
                contract.validate_evidence_offsets(public, payload)
                result["raw_main"] = raw
                attempt_record["status"] = "VALID"
                break
            except Exception as exc:
                feedback = contract.local_ndd.safe_retry_feedback(exc)
                result["main_attempts"].append(
                    {"outer": outer, "requests": inner, "status": "FAILURE", "feedback": feedback}
                )
                if outer == 3:
                    raise
        result["components"]["main"] = deepcopy(public)
        if contract.critic.route(public, payload) == "REVIEW_POSITIVE_DIRECTIONS_ONLY":
            raw = review("coverage", render_coverage(payload), contract.coverage.response_schema())
            normalized, corrections = anchor_ids.normalize_shared_anchor_ids(payload, raw)
            if corrections:
                repair_contract = anchor_ids.ZERO_PADDING_CONTRACT
                if any(c.get("action") == "remove_exact_duplicate" for c in corrections):
                    repair_contract = anchor_ids.EXACT_DUPLICATE_CONTRACT
                elif any(c["original"].isdigit() for c in corrections):
                    repair_contract = anchor_ids.CONTRACT
                result["coverage_anchor_id_repair"] = {
                    "contract_version": repair_contract,
                    "corrections": corrections,
                    "original_review": deepcopy(raw),
                    "normalized_review": deepcopy(normalized),
                    "additional_model_calls": 0,
                }
                raw = normalized
            try:
                public, rule = contract.coverage.apply_review(public, payload, raw)
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
                    contract.coverage.response_schema(),
                )
                result["coverage_evidence_repair"]["repaired_review"] = deepcopy(repaired)
                require(
                    set(repaired) == set(raw)
                    and all(repaired[k] == v for k, v in raw.items() if k not in EDITABLE_FIELDS),
                    "EXP1_EVIDENCE_REPAIR_SCOPE",
                    "only context witnesses and explanation may change",
                )
                public, rule = contract.coverage.apply_review(public, payload, repaired)
                raw = repaired
            result["coverage_review"], result["coverage_rule"] = raw, rule
        result["components"]["coverage"] = deepcopy(public)
        if contract.scope.route(public, payload) == "REVIEW_BILATERAL_SUBJECTS":
            raw = review(
                "subject",
                contract.subject.messages(
                    payload, (contract.PROMPTS / "subject_system.txt").read_text(encoding="utf-8")
                ),
                contract.subject.response_schema(payload),
            )
            result["subject_proposal"] = raw
            bound = {**payload, "subject_proposal": raw}
            if contract.verifier.route(public, bound) == "VERIFY_FIXED_SUBJECT_VETO":
                verification = review(
                    "verifier",
                    contract.verifier.messages(
                        bound, (contract.PROMPTS / "subject_verifier_system.txt").read_text(encoding="utf-8")
                    ),
                    contract.verifier.response_schema(),
                )
                public, rule = contract.verifier.apply_review(public, bound, verification)
                result["verifier_review"], result["verifier_rule"] = verification, rule
        contract.validate_judge_output(public)
        contract.validate_evidence_offsets(public, payload)
        result["public"] = public
    except Exception as exc:  # noqa: BLE001 - failures remain in the full evaluation denominator
        result.update(
            status="ENGINEERING_FAILURE",
            error_code=exc.issue.code if isinstance(exc, DedupEvaluationError) else type(exc).__name__,
            public=contract.unresolved_judge_output(),
        )
    write_json_atomic(root / "results" / (key + ".json"), result)
    return result
