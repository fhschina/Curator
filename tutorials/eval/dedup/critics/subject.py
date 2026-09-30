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

"""Subject-binding proposals and fixed-proof verification over coverage decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from .dedup_adapter import _offset, adapt_alignment, complete_text_equality, require, validate_decision

if TYPE_CHECKING:
    import data_designer.config as dd

BINDINGS = ("NONE", "LIABILITY_PARTY", "POLICY_SERVICE", "ACCESS_TARGET", "FAILED_OBJECT", "RECORD_SUBJECT")
VETO_BINDINGS = {"LIABILITY_PARTY", "POLICY_SERVICE", "FAILED_OBJECT"}
RELATIONS = ("SAME", "DIFFERENT", "UNCERTAIN", "NOT_APPLICABLE")
KINDS = ("NAMED_ACTUAL_TARGET", "GENERIC_ROLE_OR_OBJECT", "UI_LABEL_OR_INSTRUCTION", "UNSUPPORTED")
COMPARISONS = ("SUPPORTED_DIFFERENT_NAMED_TARGETS", "UNSUPPORTED_COMPARISON", "UNCERTAIN")


def response_schema() -> dict[str, Any]:
    fields = {f"{side}_{role}_span_id": {"type": "string"} for side in ("a", "b") for role in ("subject", "predicate")}
    fields.update(
        binding_type={"type": "string", "enum": list(BINDINGS)},
        target_relation={"type": "string", "enum": list(RELATIONS)},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


def verifier_schema() -> dict[str, Any]:
    fields = {f"{side}_subject_kind": {"type": "string", "enum": list(KINDS)} for side in ("a", "b")}
    fields.update(
        comparison={"type": "string", "enum": list(COMPARISONS)},
        explanation={"type": "string", "minLength": 1},
    )
    return {"type": "object", "additionalProperties": False, "required": list(fields), "properties": fields}


def _compile_proposal(
    review: object, packet: dict[str, Any], pair_id: object
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    require(
        isinstance(review, dict)
        and set(review) == set(response_schema()["required"])
        and all(isinstance(value, str) for value in review.values())
        and review["binding_type"] in BINDINGS
        and review["target_relation"] in RELATIONS
        and bool(review["explanation"].strip()),
        "invalid subject review schema",
        pair_id=pair_id,
    )
    spans = {span["span_id"]: span for span in packet["spans"]}
    selected, evidence = {"binding_type_hypothesis": review["binding_type"]}, []
    for side in ("a", "b"):
        selected[side] = {}
        for role in ("subject", "predicate"):
            sid = review[f"{side}_{role}_span_id"]
            if not sid:
                continue
            require(sid in spans, f"unknown subject evidence ID {sid!r}", pair_id=pair_id)
            span = spans[sid]
            require(
                span["kind"] == f"{side.upper()}_ONLY" or (role == "predicate" and span["kind"] == "SHARED"),
                f"wrong side or non-unique subject: {sid}",
                pair_id=pair_id,
            )
            prefix = side + "_" if span["kind"] == "SHARED" else ""
            witness = {"span_id": sid, "quote": span[prefix + "text"]}
            selected[side][role] = witness
            evidence.append(
                {
                    **witness,
                    "side": side.upper(),
                    "role": role,
                    **{key: _offset(span, prefix + key, pair_id) for key in ("start_char", "end_char")},
                }
            )
    if review["target_relation"] != "DIFFERENT":
        return None, evidence, "NO_SUPPORTED_SUBJECT_VETO_KEEP_COVERAGE"
    require(
        review["binding_type"] != "NONE"
        and all(review[f"{side}_{role}_span_id"] for side in ("a", "b") for role in ("subject", "predicate")),
        "subject veto needs both own-unique subjects and both predicate witnesses",
        pair_id=pair_id,
    )
    # Validate the proof even when a later authority gate will discard the objection.
    require(
        packet["status"] == "COMPLETE" and packet["truncated"] is False,
        "subject veto requires complete, untruncated evidence",
        pair_id=pair_id,
    )
    if review["binding_type"] not in VETO_BINDINGS:
        return None, evidence, "SUBJECT_OBJECTION_OUTSIDE_SPECIALIST_AUTHORITY"
    return selected, evidence, "VERIFY_FIXED_SUBJECT_VETO"


@dataclass(frozen=True)
class SubjectCritic:
    name: ClassVar[str] = "subject"
    review_column: ClassVar[str] = "subject_review"
    input_columns: ClassVar[tuple[str, ...]] = (
        "pair_id",
        "text_a",
        "text_b",
        "semantic_diff",
        "truncated",
        "final_decision",
    )
    temporary_columns: ClassVar[tuple[str, ...]] = ("_subject_payload", "_subject_candidate")
    drop_columns: ClassVar[tuple[str, ...]] = ()
    prepared_columns: ClassVar[tuple[str, ...]] = (
        "subject_should_run",
        "subject_reason",
        "subject_base_decision",
        "_subject_payload",
    )
    applied_columns: ClassVar[tuple[str, ...]] = (
        "subject_action",
        "subject_reason",
        "subject_evidence",
        "subject_verifier_should_run",
        "_subject_candidate",
    )
    output_columns: ClassVar[tuple[str, ...]] = (
        "subject_should_run",
        "subject_review",
        "subject_action",
        "subject_reason",
        "subject_evidence",
        "subject_base_decision",
        "subject_verifier_should_run",
        "subject_verifier_review",
    )

    def prepare(self, record: dict[str, Any]) -> dict[str, Any]:
        pair_id, base = record.get("pair_id"), record.get("final_decision")
        require(isinstance(base, dict), "subject requires a coverage final_decision", pair_id=pair_id)
        validate_decision(base, pair_id=pair_id)
        packet = adapt_alignment(record)
        kinds = {span["kind"] for span in packet["spans"]}
        if "yes" not in (base["a_can_replace_b"], base["b_can_replace_a"]):
            reason = "PRESERVE_MAIN_NEGATIVE_OR_UNRESOLVED"
        elif complete_text_equality(record, packet):
            reason = "PRESERVE_COMPLETE_EXACT_INPUT"
        elif not {"A_ONLY", "B_ONLY"} <= kinds:
            reason = "PRESERVE_WITHOUT_BILATERAL_UNIQUE_SUBJECTS"
        elif packet["truncated"]:
            reason = "PRESERVE_INCOMPLETE_SUBJECT_CONTEXT"
        else:
            reason = "REVIEW_BILATERAL_SUBJECTS"
        return {
            "subject_should_run": reason == "REVIEW_BILATERAL_SUBJECTS",
            "subject_reason": reason,
            "subject_base_decision": dict(base),
            "_subject_payload": {**packet, "text_a": record["text_a"], "text_b": record["text_b"]},
        }

    def build_column(self, *, model_alias: str, prompt: str, system_prompt: str) -> dd.LLMStructuredColumnConfig:
        import data_designer.config as dd

        return dd.LLMStructuredColumnConfig(
            name=self.review_column,
            model_alias=model_alias,
            prompt=prompt,
            system_prompt=system_prompt,
            output_format=response_schema(),
            skip=dd.SkipConfig(when="{{ not subject_should_run }}"),
        )

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        pair_id, should_run = record.get("pair_id"), record["subject_should_run"]
        require(type(should_run) is bool, "subject_should_run must be boolean", pair_id=pair_id)
        review = record.get(self.review_column)
        candidate, evidence, reason = None, [], record["subject_reason"]
        if should_run:
            candidate, evidence, reason = _compile_proposal(review, record["_subject_payload"], pair_id)
        else:
            require(review is None, "skipped subject row has an unexpected review", pair_id=pair_id)
        return {
            "subject_action": "PENDING_VERIFICATION" if candidate else "KEEP_COVERAGE" if should_run else "SKIP",
            "subject_reason": reason,
            "subject_evidence": evidence,
            "subject_verifier_should_run": candidate is not None,
            "_subject_candidate": candidate,
        }


@dataclass(frozen=True)
class SubjectVerifier:
    """Verify the fixed proposal before changing the saved coverage decision."""

    name: ClassVar[str] = "subject_verifier"
    review_column: ClassVar[str] = "subject_verifier_review"
    prepared_columns: ClassVar[tuple[str, ...]] = (
        "subject_verifier_should_run",
        "subject_base_decision",
        "subject_action",
        "subject_reason",
        "_subject_candidate",
        "_subject_payload",
    )
    temporary_columns: ClassVar[tuple[str, ...]] = SubjectCritic.temporary_columns
    drop_columns: ClassVar[tuple[str, ...]] = temporary_columns
    applied_columns: ClassVar[tuple[str, ...]] = ("subject_action", "subject_reason", "final_decision")
    output_columns: ClassVar[tuple[str, ...]] = applied_columns

    def build_column(self, *, model_alias: str, prompt: str, system_prompt: str) -> dd.LLMStructuredColumnConfig:
        import data_designer.config as dd

        return dd.LLMStructuredColumnConfig(
            name=self.review_column,
            model_alias=model_alias,
            prompt=prompt,
            system_prompt=system_prompt,
            output_format=verifier_schema(),
            skip=dd.SkipConfig(when="{{ not subject_verifier_should_run }}"),
        )

    def apply(self, record: dict[str, Any]) -> dict[str, Any]:
        pair_id, should_run = record.get("pair_id"), record["subject_verifier_should_run"]
        require(type(should_run) is bool, "subject_verifier_should_run must be boolean", pair_id=pair_id)
        review, base = record.get(self.review_column), dict(record["subject_base_decision"])
        validate_decision(base, pair_id=pair_id)
        action, reason = record["subject_action"], record["subject_reason"]
        if not should_run:
            require(review is None, "skipped verifier has an unexpected review", pair_id=pair_id)
        else:
            require(
                isinstance(record["_subject_candidate"], dict)
                and "yes" in (base["a_can_replace_b"], base["b_can_replace_a"]),
                "verifier requires a validated positive-direction proposal",
                pair_id=pair_id,
            )
            require(
                isinstance(review, dict)
                and set(review) == set(verifier_schema()["required"])
                and all(isinstance(value, str) for value in review.values())
                and all(review[f"{side}_subject_kind"] in KINDS for side in ("a", "b"))
                and review["comparison"] in COMPARISONS
                and bool(review["explanation"].strip()),
                "invalid subject verifier schema",
                pair_id=pair_id,
            )
            if review["comparison"] == "SUPPORTED_DIFFERENT_NAMED_TARGETS":
                require(
                    all(review[f"{side}_subject_kind"] == "NAMED_ACTUAL_TARGET" for side in ("a", "b")),
                    "supported subject comparison requires two named actual targets",
                    pair_id=pair_id,
                )
                base.update(
                    a_can_replace_b="no",
                    b_can_replace_a="no",
                    relation_type="related_non_duplicate",
                    material_difference="major",
                    primary_material_difference="document_identity_change",
                    primary_risk_factor="template_slot_collision",
                    confidence_tier="medium",
                )
                action, reason = "REJECT_BOTH", "VERIFIED_FIXED_SUBJECT_VETO"
            else:
                action, reason = "KEEP_COVERAGE", "UNSUPPORTED_FIXED_PROPOSAL_KEEP_COVERAGE"
        validate_decision(base, pair_id=pair_id)
        return {"subject_action": action, "subject_reason": reason, "final_decision": base}
