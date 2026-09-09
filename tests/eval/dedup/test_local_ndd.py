# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

from eval.dedup.config import LocalNddJudgeConfig
from eval.dedup.judging.local_ndd import _safe_retry_feedback, adapt_ndd_judge_output, run_local_ndd_pending
from eval.dedup.judging.schema import JUDGE_SCHEMA_V2, JUDGE_SCHEMA_V3
from eval.dedup.validation import DedupEvaluationError


def _rubric(*, relation_type: str = "exact") -> dict[str, dict[str, Any]]:
    values: dict[str, Any] = {
        "same_duplicate_group": "yes",
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": relation_type,
        "material_difference": "none",
        "fuzzy_scope": "in_scope",
        "confidence": "0.98",
        "reason_number_change": "no",
        "reason_date_time_change": "no",
        "reason_product_version_change": "no",
        "reason_url_change": "no",
        "reason_named_entity_change": "no",
        "reason_negation_change": "no",
        "reason_code_literal_change": "no",
        "reason_code_output_change": "no",
        "reason_insertion_deletion": "no",
        "reason_boilerplate": "yes",
        "reason_parser_noise": "no",
        "reason_language_mismatch": "no",
        "reason_topic_only": "no",
        "reason_insufficient_evidence": "no",
        "reason_other_material": "no",
    }
    return {name: {"score": score, "reasoning": f"private-{name}"} for name, score in values.items()}


def _local_config(tmp_path: Path) -> SimpleNamespace:
    judge = LocalNddJudgeConfig(
        backend="local_ndd",
        model="Qwen/Qwen3.8-27B",
        model_path=tmp_path / "model",
        runner_config=tmp_path / "judge.yaml",
        ray_temp_dir=tmp_path / "ray",
        checkpoint_root=tmp_path / "checkpoints",
        num_cpus=None,
        num_gpus=1,
        max_retries=2,
        max_visible_tokens=128,
        window_tokens=32,
        window_overlap_tokens=4,
        prompt_version="dedup-judge-sarah-minhash-v1",
        schema_version="dedup-judge-output-v0",
        visible_payload_version="judge-visible-payload-v2",
    )
    return SimpleNamespace(judge=judge)


def _rubric_v2() -> dict[str, dict[str, Any]]:
    values: dict[str, tuple[Any, str]] = {
        "a_can_replace_b": ("no", 'B: "beta two" is not preserved by A.'),
        "b_can_replace_a": ("no", 'A: "alpha one" is not preserved by B.'),
        "relation_type": ("unrelated", "Distinct main content."),
        "material_difference": ("major", "The main content differs."),
        "primary_material_difference": ("document_identity_change", "Different records."),
        "expected_minhash_action": ("keep_separate", "Low document-wide literal overlap."),
        "dominant_overlap_source": ("none", "No meaningful shared surface content."),
        "primary_risk_factor": ("none", "No fuzzy-dedup trap."),
        "evidence_quality": ("sufficient", "Visible text is complete."),
        "quote_evidence": ("provided", 'A: "alpha one" B: "beta two"'),
        "confidence": ("0.9", "Direct evidence."),
    }
    return {name: {"score": score, "reasoning": reasoning} for name, (score, reasoning) in values.items()}


def _rubric_v3() -> dict[str, dict[str, Any]]:
    values: dict[str, tuple[Any, str]] = {
        "a_can_replace_b": ("no", 'B: "beta two" is a different record.'),
        "b_can_replace_a": ("no", 'A: "alpha one" is a different record.'),
        "relation_type": ("related_non_duplicate", "The record identities differ."),
        "material_difference": ("major", "The main identities differ."),
        "primary_material_difference": ("document_identity_change", "Different titles."),
        "dominant_overlap_source": ("shared_page_template", "Only the template is shared."),
        "primary_risk_factor": ("template_slot_collision", "Template slots mask identity."),
        "confidence_tier": ("high", "Complete direct evidence."),
        "quote_evidence": ("provided", 'A: "alpha one" B: "beta two"'),
    }
    return {name: {"score": score, "reasoning": reasoning} for name, (score, reasoning) in values.items()}


def _with_semantic_ledger(
    rubric: dict[str, dict[str, Any]],
    *,
    profile_b: str = "substantive_main",
    basis: str = "substantive_anchor",
    conflict: str = "none",
    difference: str = "none",
) -> dict[str, dict[str, Any]]:
    rubric.update(
        {
            "content_profile_a": {"score": "substantive_main", "reasoning": "A content profile."},
            "content_profile_b": {"score": profile_b, "reasoning": "B content profile."},
            "shared_content_basis": {"score": basis, "reasoning": "Shared-content basis."},
            "hard_conflict": {"score": conflict, "reasoning": "Conflict scan."},
            "decisive_difference_location": {"score": difference, "reasoning": "Difference location."},
        }
    )
    return rubric


def _with_semantic_ledger_v2(  # noqa: PLR0913
    rubric: dict[str, dict[str, Any]],
    *,
    profile_a: str = "substantive_main",
    profile_b: str = "substantive_main",
    basis: str = "substantive_anchor",
    conflict: str = "none",
    difference: str = "none",
    alignment: str = "same_substantive_record",
    non_main: str = "not_applicable",
    translation: str = "not_translation",
) -> dict[str, dict[str, Any]]:
    _with_semantic_ledger(
        rubric,
        profile_b=profile_b,
        basis=basis,
        conflict=conflict,
        difference=difference,
    )
    rubric["content_profile_a"]["score"] = profile_a
    rubric.update(
        {
            "record_alignment": {"score": alignment, "reasoning": "Record-alignment gate."},
            "non_main_difference": {"score": non_main, "reasoning": "Non-main difference gate."},
            "translation_status": {"score": translation, "reasoning": "Translation gate."},
        }
    )
    return rubric


def _with_semantic_ledger_v3(  # noqa: PLR0913
    rubric: dict[str, dict[str, Any]],
    *,
    profile_a: str = "substantive_main",
    profile_b: str = "substantive_main",
    basis: str = "substantive_anchor",
    conflict: str = "none",
    difference: str = "none",
    alignment: str = "same_substantive_record",
    non_main: str = "not_applicable",
    translation: str = "not_translation",
    identity: str = "verbatim_shared_identifier",
    scope: str = "document_wide_same_record",
    surface: str = "not_applicable",
) -> dict[str, dict[str, Any]]:
    _with_semantic_ledger_v2(
        rubric,
        profile_a=profile_a,
        profile_b=profile_b,
        basis=basis,
        conflict=conflict,
        difference=difference,
        alignment=alignment,
        non_main=non_main,
        translation=translation,
    )
    rubric.update(
        {
            "record_identity_support": {
                "score": identity,
                "reasoning": 'A: "record 7" B: "record 7"',
            },
            "overlap_scope": {"score": scope, "reasoning": "Overlap-scope gate."},
            "surface_delta_type": {"score": surface, "reasoning": "Surface-delta gate."},
        }
    )
    return rubric


def _with_semantic_ledger_v4(
    rubric: dict[str, dict[str, Any]],
    *,
    boundary: str,
    **ledger_kwargs: str,
) -> dict[str, dict[str, Any]]:
    _with_semantic_ledger_v3(rubric, **ledger_kwargs)
    rubric["boundary_delta_class"] = {"score": boundary, "reasoning": "Decisive boundary class."}
    return rubric


def _span_payload(
    text_a: str,
    text_b: str,
    spans: list[dict[str, Any]],
    *,
    status: str = "COMPLETE",
) -> dict[str, Any]:
    counts = {kind: sum(span["kind"] == kind for span in spans) for kind in ("SHARED", "A_ONLY", "B_ONLY")}
    return {
        "payload_schema_version": "judge-visible-payload-v3",
        "document_a": {"text": text_a if status == "COMPLETE" else None},
        "document_b": {"text": text_b if status == "COMPLETE" else None},
        "long_document_evidence": {"truncated": status != "COMPLETE", "windows": []},
        "semantic_diff_evidence": {
            "contract_version": "visible-semantic-diff-v1",
            "status": status,
            "tokenization": "nfkc-casefold-visible-token-v1",
            "span_counts": counts,
            "spans": spans,
        },
    }


def _with_span_ledger(  # noqa: PLR0913
    rubric: dict[str, dict[str, Any]],
    *,
    profile_a: str = "substantive_main",
    profile_b: str = "substantive_main",
    basis: str = "verified_substantive_record",
    basis_reasoning: str = "S001 establishes the same record.",
    delta_a: str = "none",
    delta_a_reasoning: str = "No A-only spans.",
    delta_b: str = "none",
    delta_b_reasoning: str = "No B-only spans.",
    conflict: str = "none",
    conflict_reasoning: str = "No conflict.",
    translation: str = "not_translation",
    translation_reasoning: str = "Not a translation.",
) -> dict[str, dict[str, Any]]:
    rubric.pop("quote_evidence", None)
    rubric.update(
        {
            "span_content_profile_a": {"score": profile_a, "reasoning": "A profile."},
            "span_content_profile_b": {"score": profile_b, "reasoning": "B profile."},
            "span_shared_basis": {"score": basis, "reasoning": basis_reasoning},
            "span_a_delta": {"score": delta_a, "reasoning": delta_a_reasoning},
            "span_b_delta": {"score": delta_b, "reasoning": delta_b_reasoning},
            "span_hard_conflict": {"score": conflict, "reasoning": conflict_reasoning},
            "span_translation_status": {"score": translation, "reasoning": translation_reasoning},
        }
    )
    return rubric


def _record_binding_critic(verdict: str, reasoning: str) -> dict[str, dict[str, str]]:
    return {"record_binding_verdict": {"score": verdict, "reasoning": reasoning}}


@pytest.mark.parametrize(
    ("subtype", "expected"),
    [("equivalent_message_or_wrapper", "YES"), ("policy_proposition_change", "NO"), ("cookie_inventory_change", "NO")],
)
@pytest.mark.parametrize("policy", ["v4", "v5", "v5-policy"])
def test_boundary_adapter_distinguishes_cookie_wrapper_and_policy(subtype: str, expected: str, policy: str) -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 6,
            "a_text": "Cookie",
            "b_start_char": 0,
            "b_end_char": 6,
            "b_text": "Cookie",
        },
        {"span_id": "B001", "kind": "B_ONLY", "side": "B", "start_char": 7, "end_char": 14, "text": "options"},
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="verified_equivalent_non_main_message",
        delta_b="universal_ui_or_repetition",
        delta_b_reasoning="B001 is provisionally treated as wrapper.",
    )
    critic = {
        "non_main_delta_subtype": {"score": subtype, "reasoning": "S001 establishes context for B001."},
        "translation_delta_direction": {"score": "not_translation", "reasoning": "Same language."},
        "record_binding_verdict": {"score": "not_applicable", "reasoning": "Non-main boundary only."},
    }
    payload = _span_payload("Cookie", "Cookie options", spans)
    result = adapt_ndd_judge_output(
        rubric, JUDGE_SCHEMA_V3, payload=payload, record_binding_critic=critic, record_binding_policy=policy
    )
    assert result["a_can_replace_b"] == result["b_can_replace_a"] == expected
    assert f"BOUNDARY_NON_MAIN_DELTA_SUBTYPE:{subtype.upper()}" in result["reason_codes"]
    assert {item["side"] for item in result["evidence"]} == {"A", "B"}
    for item in result["evidence"]:
        text = payload[f"document_{item['side'].lower()}"]["text"]
        assert text[item["start_char"] : item["end_char"]] == item["quote"]


@pytest.mark.parametrize("direction", ["a_adds", "b_adds", "equivalent"])
@pytest.mark.parametrize("policy", ["v4", "v5"])
def test_boundary_adapter_aligns_translation_without_lexical_shared_identity(direction: str, policy: str) -> None:
    text_a, text_b = "The group helps.", "小组提供帮助和服务。"
    spans = [
        {"span_id": "A001", "kind": "A_ONLY", "side": "A", "start_char": 0, "end_char": len(text_a), "text": text_a},
        {"span_id": "B001", "kind": "B_ONLY", "side": "B", "start_char": 0, "end_char": len(text_b), "text": text_b},
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        basis_reasoning="A001 B001 describe the group.",
        delta_a="semantically_covered",
        delta_a_reasoning="A001 is provisionally covered.",
        delta_b="semantically_covered",
        delta_b_reasoning="B001 is provisionally covered.",
        translation="complete_faithful",
        translation_reasoning="A001 B001 provisionally translate.",
    )
    critic = {
        "non_main_delta_subtype": {"score": "not_applicable", "reasoning": "Substantive group description."},
        "translation_delta_direction": {"score": direction, "reasoning": "A001 B001 align the subject and facts."},
        "record_binding_verdict": {
            "score": "atomic_same_record_extension" if direction != "equivalent" else "semantic_equivalence",
            "reasoning": "A001 B001 establish the group and its services.",
        },
    }
    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload(text_a, text_b, spans),
        record_binding_critic={
            **critic,
            "record_binding_verdict": {
                **critic["record_binding_verdict"],
                "score": "benign_non_record_delta"
                if policy == "v5" and direction == "equivalent"
                else critic["record_binding_verdict"]["score"],
            },
        },
        record_binding_policy=policy,
    )
    assert result["a_can_replace_b"] == ("NO" if direction == "b_adds" else "YES")
    assert result["b_can_replace_a"] == ("NO" if direction == "a_adds" else "YES")
    assert result["material_difference"] == ("NONE" if direction == "equivalent" else "MAJOR")
    assert {item["side"] for item in result["evidence"]} == {"A", "B"}


def test_adapt_ndd_output_preserves_existing_contract_without_reasoning() -> None:
    result = adapt_ndd_judge_output(_rubric())

    assert result == {
        "same_duplicate_group": "YES",
        "a_can_replace_b": "YES",
        "b_can_replace_a": "YES",
        "relation_type": "EXACT",
        "material_difference": "NONE",
        "fuzzy_scope": "IN_SCOPE",
        "confidence": 0.98,
        "reason_codes": ["BOILERPLATE"],
        "evidence": [],
    }
    assert "private" not in json.dumps(result)


def test_adapt_ndd_output_accepts_hs_ab_diagnostics_without_changing_v0_output() -> None:
    rubric = _rubric()
    rubric.update(
        {
            "dominant_overlap_source": {"score": "main_content", "reasoning": "private-overlap"},
            "primary_risk_factor": {"score": "none", "reasoning": "private-risk"},
            "evidence_quality": {"score": "sufficient", "reasoning": "private-evidence"},
        }
    )

    assert adapt_ndd_judge_output(rubric) == adapt_ndd_judge_output(_rubric())


def test_adapt_ndd_v2_derives_minhash_outcome_and_retains_aligned_quotes() -> None:
    payload = {
        "document_a": {"text": "alpha one"},
        "document_b": {"text": "beta two"},
        "long_document_evidence": {"truncated": False, "windows": []},
    }

    result = adapt_ndd_judge_output(_rubric_v2(), JUDGE_SCHEMA_V2, payload=payload)

    assert result["expected_minhash_action"] == "KEEP_SEPARATE"
    assert result["surface_evidence_sufficiency"] == "SUFFICIENT"
    assert result["expected_minhash_outcome"] == "EXPECTED_TRUE_NEGATIVE"
    assert result["evidence"][:2] == [
        {"side": "A", "start_char": 0, "end_char": 9, "quote": "alpha one"},
        {"side": "B", "start_char": 0, "end_char": 8, "quote": "beta two"},
    ]
    assert result["reason_codes"] == [
        "MATERIAL_DELTA:DOCUMENT_IDENTITY_CHANGE",
        "EVIDENCE_STATUS:SUFFICIENT",
    ]
    assert "private" not in json.dumps(result)


def test_adapt_ndd_v2_rejects_non_exact_result_without_two_sided_quote_evidence() -> None:
    rubric = _rubric_v2()
    rubric["quote_evidence"]["reasoning"] = 'A: "alpha one" only.'
    rubric["a_can_replace_b"]["reasoning"] = "B lacks visible quote syntax."

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(
            rubric,
            JUDGE_SCHEMA_V2,
            payload={
                "document_a": {"text": "alpha one"},
                "document_b": {"text": ""},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        )

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_adapt_ndd_v2_unresolved_result_does_not_add_fallback_evidence() -> None:
    rubric = _rubric_v2()
    for field in ("a_can_replace_b", "b_can_replace_a", "relation_type", "material_difference"):
        rubric[field]["score"] = "unresolved"
    for field in ("primary_material_difference", "expected_minhash_action", "dominant_overlap_source"):
        rubric[field]["score"] = "unresolved"
    rubric["primary_risk_factor"]["score"] = "extraction_or_payload_limit"
    rubric["evidence_quality"]["score"] = "insufficient"
    rubric["quote_evidence"]["score"] = "insufficient"
    rubric["confidence"]["score"] = "0.5"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V2,
        payload={
            "document_a": {"text": "visible alpha"},
            "document_b": {"text": "visible beta"},
            "long_document_evidence": {"truncated": True, "windows": []},
        },
    )

    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["evidence"] == []


def test_adapt_ndd_v2_supplies_exact_visible_excerpt_when_model_quote_does_not_align() -> None:
    rubric = _rubric_v2()
    rubric["quote_evidence"]["reasoning"] = 'A: "alpha..." B: "beta..."'

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V2,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"] == [
        {"side": "A", "start_char": 0, "end_char": 9, "quote": "alpha one"},
        {"side": "B", "start_char": 0, "end_char": 8, "quote": "beta two"},
    ]


def test_adapt_ndd_v2_derives_group_from_directional_containment() -> None:
    rubric = _rubric_v2()
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["relation_type"]["score"] = "containment"
    rubric["material_difference"]["score"] = "major"
    rubric["expected_minhash_action"]["score"] = "group"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V2,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["same_duplicate_group"] == "YES"
    assert result["expected_minhash_outcome"] == "EXPECTED_TRUE_POSITIVE"


def test_adapt_ndd_v2_aligns_unlabeled_single_quoted_evidence_to_both_sides() -> None:
    rubric = _rubric_v2()
    rubric["quote_evidence"]["reasoning"] = "Shared 'common text'; A adds 'alpha one'."

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V2,
        payload={
            "document_a": {"text": "common text alpha one"},
            "document_b": {"text": "common text beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    retained = {(item["side"], item["quote"]) for item in result["evidence"]}
    assert {("A", "common text"), ("B", "common text")} <= retained


def test_adapt_ndd_v3_derives_semantic_group_without_minhash_or_numeric_confidence() -> None:
    result = adapt_ndd_judge_output(
        _rubric_v3(),
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["same_duplicate_group"] == "NO"
    assert result["confidence_tier"] == "MEDIUM"
    assert "expected_minhash_action" not in result
    assert "evidence_quality" not in result
    assert "confidence" not in result
    assert {item["side"] for item in result["evidence"]} == {"A", "B"}


@pytest.mark.parametrize(
    ("directions", "source_relation", "source_material", "expected"),
    [
        (("no", "yes"), "unrelated", "minor", ("CONTAINMENT", "MAJOR", "MAIN_CONTENT_ADDITION_DELETION")),
        (("no", "no"), "containment", "major", ("RELATED_NON_DUPLICATE", "MAJOR", "DOCUMENT_IDENTITY_CHANGE")),
        (
            ("no", "no"),
            "related_non_duplicate",
            "minor",
            ("RELATED_NON_DUPLICATE", "MAJOR", "DOCUMENT_IDENTITY_CHANGE"),
        ),
        (("yes", "yes"), "unrelated", "major", ("NEAR_SURFACE", "MINOR", "OTHER_MATERIAL")),
    ],
)
def test_adapt_ndd_v3_reconciles_diagnostics_from_authoritative_directions(
    directions: tuple[str, str],
    source_relation: str,
    source_material: str,
    expected: tuple[str, str, str],
) -> None:
    rubric = _rubric_v3()
    rubric["a_can_replace_b"]["score"], rubric["b_can_replace_a"]["score"] = directions
    rubric["relation_type"]["score"] = source_relation
    rubric["material_difference"]["score"] = source_material

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (
        result["relation_type"],
        result["material_difference"],
        result["primary_material_difference"],
    ) == expected


def test_adapt_ndd_v3_rejects_unaligned_non_exact_evidence_instead_of_fabricating_fallback() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = 'A: "missing alpha" B: "missing beta"'
    rubric["a_can_replace_b"]["reasoning"] = "Different record without a quote."
    rubric["b_can_replace_a"]["reasoning"] = "Different record without a quote."

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(
            rubric,
            JUDGE_SCHEMA_V3,
            payload={
                "document_a": {"text": "alpha one"},
                "document_b": {"text": "beta two"},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        )

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_adapt_ndd_v3_aligns_substantial_explicit_excerpt_before_model_ellipsis() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = (
        'A: "The University of Western Australia acknowledges..." B: "THE UNIVERSITY OF WESTERN AUSTRALIA"'
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "The University of Western Australia acknowledges the Traditional Owners."},
            "document_b": {"text": "THE UNIVERSITY OF WESTERN AUSTRALIA"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"][:2] == [
        {
            "side": "A",
            "start_char": 0,
            "end_char": 48,
            "quote": "The University of Western Australia acknowledges",
        },
        {
            "side": "B",
            "start_char": 0,
            "end_char": 35,
            "quote": "THE UNIVERSITY OF WESTERN AUSTRALIA",
        },
    ]


def test_adapt_ndd_v3_aligns_short_non_latin_ellipsis_excerpt_and_trailing_punctuation() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = (
        'A: "ボア・セレステ..." B: "This website uses cookies to improve your experience."'
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "記事\uff1aボア・セレステ"},
            "document_b": {"text": "This website uses cookies to improve your experience while browsing."},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"][:2] == [
        {"side": "A", "start_char": 3, "end_char": 10, "quote": "ボア・セレステ"},
        {
            "side": "B",
            "start_char": 0,
            "end_char": 52,
            "quote": "This website uses cookies to improve your experience",
        },
    ]


def test_adapt_ndd_v3_caps_high_confidence_for_truncated_input() -> None:
    result = adapt_ndd_judge_output(
        _rubric_v3(),
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": True, "windows": []},
        },
    )

    assert result["confidence_tier"] == "MEDIUM"


def test_adapt_ndd_v3_snaps_quote_across_line_breaks_and_short_latin_annotations() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = 'A: "上海廣樹機電有限公司" B: "0 comments:"'

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "上海廣樹(shù)機(jī)電有限公司"},
            "document_b": {"text": "0\ncomments:"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"] == [
        {"side": "A", "start_char": 0, "end_char": 19, "quote": "上海廣樹(shù)機(jī)電有限公司"},
        {"side": "B", "start_char": 0, "end_char": 11, "quote": "0\ncomments:"},
    ]


def test_adapt_ndd_v3_aligns_exact_prefix_when_model_changes_only_quote_tail() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = 'A: "alpha record visible corrected" B: "beta record visible corrected"'

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha record visible raw"},
            "document_b": {"text": "beta record visible raw"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"] == [
        {"side": "A", "start_char": 0, "end_char": 20, "quote": "alpha record visible"},
        {"side": "B", "start_char": 0, "end_char": 19, "quote": "beta record visible"},
    ]


def test_adapt_ndd_v3_truncates_long_explicit_quotes_to_evidence_limit() -> None:
    document_a = "alpha " * 50 + "A tail"
    document_b = "beta " * 60 + "B tail"
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = f'A: "{document_a}" B: "{document_b}"'

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": document_a},
            "document_b": {"text": document_b},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert {item["side"] for item in result["evidence"]} == {"A", "B"}
    assert all(len(item["quote"]) <= 240 for item in result["evidence"])
    assert all(item["quote"] in (document_a if item["side"] == "A" else document_b) for item in result["evidence"])


def test_adapt_ndd_v3_aligns_guillemets_literal_newline_escapes_and_short_cjk() -> None:
    rubric = _rubric_v3()
    rubric["quote_evidence"]["reasoning"] = 'A: «巻次 九» B: "媒体\\n在线"'

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "巻次\n\n九"},
            "document_b": {"text": "媒体\n在线"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["evidence"] == [
        {"side": "A", "start_char": 0, "end_char": 5, "quote": "巻次\n\n九"},
        {"side": "B", "start_char": 0, "end_char": 5, "quote": "媒体\n在线"},
    ]


def test_adapt_ndd_v3_accepts_strict_nonempty_containment() -> None:
    rubric = _rubric_v3()
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["relation_type"]["score"] = "containment"
    rubric["primary_material_difference"]["score"] = "main_content_addition_deletion"
    rubric["dominant_overlap_source"]["score"] = "main_content"
    rubric["primary_risk_factor"]["score"] = "containment_asymmetry"
    rubric["confidence_tier"]["score"] = "medium"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert result["same_duplicate_group"] == "YES"
    assert result["a_can_replace_b"] == "YES"
    assert result["b_can_replace_a"] == "NO"
    assert result["relation_type"] == "CONTAINMENT"


def test_adapt_ndd_v3_semantic_ledger_vetoes_vacuous_containment() -> None:
    rubric = _with_semantic_ledger(
        _rubric_v3(),
        profile_b="non_main_only",
        basis="none",
        difference="a_only_main_addition",
    )
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["relation_type"]["score"] = "containment"
    rubric["primary_material_difference"]["score"] = "main_content_addition_deletion"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["relation_type"] == "UNRELATED"
    assert "SEMANTIC_LEDGER_RULE:SUBSTANTIVE_NON_MAIN_PROFILE_MISMATCH" in result["reason_codes"]


def test_adapt_ndd_v3_semantic_ledger_makes_chrome_only_delta_bidirectional() -> None:
    rubric = _with_semantic_ledger(_rubric_v3(), difference="non_main_only")
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["relation_type"]["score"] = "containment"
    rubric["primary_material_difference"]["score"] = "main_content_addition_deletion"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["relation_type"] == "NEAR_SURFACE"
    assert result["material_difference"] == "MINOR"


def test_adapt_ndd_v3_semantic_ledger_enforces_nonempty_containment_direction() -> None:
    rubric = _with_semantic_ledger(_rubric_v3(), difference="a_only_main_addition")

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert result["primary_material_difference"] == "MAIN_CONTENT_ADDITION_DELETION"


def test_adapt_ndd_v3_semantic_ledger_hard_conflict_has_priority() -> None:
    rubric = _with_semantic_ledger(_rubric_v3(), conflict="page_role", difference="none")
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["b_can_replace_a"]["score"] = "yes"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["primary_material_difference"] == "PAGE_ROLE_CHANGE"
    assert "HARD_CONFLICT:PAGE_ROLE" in result["reason_codes"]


def test_adapt_ndd_v3_semantic_ledger_requires_all_fields_together() -> None:
    rubric = _rubric_v3()
    rubric["content_profile_a"] = {"score": "substantive_main", "reasoning": "A profile."}

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(
            rubric,
            JUDGE_SCHEMA_V3,
            payload={
                "document_a": {"text": "alpha one"},
                "document_b": {"text": "beta two"},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        )

    assert error.value.issue.code == "LOCAL_NDD_OUTPUT_INVALID"


def test_adapt_ndd_v3_ledger_v2_rejects_generic_anchor_containment() -> None:
    rubric = _with_semantic_ledger_v2(
        _rubric_v3(),
        difference="a_only_main_addition",
        alignment="generic_or_template_overlap_only",
    )
    rubric["a_can_replace_b"]["score"] = "yes"
    rubric["relation_type"]["score"] = "containment"
    rubric["primary_material_difference"]["score"] = "main_content_addition_deletion"

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:NO_CONFIRMED_SAME_SUBSTANTIVE_RECORD" in result["reason_codes"]
    assert "RECORD_ALIGNMENT:GENERIC_OR_TEMPLATE_OVERLAP_ONLY" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v2_accepts_equivalent_non_main_chrome() -> None:
    rubric = _with_semantic_ledger_v2(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="equivalent_non_main_message",
        difference="non_main_only",
        alignment="same_non_main_message",
        non_main="none_or_ignorable_chrome",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["relation_type"] == "NEAR_SURFACE"
    assert "NON_MAIN_DIFFERENCE:NONE_OR_IGNORABLE_CHROME" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v2_rejects_material_non_main_delta() -> None:
    rubric = _with_semantic_ledger_v2(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="equivalent_non_main_message",
        difference="non_main_only",
        alignment="same_non_main_message",
        non_main="material_message_or_state",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:MATERIAL_NON_MAIN_DIFFERENCE" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v2_preserves_complete_translation() -> None:
    rubric = _with_semantic_ledger_v2(
        _rubric_v3(),
        difference="two_sided_main_divergence",
        translation="complete_faithful",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["material_difference"] == "NONE"
    assert "SEMANTIC_LEDGER_RULE:COMPLETE_FAITHFUL_TRANSLATION" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v2_translation_cannot_bypass_record_alignment() -> None:
    rubric = _with_semantic_ledger_v2(
        _rubric_v3(),
        basis="none",
        difference="two_sided_main_divergence",
        alignment="generic_or_template_overlap_only",
        translation="complete_faithful",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:NO_CONFIRMED_SAME_SUBSTANTIVE_RECORD" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v2_requires_every_added_gate() -> None:
    rubric = _with_semantic_ledger(_rubric_v3())
    rubric["record_alignment"] = {"score": "same_substantive_record", "reasoning": "Alignment."}

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(
            rubric,
            JUDGE_SCHEMA_V3,
            payload={
                "document_a": {"text": "alpha one"},
                "document_b": {"text": "beta two"},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        )

    assert error.value.issue.code == "LOCAL_NDD_OUTPUT_INVALID"


def test_adapt_ndd_v3_ledger_v3_rejects_local_passage_containment() -> None:
    rubric = _with_semantic_ledger_v3(
        _rubric_v3(),
        difference="a_only_main_addition",
        scope="local_passage_only",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "record 7 alpha one"},
            "document_b": {"text": "record 7 beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:NON_DOCUMENT_WIDE_SUBSTANTIVE_OVERLAP" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v3_accepts_benign_non_main_label() -> None:
    rubric = _with_semantic_ledger_v3(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="equivalent_non_main_message",
        difference="non_main_only",
        alignment="same_non_main_message",
        non_main="none_or_ignorable_chrome",
        identity="same_non_main_message",
        scope="complete_non_main_message",
        surface="nav_label_byline_only",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert "SURFACE_DELTA_TYPE:NAV_LABEL_BYLINE_ONLY" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v3_rejects_material_surface_proposition() -> None:
    rubric = _with_semantic_ledger_v3(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="equivalent_non_main_message",
        difference="non_main_only",
        alignment="same_non_main_message",
        non_main="none_or_ignorable_chrome",
        identity="same_non_main_message",
        scope="complete_non_main_message",
        surface="material_proposition_or_record",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:MATERIAL_SURFACE_PROPOSITION_OR_RECORD" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v3_requires_field_specific_identity_quotes() -> None:
    rubric = _with_semantic_ledger_v3(_rubric_v3())
    rubric["record_identity_support"]["reasoning"] = "The record identity is obvious."

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(
            rubric,
            JUDGE_SCHEMA_V3,
            payload={
                "document_a": {"text": "record 7 alpha one"},
                "document_b": {"text": "record 7 beta two"},
                "long_document_evidence": {"truncated": False, "windows": []},
            },
        )

    assert error.value.issue.code == "LOCAL_NDD_OUTPUT_INVALID"


def test_adapt_ndd_v3_ledger_v4_accepts_verified_same_record_extension() -> None:
    rubric = _with_semantic_ledger_v4(
        _rubric_v3(),
        boundary="same_record_content_extension",
        difference="a_only_main_addition",
        surface="material_proposition_or_record",
    )
    rubric["record_identity_support"]["reasoning"] = "Same record identifier is visible on both sides."

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "record 7 alpha one"},
            "document_b": {"text": "record 7 beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert "SEMANTIC_LEDGER_RULE:VERIFIED_SAME_RECORD_CONTENT_EXTENSION" in result["reason_codes"]


@pytest.mark.parametrize(
    ("boundary", "rule"),
    [
        ("record_identity_role_or_state_change", "BOUNDARY_RECORD_IDENTITY_ROLE_OR_STATE_CHANGE"),
        ("material_non_main_message_change", "BOUNDARY_MATERIAL_NON_MAIN_MESSAGE_CHANGE"),
        ("two_sided_content_change", "BOUNDARY_TWO_SIDED_CONTENT_CHANGE"),
    ],
)
def test_adapt_ndd_v3_ledger_v4_rejects_material_boundary_classes(boundary: str, rule: str) -> None:
    rubric = _with_semantic_ledger_v4(_rubric_v3(), boundary=boundary)

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "record 7 alpha one"},
            "document_b": {"text": "record 7 beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert f"SEMANTIC_LEDGER_RULE:{rule}" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v4_accepts_verified_universal_ui_delta() -> None:
    rubric = _with_semantic_ledger_v4(
        _rubric_v3(),
        boundary="universal_ui_or_redundant_repetition",
        difference="non_main_only",
        surface="nav_label_byline_only",
    )
    rubric["record_identity_support"]["reasoning"] = "Same record identifier is visible on both sides."

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "record 7 alpha one"},
            "document_b": {"text": "record 7 beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert "BOUNDARY_DELTA_CLASS:UNIVERSAL_UI_OR_REDUNDANT_REPETITION" in result["reason_codes"]


def test_adapt_ndd_v3_ledger_v4_rejects_extension_without_verified_identity() -> None:
    rubric = _with_semantic_ledger_v4(
        _rubric_v3(),
        boundary="same_record_content_extension",
        difference="b_only_main_addition",
        identity="not_proven",
        scope="generic_family_or_template_only",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload={
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        },
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert "SEMANTIC_LEDGER_RULE:UNVERIFIED_SAME_RECORD_CONTENT_EXTENSION" in result["reason_codes"]


def test_adapt_ndd_v3_span_ledger_derives_containment_and_exact_evidence() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "record 7",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "record 7",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 25,
            "text": "adds full detail",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="A001 adds a substantive same-record section.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record 7 adds full detail", "record 7", spans),
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert result["evidence"] == [
        {"side": "A", "start_char": 0, "end_char": 8, "quote": "record 7"},
        {"side": "B", "start_char": 0, "end_char": 8, "quote": "record 7"},
        {"side": "A", "start_char": 9, "end_char": 25, "quote": "adds full detail"},
    ]
    assert "SPAN_LEDGER_RULE:SPAN_VERIFIED_SAME_RECORD_CONTENT_EXTENSION" in result["reason_codes"]


def test_adapt_ndd_v3_record_binding_critic_confirms_atomic_extension() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "record 7",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "record 7",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 25,
            "text": "adds full detail",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="A001 adds a substantive same-record section.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record 7 adds full detail", "record 7", spans),
        record_binding_critic=_record_binding_critic(
            "atomic_same_record_extension", "S001 identifies the job and A001 adds its application section."
        ),
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert "SPAN_RECORD_BINDING_CRITIC:ATOMIC_SAME_RECORD_EXTENSION" in result["reason_codes"]
    assert "CRITIC_CONFIRMED_ATOMIC_EXTENSION" in next(
        code for code in result["reason_codes"] if code.startswith("SPAN_LEDGER_RULE:")
    )


def test_adapt_ndd_v3_record_binding_critic_vetoes_template_attachment() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "site bio",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "site bio",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 24,
            "text": "different article",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="A001 appears to extend S001.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("site bio different article", "site bio", spans),
        record_binding_critic=_record_binding_critic(
            "separate_record_or_template_attachment", "S001 is reusable context; A001 is another article."
        ),
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["relation_type"] == "RELATED_NON_DUPLICATE"
    assert result["primary_material_difference"] == "PAGE_ROLE_CHANGE"
    assert "SPAN_RECORD_BINDING_CRITIC:SEPARATE_RECORD_OR_TEMPLATE_ATTACHMENT" in result["reason_codes"]


def test_adapt_ndd_v3_record_binding_critic_fails_closed_on_unknown_citation() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "record 7",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "record 7",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 25,
            "text": "adds full detail",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="A001 adds a substantive same-record section.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record 7 adds full detail", "record 7", spans),
        record_binding_critic=_record_binding_critic(
            "atomic_same_record_extension", "S001 identifies the record and A999 adds detail."
        ),
    )

    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"
    assert "SPAN_CRITIC_ISSUE:CRITIC_UNKNOWN_SPAN_CITATION" in result["reason_codes"]


def test_adapt_ndd_v3_calibrated_critic_vetoes_substantive_positive() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 6,
            "a_text": "record",
            "b_start_char": 0,
            "b_end_char": 6,
            "b_text": "record",
        },
        {"span_id": "A001", "kind": "A_ONLY", "side": "A", "start_char": 7, "end_char": 8, "text": "a"},
        {"span_id": "B001", "kind": "B_ONLY", "side": "B", "start_char": 7, "end_char": 8, "text": "b"},
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="semantically_covered",
        delta_a_reasoning="A001 is reordered content.",
        delta_b="semantically_covered",
        delta_b_reasoning="B001 is reordered content.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record a", "record b", spans),
        record_binding_critic=_record_binding_critic(
            "two_sided_or_conflicting", "S001 is shared but A001 and B001 identify different records."
        ),
        record_binding_policy="v2",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["relation_type"] == "RELATED_NON_DUPLICATE"
    assert "CALIBRATED_CRITIC_VETO_TWO_SIDED_OR_CONFLICTING" in next(
        code for code in result["reason_codes"] if code.startswith("SPAN_LEDGER_RULE:")
    )


def test_adapt_ndd_v3_calibrated_critic_does_not_reopen_non_main_equivalence() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 13,
            "a_text": "cookie notice",
            "b_start_char": 0,
            "b_end_char": 13,
            "b_text": "cookie notice",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 14,
            "end_char": 20,
            "text": "accept",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="verified_equivalent_non_main_message",
        delta_a="universal_ui_or_repetition",
        delta_a_reasoning="A001 is harmless cookie UI.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("cookie notice accept", "cookie notice", spans),
        record_binding_critic=_record_binding_critic(
            "separate_record_or_template_attachment", "S001 is shared and A001 is wrapper UI."
        ),
        record_binding_policy="v2",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["relation_type"] == "NEAR_SURFACE"


def test_adapt_ndd_v3_calibrated_critic_vetoes_non_main_policy_change() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 13,
            "a_text": "cookie notice",
            "b_start_char": 0,
            "b_end_char": 13,
            "b_text": "cookie notice",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 14,
            "end_char": 34,
            "text": "analytics permission",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="verified_equivalent_non_main_message",
        delta_a="universal_ui_or_repetition",
        delta_a_reasoning="A001 was provisionally treated as UI.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("cookie notice analytics permission", "cookie notice", spans),
        record_binding_critic=_record_binding_critic(
            "non_main_policy_or_state_change", "S001 is shared but A001 adds analytics permission."
        ),
        record_binding_policy="v2",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["relation_type"] == "RELATED_NON_DUPLICATE"


def test_adapt_ndd_v3_asymmetric_critic_defers_all_non_main_decisions_to_main_ledger() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 13,
            "a_text": "cookie notice",
            "b_start_char": 0,
            "b_end_char": 13,
            "b_text": "cookie notice",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 14,
            "end_char": 20,
            "text": "accept",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="verified_equivalent_non_main_message",
        delta_a="universal_ui_or_repetition",
        delta_a_reasoning="A001 is harmless cookie UI.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("cookie notice accept", "cookie notice", spans),
        record_binding_critic=_record_binding_critic(
            "non_main_policy_or_state_change", "S001 is the message and A001 is the accept control."
        ),
        record_binding_policy="v3",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["relation_type"] == "NEAR_SURFACE"
    assert "ASYMMETRIC_CRITIC_DEFERRED_TO_NON_MAIN_LEDGER" in next(
        code for code in result["reason_codes"] if code.startswith("SPAN_LEDGER_RULE:")
    )


def test_adapt_ndd_v3_asymmetric_critic_does_not_flatten_verified_extension() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "record 7",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "record 7",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 25,
            "text": "adds full detail",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="S001 identifies the record and A001 adds its application section.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record 7 adds full detail", "record 7", spans),
        record_binding_critic=_record_binding_critic(
            "benign_non_record_delta", "S001 is shared and A001 was classified as benign."
        ),
        record_binding_policy="v3",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert "ASYMMETRIC_CRITIC_BENIGN_DID_NOT_OVERRIDE_EXTENSION" in next(
        code for code in result["reason_codes"] if code.startswith("SPAN_LEDGER_RULE:")
    )


def test_adapt_ndd_v3_asymmetric_critic_preserves_additive_translation_direction() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 8,
            "a_text": "service",
            "b_start_char": 0,
            "b_end_char": 8,
            "b_text": "Dienst",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 9,
            "end_char": 22,
            "text": "contact email",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="same_record_content_extension",
        delta_a_reasoning="S001 binds the translated record; A001 adds contact information.",
        translation="partial_or_additive",
        translation_reasoning="S001 is translated and A001 is additive.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("service contact email", "Dienst", spans),
        record_binding_critic=_record_binding_critic(
            "two_sided_or_conflicting", "S001 is translated while A001 contains unmatched text."
        ),
        record_binding_policy="v3",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "NO")
    assert result["relation_type"] == "CONTAINMENT"
    assert result["confidence_tier"] == "LOW"
    assert "ASYMMETRIC_CRITIC_DEFERRED_TO_ADDITIVE_TRANSLATION_LEDGER" in next(
        code for code in result["reason_codes"] if code.startswith("SPAN_LEDGER_RULE:")
    )


def test_adapt_ndd_v3_asymmetric_critic_keeps_substantive_attachment_veto() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 6,
            "a_text": "record",
            "b_start_char": 0,
            "b_end_char": 6,
            "b_text": "record",
        },
        {"span_id": "A001", "kind": "A_ONLY", "side": "A", "start_char": 7, "end_char": 8, "text": "a"},
        {"span_id": "B001", "kind": "B_ONLY", "side": "B", "start_char": 7, "end_char": 8, "text": "b"},
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_a="semantically_covered",
        delta_a_reasoning="A001 was provisionally covered.",
        delta_b="semantically_covered",
        delta_b_reasoning="B001 was provisionally covered.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("record a", "record b", spans),
        record_binding_critic=_record_binding_critic(
            "separate_record_or_template_attachment", "S001 is reusable; A001 and B001 are separate records."
        ),
        record_binding_policy="v3",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["primary_material_difference"] == "PAGE_ROLE_CHANGE"


def test_adapt_ndd_v3_asymmetric_critic_keeps_substantive_policy_veto() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 14,
            "a_text": "privacy policy",
            "b_start_char": 0,
            "b_end_char": 14,
            "b_text": "privacy policy",
        },
        {
            "span_id": "B001",
            "kind": "B_ONLY",
            "side": "B",
            "start_char": 15,
            "end_char": 28,
            "text": "legal warning",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        delta_b="same_record_content_extension",
        delta_b_reasoning="S001 identifies the policy and B001 adds a warning.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("privacy policy", "privacy policy legal warning", spans),
        record_binding_critic=_record_binding_critic(
            "non_main_policy_or_state_change", "S001 is shared and B001 changes legal meaning."
        ),
        record_binding_policy="v3",
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["primary_material_difference"] == "LEGAL_CONTEXT_CHANGE"


def test_adapt_ndd_v3_span_ledger_forbids_non_main_containment() -> None:
    spans = [
        {
            "span_id": "S001",
            "kind": "SHARED",
            "a_start_char": 0,
            "a_end_char": 13,
            "a_text": "cookie notice",
            "b_start_char": 0,
            "b_end_char": 13,
            "b_text": "cookie notice",
        },
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 14,
            "end_char": 34,
            "text": "analytics permission",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        profile_a="non_main_only",
        profile_b="non_main_only",
        basis="verified_equivalent_non_main_message",
        delta_a="material_non_main_message_change",
        delta_a_reasoning="A001 adds analytics permission.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("cookie notice analytics permission", "cookie notice", spans),
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("NO", "NO")
    assert result["relation_type"] == "RELATED_NON_DUPLICATE"
    assert "SPAN_LEDGER_RULE:SPAN_MATERIAL_NON_MAIN_DELTA" in result["reason_codes"]


def test_adapt_ndd_v3_span_ledger_preserves_complete_translation() -> None:
    spans = [
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 0,
            "end_char": 11,
            "text": "hello world",
        },
        {
            "span_id": "B001",
            "kind": "B_ONLY",
            "side": "B",
            "start_char": 0,
            "end_char": 13,
            "text": "bonjour monde",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        basis_reasoning="A001 and B001 establish the same translated record.",
        delta_a="semantically_covered",
        delta_a_reasoning="A001 is fully translated by B001.",
        delta_b="semantically_covered",
        delta_b_reasoning="B001 is fully translated by A001.",
        translation="complete_faithful",
        translation_reasoning="A001 and B001 are a complete faithful translation.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("hello world", "bonjour monde", spans),
    )

    assert (result["a_can_replace_b"], result["b_can_replace_a"]) == ("YES", "YES")
    assert result["material_difference"] == "NONE"
    assert "SPAN_LEDGER_RULE:SPAN_COMPLETE_FAITHFUL_TRANSLATION" in result["reason_codes"]


def test_adapt_ndd_v3_span_ledger_fails_closed_on_bad_citation_without_retry() -> None:
    spans = [
        {
            "span_id": "A001",
            "kind": "A_ONLY",
            "side": "A",
            "start_char": 0,
            "end_char": 5,
            "text": "alpha",
        },
        {
            "span_id": "B001",
            "kind": "B_ONLY",
            "side": "B",
            "start_char": 0,
            "end_char": 4,
            "text": "beta",
        },
    ]
    rubric = _with_span_ledger(
        _rubric_v3(),
        basis="none",
        basis_reasoning="A999 and B001 are unrelated.",
    )

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("alpha", "beta", spans),
    )

    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"
    assert "SPAN_CITATION_ISSUE:UNKNOWN_SPAN_CITATION" in result["reason_codes"]


def test_adapt_ndd_v3_span_ledger_fails_closed_on_truncated_packet() -> None:
    rubric = _with_span_ledger(_rubric_v3())

    result = adapt_ndd_judge_output(
        rubric,
        JUDGE_SCHEMA_V3,
        payload=_span_payload("alpha", "alpha", [], status="UNAVAILABLE_TRUNCATED"),
    )

    assert result["same_duplicate_group"] == "UNRESOLVED"
    assert result["relation_type"] == "UNRESOLVED"
    assert result["confidence_tier"] == "LOW"


def test_adapt_ndd_output_rejects_cross_field_inconsistency() -> None:
    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(_rubric(relation_type="unrelated"))

    assert error.value.issue.code == "JUDGE_CONSISTENCY_INVALID"


def test_v2_equivalent_relation_retry_feedback_restates_translation_rule() -> None:
    error = DedupEvaluationError(
        "JUDGE_CONSISTENCY_INVALID",
        "equivalent relations require bidirectional replacement without a major difference",
    )

    feedback = _safe_retry_feedback(error)

    assert "a_can_replace_b=yes" in feedback["details"]["required_consistency"]
    assert "faithful full translation" in feedback["details"]["required_consistency"]


def test_adapt_ndd_output_rejects_invalid_enum() -> None:
    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(_rubric(relation_type="invented"))

    assert error.value.issue.code == "JUDGE_SCHEMA_INVALID"


def test_adapt_ndd_output_rejects_non_discrete_confidence() -> None:
    rubric = _rubric()
    rubric["confidence"]["score"] = "0.91"

    with pytest.raises(DedupEvaluationError) as error:
        adapt_ndd_judge_output(rubric)

    assert error.value.issue.code == "LOCAL_NDD_OUTPUT_INVALID"


def test_local_ndd_retries_only_missing_rows_and_persists_terminal_results(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": f"cp1_{index}",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": f"hash-{index}",
        }
        for index in range(2)
    ]
    prepared = {
        row["canonical_pair_id"]: {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
        for row in pending
    }
    call_sizes: list[int] = []

    class FakeRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def run(self, **kwargs: object) -> None:
            input_rows = [json.loads(line) for line in Path(str(kwargs["input_path"])).read_text().splitlines()]
            call_sizes.append(len(input_rows))
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True)
            with (output_path / "part.jsonl").open("w", encoding="utf-8") as file:
                for row in input_rows[:1]:
                    file.write(json.dumps({**row, "qwen_minhash_fuzzy_dedup_judge": _rubric()}) + "\n")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "work",
        persist=persisted.append,
        runtime_factory=FakeRuntime,
    )

    assert call_sizes == [2, 1]
    assert [row["record_type"] for row in persisted] == ["result", "result"]
    assert [row["attempts"] for row in persisted] == [1, 2]
    assert all("private" not in json.dumps(row) for row in persisted)
    retry_input = json.loads((tmp_path / "work" / "attempt_02" / "input.jsonl").read_text())
    assert retry_input["repair_feedback"]["code"] == "LOCAL_NDD_MISSING_ROW"


def test_local_ndd_duplicate_rows_exhaust_retry_budget_and_write_terminal_error(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": "cp1_duplicate",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": "hash-duplicate",
        }
    ]
    prepared = {
        "cp1_duplicate": {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
    }
    call_sizes: list[int] = []

    class DuplicateRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def run(self, **kwargs: object) -> None:
            row = json.loads(Path(str(kwargs["input_path"])).read_text())
            call_sizes.append(1)
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True)
            output = json.dumps({**row, "qwen_minhash_fuzzy_dedup_judge": _rubric()}) + "\n"
            (output_path / "part.jsonl").write_text(output + output, encoding="utf-8")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "duplicate-work",
        persist=persisted.append,
        runtime_factory=DuplicateRuntime,
    )

    assert call_sizes == [1, 1, 1]
    assert len(persisted) == 1
    assert persisted[0]["record_type"] == "error"
    assert persisted[0]["attempts"] == 3
    assert [error["validation_issue"]["code"] for error in persisted[0]["errors"]] == [
        "LOCAL_NDD_DUPLICATE_ROW",
        "LOCAL_NDD_DUPLICATE_ROW",
        "LOCAL_NDD_DUPLICATE_ROW",
    ]
    assert "private-" not in json.dumps(persisted[0])


def test_local_ndd_v3_downgrades_exhausted_consistency_failure_to_unresolved(tmp_path: Path) -> None:
    base = _local_config(tmp_path)
    config = SimpleNamespace(
        judge=replace(
            base.judge,
            prompt_version="dedup-judge-hs-v0.6.2",
            schema_version=JUDGE_SCHEMA_V3,
        )
    )
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": "cp1_contract_conflict",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": "hash-contract-conflict",
        }
    ]
    prepared = {
        "cp1_contract_conflict": {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha one"},
            "document_b": {"text": "beta two"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
    }
    calls = []

    class InvalidEvidenceRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def run(self, **kwargs: object) -> None:
            row = json.loads(Path(str(kwargs["input_path"])).read_text())
            calls.append(row["repair_feedback"])
            rubric = _rubric_v3()
            rubric["a_can_replace_b"]["reasoning"] = "Different records without a literal quote."
            rubric["b_can_replace_a"]["reasoning"] = "Different records without a literal quote."
            rubric["quote_evidence"]["reasoning"] = 'A: "missing alpha" B: "missing beta"'
            output_path = Path(str(kwargs["output_path"]))
            output_path.mkdir(parents=True)
            (output_path / "part.jsonl").write_text(
                json.dumps({**row, "qwen_dedup_semantic_judge": rubric}) + "\n",
                encoding="utf-8",
            )

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v3",
        work_root=tmp_path / "contract-conflict-work",
        persist=persisted.append,
        runtime_factory=InvalidEvidenceRuntime,
    )

    assert len(calls) == 3
    assert len(persisted) == 1
    assert persisted[0]["record_type"] == "result"
    assert persisted[0]["same_duplicate_group"] == "UNRESOLVED"
    assert persisted[0]["confidence_tier"] == "LOW"
    assert persisted[0]["evidence"] == []
    assert persisted[0]["attempts"] == 3
    assert persisted[0]["retried"] is True
    assert persisted[0]["deterministic_repair_events"] == [
        {
            "action": "downgrade_exhausted_contract_conflict_to_unresolved",
            "source_issue": "resolved non-exact decisions require aligned quote evidence from both documents",
            "failed_attempts": 3,
        }
    ]
    assert calls[0] is None
    assert all(call["code"] == "JUDGE_CONSISTENCY_INVALID" for call in calls[1:])


def test_local_ndd_runtime_failures_exhaust_retry_budget_and_write_terminal_error(tmp_path: Path) -> None:
    config = _local_config(tmp_path)
    pending = [
        {
            "evaluation_run_id": "fixture-run",
            "sut_run_id": "fixture-sut",
            "canonical_pair_id": "cp1_runtime",
            "canonical_pair_id_version": "cp1",
            "judge_payload_hash": "hash-runtime",
        }
    ]
    prepared = {
        "cp1_runtime": {
            "payload_schema_version": "judge-visible-payload-v2",
            "document_a": {"text": "alpha"},
            "document_b": {"text": "alpha"},
            "long_document_evidence": {"truncated": False, "windows": []},
        }
    }
    calls: list[dict[str, Any] | None] = []
    lifecycle: list[str] = []

    class FailingRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            lifecycle.append("enter")
            return self

        def __exit__(self, *args: object) -> None:
            del args
            lifecycle.append("exit")

        def run(self, **kwargs: object) -> None:
            row = json.loads(Path(str(kwargs["input_path"])).read_text())
            calls.append(row["repair_feedback"])
            raise RuntimeError("sensitive provider detail must not enter artifacts or retries")

    persisted: list[dict[str, Any]] = []
    run_local_ndd_pending(
        config,
        pending=pending,
        prepared=prepared,
        contract_digest="contract",
        contract_version="judge-execution-contract-v2",
        work_root=tmp_path / "runtime-failure-work",
        persist=persisted.append,
        runtime_factory=FailingRuntime,
    )

    assert lifecycle == ["enter", "exit"]
    assert len(persisted) == 1
    assert persisted[0]["record_type"] == "error"
    assert persisted[0]["attempts"] == 3
    assert [error["validation_issue"]["code"] for error in persisted[0]["errors"]] == [
        "RuntimeError",
        "RuntimeError",
        "RuntimeError",
    ]
    assert calls[0] is None
    assert calls[1] == {
        "code": "RuntimeError",
        "message": "local NDD runtime failed before producing a valid batch",
    }
    assert calls[2] == calls[1]
    assert "sensitive provider detail" not in json.dumps(persisted)
