# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jinja2 import Template

from eval.dedup.analysis.critic_diagnostic import (
    RESOURCES,
    VARIANTS,
    bind_fixed_main,
    blind_rows,
    critic_config,
    primary_changes,
    run_cell,
    validate_critic_config,
    validate_freeze,
)
from eval.dedup.judging.local_ndd import RECORD_BINDING_CRITIC_COLUMN, adapt_ndd_judge_output
from eval.dedup.judging.payload import _semantic_diff_packet, validate_evidence_offsets
from eval.dedup.validation import DedupEvaluationError, sha256_file, sha256_json, write_json_atomic


def _case():
    text_a = "Continuing to browse means consent."
    text_b = text_a + " Accept"
    payload = {
        "payload_schema_version": "judge-visible-payload-v3",
        "document_a": {"text": text_a},
        "document_b": {"text": text_b},
        "long_document_evidence": {"truncated": False, "windows": []},
        "semantic_diff_evidence": _semantic_diff_packet(text_a, text_b, truncated=False),
    }
    scores = {
        "a_can_replace_b": "yes",
        "b_can_replace_a": "yes",
        "relation_type": "near_surface",
        "material_difference": "none",
        "primary_material_difference": "none",
        "dominant_overlap_source": "cookie_consent",
        "primary_risk_factor": "template_slot_collision",
        "confidence_tier": "medium",
        "span_content_profile_a": "non_main_only",
        "span_content_profile_b": "non_main_only",
        "span_shared_basis": "verified_equivalent_non_main_message",
        "span_a_delta": "none",
        "span_b_delta": "universal_ui_or_repetition",
        "span_hard_conflict": "none",
        "span_translation_status": "not_translation",
    }
    main = {
        name: {"score": score, "reasoning": "S001 B001 retain the same condition; only an extra control."}
        for name, score in scores.items()
    }
    critic = {
        "record_scope": {
            "score": "equivalent_complete_message",
            "reasoning": "S001 has the same condition; B001 is a control.",
        },
        "record_binding_verdict": {
            "score": "benign_non_record_delta",
            "reasoning": "S001 is preserved and B001 adds only a control.",
        },
        "retained_conflict": {
            "score": "none",
            "reasoning": "S001 is the same condition; B001 alone changes no permission.",
        },
    }
    inputs = blind_rows([{"canonical_pair_id": "p", "payload": payload}], repeat=1)
    output = {**inputs[0], RECORD_BINDING_CRITIC_COLUMN: critic}
    return inputs, output, main


@pytest.mark.parametrize("variant", VARIANTS)
def test_actual_ndd_builder_uses_only_critic_and_same_rubric_schema(variant):
    from eval.llm_judge.run_llm_judge import build_config_builder

    path = critic_config(variant)
    validate_critic_config(path)
    config = yaml.safe_load(path.read_text())
    judges = config["execution"]["stages"][0]["judges"]
    builder, _ = build_config_builder(path, endpoint="http://127.0.0.1:1/v1", models=config["models"], judges=judges)
    columns = builder.get_column_configs()
    assert len(columns) == 1
    assert columns[0].name == RECORD_BINDING_CRITIC_COLUMN
    assert {s.name for s in columns[0].scores} == {
        "record_scope",
        "record_binding_verdict",
        "retained_conflict",
    }


@pytest.mark.parametrize("variant", ["consent", "pointer"])
def test_single_patch_preserves_all_existing_boundaries_and_same_span_packet(variant):
    original = (RESOURCES / "hs_v06214_record_binding_system.jinja").read_text()
    system = (RESOURCES / f"hs_v06216_{variant}_system.jinja").read_text()
    assert system.startswith(original.replace("V0.6.2.14", f"V0.6.2.16 {variant} diagnostic"))
    inputs, _, _ = _case()
    data = {"payload": inputs[0]["payload"], "repair_feedback": None}
    before = Template((RESOURCES / "hs_v06214_record_binding_pair.jinja").read_text()).render(**data)
    after = Template((RESOURCES / f"hs_v06216_{variant}_pair.jinja").read_text()).render(**data)
    assert before[before.index("<semantic_diff") :] == after[after.index("<semantic_diff") :]
    opposite = "SHORT POINTER COUNTERPART CHECK" if variant == "consent" else "CONSENT COUNTERPART CHECK"
    assert opposite not in system


def test_blind_input_drops_labels_main_and_arm_and_keeps_deterministic_pair_order():
    inputs, _, main = _case()
    packets = [
        {
            "canonical_pair_id": str(i),
            "payload": inputs[0]["payload"],
            "review_id": "H9999",
            "human_same_duplicate_group": "NO",
            "main": main,
            "variant": "pointer",
        }
        for i in range(8)
    ]
    first = blind_rows(packets, repeat=1)
    assert first == blind_rows(list(reversed(packets)), repeat=1)
    assert [r["canonical_pair_id"] for r in first] != [r["canonical_pair_id"] for r in blind_rows(packets, repeat=2)]
    assert all(set(row) == {"canonical_pair_id", "judge_payload_hash", "payload", "repair_feedback"} for row in first)


def test_fixed_main_binding_uses_real_v8_adapter_without_mutating_either_response():
    inputs, output, main = _case()
    snapshot = deepcopy((inputs, output, main))
    result = bind_fixed_main(inputs, [output], {"p": main})[0]
    expected = adapt_ndd_judge_output(
        main,
        "dedup-judge-output-v3",
        payload=inputs[0]["payload"],
        record_binding_critic=output[RECORD_BINDING_CRITIC_COLUMN],
        record_binding_policy="v8",
    )
    assert all(result[k] == v for k, v in expected.items())
    assert result["same_duplicate_group"] == "YES"
    assert result["diagnostic_only"] is True
    assert result["fixed_main_sha256"] == sha256_json(main)
    validate_evidence_offsets(result, inputs[0]["payload"])
    assert (inputs, output, main) == snapshot


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "hash", "main", "critic"])
def test_wrong_or_incomplete_critic_output_cannot_be_silently_scored(failure):
    inputs, output, main = _case()
    outputs = [output]
    if failure == "missing":
        outputs = []
    elif failure == "duplicate":
        outputs *= 2
    elif failure == "unknown":
        output["canonical_pair_id"] = "another"
    elif failure == "hash":
        output["judge_payload_hash"] = "changed"
    elif failure == "main":
        output["qwen_dedup_semantic_judge"] = main
    else:
        output.pop(RECORD_BINDING_CRITIC_COLUMN)
    with pytest.raises(DedupEvaluationError):
        bind_fixed_main(inputs, outputs, {"p": main})


def test_invalid_evidence_remains_unresolved_not_a_repaired_negative():
    inputs, output, main = _case()
    critic = output[RECORD_BINDING_CRITIC_COLUMN]
    critic["retained_conflict"] = {"score": "policy_permission_change", "reasoning": "A999 B999 change activation."}
    critic["record_binding_verdict"] = {"score": "non_main_policy_or_state_change", "reasoning": "A999 B999 conflict."}
    result = bind_fixed_main(inputs, [output], {"p": main})[0]
    assert result["same_duplicate_group"] == "UNRESOLVED"
    label = {
        "canonical_pair_id": "p",
        "review_id": "negative",
        "human_reason_code": "boilerplate",
        "human_same_duplicate_group": "NO",
        "human_a_can_replace_b": "NO",
        "human_b_can_replace_a": "NO",
    }
    before = {**result, "same_duplicate_group": "YES", "a_can_replace_b": "YES", "b_can_replace_a": "YES"}
    change = primary_changes([label], [before], [result])[0]
    assert change["source"]["error"] == "OVER_GROUP"
    assert change["candidate"]["error"] == "UNRESOLVED"


def test_config_rejects_accidental_second_main_judge(tmp_path: Path):
    config = yaml.safe_load(critic_config("control").read_text())
    judges = config["execution"]["stages"][0]["judges"]
    judges.append({**judges[0], "name": "qwen_dedup_semantic_judge"})
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(DedupEvaluationError, match="CRITIC_ONLY"):
        validate_critic_config(path)


def test_cell_retries_only_missing_pair_and_never_reuses_an_observed_cell(tmp_path: Path):
    inputs, output, main = _case()
    inputs.append({**inputs[0], "canonical_pair_id": "q"})
    batches, contexts = [], []

    class Runtime:
        def run(self, **kwargs):
            rows = [json.loads(line) for line in Path(kwargs["input_path"]).read_text().splitlines()]
            batches.append([row["canonical_pair_id"] for row in rows])
            folder = Path(kwargs["output_path"])
            folder.mkdir()
            rows = rows[:1] if len(batches) == 1 else rows
            (folder / "part.jsonl").write_text(
                "".join(
                    json.dumps({**row, RECORD_BINDING_CRITIC_COLUMN: output[RECORD_BINDING_CRITIC_COLUMN]}) + "\n"
                    for row in rows
                )
            )

    class Relay:
        def set_context(self, context):
            contexts.append(context)

    settings = {
        "max_retries": 2,
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 4096,
        "timeout_seconds": 600,
        "max_parallel_requests": 64,
    }
    root = tmp_path / "cell"
    complete = run_cell(root, inputs, {"p": main, "q": main}, Runtime(), Relay(), settings)
    assert batches == [["p", "q"], ["q"]]
    assert [c.outer_attempt for c in contexts] == [1, 2]
    assert (complete["requested"], complete["valid"], complete["retried"], complete["errors"]) == (2, 2, 1, 0)
    results = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    assert [row["attempts"] for row in results] == [1, 2]
    assert all(row["diagnostic_only"] for row in results)
    with pytest.raises(DedupEvaluationError, match="CRITIC_CELL_EXISTS"):
        run_cell(root, inputs, {"p": main, "q": main}, Runtime(), Relay(), settings)


@pytest.mark.parametrize("changed", ["reference", "main"])
def test_freeze_rejects_changed_reference_or_changed_main_even_if_input_digest_is_rewritten(tmp_path: Path, changed):
    from eval.dedup.rejudge_comparison import _source_digest

    reference = tmp_path / "reference.csv"
    reference.write_text("immutable reference\n")
    mains = {"pair": {"score": "yes"}}
    write_json_atomic(tmp_path / "fixed_main.json", mains)
    manifest = {
        "source_implementation_sha256": _source_digest(),
        "frozen_files": {str(reference): sha256_file(reference)},
        "fixed_main_sha256": sha256_json(mains),
    }
    manifest["diagnostic_contract_digest"] = sha256_json(manifest)
    write_json_atomic(tmp_path / "manifest.json", manifest)
    write_json_atomic(
        tmp_path / "input_freeze.json", {str(tmp_path / "fixed_main.json"): sha256_file(tmp_path / "fixed_main.json")}
    )
    validate_freeze(tmp_path)
    if changed == "reference":
        reference.write_text("modified reference\n")
    else:
        (tmp_path / "fixed_main.json").write_text(json.dumps({"pair": {"score": "no"}}))
        (tmp_path / "input_freeze.json").write_text(
            json.dumps({str(tmp_path / "fixed_main.json"): sha256_file(tmp_path / "fixed_main.json")})
        )
    with pytest.raises(DedupEvaluationError, match=r"CRITIC_FREEZE_CHANGED|CRITIC_MAIN_CHANGED"):
        validate_freeze(tmp_path)
