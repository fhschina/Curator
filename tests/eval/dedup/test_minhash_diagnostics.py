# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import pytest

from eval.dedup.analysis.minhash_diagnostics import (
    AVAILABLE,
    UNAVAILABLE_MISSING_CONTRACT,
    diagnose_pairwise_minhash,
    replay_pairwise_collision,
    resolve_sut_minhash_contract,
)
from eval.dedup.validation import DedupEvaluationError


def _contract() -> dict:
    return {
        "source": "sut_resolved_config",
        "normalization": {"name": "lowercase_nfc"},
        "shingling": {"kind": "character", "width": 5},
        "hash": {"family": "murmur3", "num_hashes": 6, "seed": 17},
        "lsh": {"bands": 2, "rows_per_band": 3},
    }


def test_missing_sut_config_is_explicitly_unavailable() -> None:
    result = diagnose_pairwise_minhash(contract_value=None)

    assert result["status"] == UNAVAILABLE_MISSING_CONTRACT
    assert "normalization.name" in result["missing_contract_fields"]


def test_evaluation_retriever_config_cannot_impersonate_resolved_sut_contract() -> None:
    value = _contract()
    value["source"] = "evaluation_retriever_config"

    result = diagnose_pairwise_minhash(contract_value=value)

    assert result["status"] == UNAVAILABLE_MISSING_CONTRACT
    assert "source=sut_resolved_config" in result["missing_contract_fields"]


def test_pairwise_collision_replays_exact_sut_bands() -> None:
    result = diagnose_pairwise_minhash(
        contract_value=_contract(),
        signature_a=[1, 2, 3, 4, 5, 6],
        signature_b=[1, 2, 3, 9, 9, 9],
        semantic_same_duplicate_group="NO",
    )

    assert result["status"] == AVAILABLE
    assert result["matched_band_indices"] == [0]
    assert result["pairwise_collision"] is True
    assert result["minhash_action"] == "GROUP"
    assert result["semantic_alignment"] == "OVER_GROUP_RISK"


def test_exact_replay_rejects_signature_or_band_contract_mismatch() -> None:
    contract, _ = resolve_sut_minhash_contract(_contract())
    assert contract is not None

    with pytest.raises(DedupEvaluationError) as error:
        replay_pairwise_collision([1], [1], contract)
    assert error.value.issue.code == "MINHASH_SIGNATURE_INVALID"

    invalid = _contract()
    invalid["lsh"] = {"bands": 2, "rows_per_band": 2}
    with pytest.raises(DedupEvaluationError) as error:
        resolve_sut_minhash_contract(invalid)
    assert error.value.issue.code == "MINHASH_CONTRACT_INVALID"
