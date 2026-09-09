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

"""Optional deterministic MinHash diagnostics using the resolved SUT contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from eval.dedup.validation import require, sha256_json

MINHASH_DIAGNOSTIC_SCHEMA: Final = "dedup-minhash-diagnostic-v1"
UNAVAILABLE_MISSING_CONTRACT: Final = "UNAVAILABLE_MISSING_CONTRACT"
UNAVAILABLE_MISSING_SIGNATURES: Final = "UNAVAILABLE_MISSING_SIGNATURES"
AVAILABLE: Final = "AVAILABLE"

_REQUIRED_CONTRACT_PATHS = (
    "normalization.name",
    "shingling.kind",
    "shingling.width",
    "hash.family",
    "hash.num_hashes",
    "hash.seed",
    "lsh.bands",
    "lsh.rows_per_band",
)


@dataclass(frozen=True, slots=True)
class SutMinHashContract:
    normalization: dict[str, Any]
    shingling: dict[str, Any]
    hash: dict[str, Any]
    lsh: dict[str, Any]
    source: str

    @property
    def num_hashes(self) -> int:
        return int(self.hash["num_hashes"])

    @property
    def bands(self) -> int:
        return int(self.lsh["bands"])

    @property
    def rows_per_band(self) -> int:
        return int(self.lsh["rows_per_band"])

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "normalization": self.normalization,
                "shingling": self.shingling,
                "hash": self.hash,
                "lsh": self.lsh,
                "source": self.source,
            }
        )


def _get_path(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def resolve_sut_minhash_contract(
    value: dict[str, Any] | None,
) -> tuple[SutMinHashContract | None, dict[str, Any]]:
    """Resolve a real SUT contract or return an explicit unavailable diagnostic."""

    missing = (
        list(_REQUIRED_CONTRACT_PATHS)
        if value is None
        else [path for path in _REQUIRED_CONTRACT_PATHS if _get_path(value, path) is None]
    )
    source = value.get("source") if isinstance(value, dict) else None
    if source != "sut_resolved_config":
        missing.append("source=sut_resolved_config")
    if missing:
        return None, {
            "schema_version": MINHASH_DIAGNOSTIC_SCHEMA,
            "status": UNAVAILABLE_MISSING_CONTRACT,
            "missing_contract_fields": sorted(set(missing)),
        }
    require(value is not None, "MINHASH_CONTRACT_INVALID", "resolved contract cannot be null")
    contract = SutMinHashContract(
        normalization=dict(value["normalization"]),
        shingling=dict(value["shingling"]),
        hash=dict(value["hash"]),
        lsh=dict(value["lsh"]),
        source=str(value["source"]),
    )
    require(
        isinstance(contract.normalization["name"], str) and bool(contract.normalization["name"].strip()),
        "MINHASH_CONTRACT_INVALID",
        "normalization name must be non-empty",
    )
    require(
        isinstance(contract.shingling["kind"], str) and bool(contract.shingling["kind"].strip()),
        "MINHASH_CONTRACT_INVALID",
        "shingling kind must be non-empty",
    )
    require(
        isinstance(contract.shingling["width"], int)
        and not isinstance(contract.shingling["width"], bool)
        and contract.shingling["width"] > 0,
        "MINHASH_CONTRACT_INVALID",
        "shingling width must be a positive integer",
    )
    require(
        isinstance(contract.hash["family"], str) and bool(contract.hash["family"].strip()),
        "MINHASH_CONTRACT_INVALID",
        "hash family must be non-empty",
    )
    require(
        isinstance(contract.hash["num_hashes"], int)
        and not isinstance(contract.hash["num_hashes"], bool)
        and isinstance(contract.hash["seed"], int)
        and not isinstance(contract.hash["seed"], bool),
        "MINHASH_CONTRACT_INVALID",
        "num_hashes and seed must be integers",
    )
    require(
        isinstance(contract.lsh["bands"], int)
        and not isinstance(contract.lsh["bands"], bool)
        and isinstance(contract.lsh["rows_per_band"], int)
        and not isinstance(contract.lsh["rows_per_band"], bool),
        "MINHASH_CONTRACT_INVALID",
        "bands and rows_per_band must be integers",
    )
    require(contract.num_hashes > 0, "MINHASH_CONTRACT_INVALID", "num_hashes must be positive")
    require(contract.bands > 0, "MINHASH_CONTRACT_INVALID", "bands must be positive")
    require(contract.rows_per_band > 0, "MINHASH_CONTRACT_INVALID", "rows_per_band must be positive")
    require(
        contract.bands * contract.rows_per_band == contract.num_hashes,
        "MINHASH_CONTRACT_INVALID",
        "bands multiplied by rows_per_band must equal num_hashes for exact replay",
    )
    return contract, {
        "schema_version": MINHASH_DIAGNOSTIC_SCHEMA,
        "status": AVAILABLE,
        "sut_contract_digest": contract.digest,
    }


def replay_pairwise_collision(
    signature_a: Sequence[int],
    signature_b: Sequence[int],
    contract: SutMinHashContract,
) -> dict[str, Any]:
    """Replay the exact band collision decision from two SUT-produced signatures."""

    require(
        len(signature_a) == len(signature_b) == contract.num_hashes,
        "MINHASH_SIGNATURE_INVALID",
        "both signatures must match the SUT num_hashes contract",
        expected=contract.num_hashes,
        signature_a=len(signature_a),
        signature_b=len(signature_b),
    )
    require(
        all(isinstance(item, int) and not isinstance(item, bool) for item in (*signature_a, *signature_b)),
        "MINHASH_SIGNATURE_INVALID",
        "MinHash signatures must contain integers",
    )
    matched_bands = []
    for band in range(contract.bands):
        start = band * contract.rows_per_band
        end = start + contract.rows_per_band
        if tuple(signature_a[start:end]) == tuple(signature_b[start:end]):
            matched_bands.append(band)
    matched_hashes = sum(left == right for left, right in zip(signature_a, signature_b, strict=True))
    collision = bool(matched_bands)
    return {
        "schema_version": MINHASH_DIAGNOSTIC_SCHEMA,
        "status": AVAILABLE,
        "sut_contract_digest": contract.digest,
        "matched_hashes": matched_hashes,
        "signature_similarity": matched_hashes / contract.num_hashes,
        "matched_band_indices": matched_bands,
        "pairwise_collision": collision,
        "minhash_action": "GROUP" if collision else "KEEP_SEPARATE",
    }


def diagnose_pairwise_minhash(
    *,
    contract_value: dict[str, Any] | None,
    signature_a: Sequence[int] | None = None,
    signature_b: Sequence[int] | None = None,
    semantic_same_duplicate_group: str | None = None,
) -> dict[str, Any]:
    """Return deterministic diagnostics; never ask an LLM to guess missing SUT behavior."""

    contract, status = resolve_sut_minhash_contract(contract_value)
    if contract is None:
        return status
    if signature_a is None or signature_b is None:
        return {
            "schema_version": MINHASH_DIAGNOSTIC_SCHEMA,
            "status": UNAVAILABLE_MISSING_SIGNATURES,
            "sut_contract_digest": contract.digest,
            "missing_contract_fields": ["sut_signature_a", "sut_signature_b"],
        }
    result = replay_pairwise_collision(signature_a, signature_b, contract)
    semantic_alignment = None
    if semantic_same_duplicate_group in {"YES", "NO"}:
        semantic_alignment = {
            ("YES", "GROUP"): "ALIGNED",
            ("NO", "KEEP_SEPARATE"): "ALIGNED",
            ("YES", "KEEP_SEPARATE"): "UNDER_GROUP_RISK",
            ("NO", "GROUP"): "OVER_GROUP_RISK",
        }[(semantic_same_duplicate_group, result["minhash_action"])]
    return {**result, "semantic_alignment": semantic_alignment}
