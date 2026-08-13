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

"""Shared schemas, enums, identifiers, and artifact contracts."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

ARTIFACT_SCHEMA_VERSION: Final = "dedup-eval-artifacts-v1"
EVALUATION_MANIFEST_SCHEMA_VERSION: Final = "dedup-eval-manifest-v1"
CANONICAL_PAIR_ID_VERSION: Final = "cp1"
PAIR_DOMAIN_TAG: Final = b"nemo-curator-dedup-pair-v1"


class Action(StrEnum):
    KEEP = "KEEP"
    REMOVE = "REMOVE"


class Track(StrEnum):
    REMOVAL = "5a"
    CROSS_GROUP = "5b"


class DuplicateAnswer(StrEnum):
    YES = "YES"
    NO = "NO"
    UNRESOLVED = "UNRESOLVED"


class RelationType(StrEnum):
    EXACT = "EXACT"
    CANONICAL_EXACT = "CANONICAL_EXACT"
    NEAR_SURFACE = "NEAR_SURFACE"
    CONTAINMENT = "CONTAINMENT"
    VERSION_RELATED = "VERSION_RELATED"
    RELATED_NON_DUPLICATE = "RELATED_NON_DUPLICATE"
    UNRELATED = "UNRELATED"
    UNRESOLVED = "UNRESOLVED"


class MaterialDifference(StrEnum):
    NONE = "NONE"
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    UNRESOLVED = "UNRESOLVED"


class FuzzyScope(StrEnum):
    IN_SCOPE = "IN_SCOPE"
    BORDERLINE = "BORDERLINE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNRESOLVED = "UNRESOLVED"


REASON_CODES: Final = frozenset(
    {
        "NUMBER_CHANGE",
        "DATE_TIME_CHANGE",
        "PRODUCT_VERSION_CHANGE",
        "URL_CHANGE",
        "NAMED_ENTITY_CHANGE",
        "NEGATION_CHANGE",
        "CODE_LITERAL_CHANGE",
        "CODE_OUTPUT_CHANGE",
        "INSERTION_DELETION",
        "BOILERPLATE",
        "PARSER_NOISE",
        "LANGUAGE_MISMATCH",
        "TOPIC_ONLY",
        "INSUFFICIENT_EVIDENCE",
        "OTHER_MATERIAL",
    }
)


DOCUMENT_OUTCOME_COLUMNS: Final = (
    "evaluation_run_id",
    "sut_run_id",
    "doc_id",
    "predicted_group_id",
    "predicted_cluster_key",
    "predicted_group_size",
    "action",
    "final_keeper_id",
    "char_count",
    "token_count",
    "length_bucket",
    "source_id",
    "warc_path",
    "warc_id",
    "url",
    "crawl_timestamp",
    "language",
    "hostname",
    "canonical_url_v0",
    "shard_index",
    "physical_row_index",
)


@dataclass(frozen=True, slots=True)
class CanonicalPair:
    canonical_pair_id: str
    canonical_pair_id_version: str
    doc_id_low: str
    doc_id_high: str


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashes and immutable artifacts."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")


def cp1_pair(doc_i: str | int, doc_j: str | int) -> CanonicalPair:
    """Create the proposal-defined canonical unordered pair identifier."""

    left = str(doc_i).encode("utf-8", errors="strict")
    right = str(doc_j).encode("utf-8", errors="strict")
    if left == right:
        msg = "self-pairs are not valid canonical candidates"
        raise ValueError(msg)
    low_bytes, high_bytes = sorted((left, right))
    payload = (
        PAIR_DOMAIN_TAG
        + struct.pack(">Q", len(low_bytes))
        + low_bytes
        + struct.pack(">Q", len(high_bytes))
        + high_bytes
    )
    pair_id = f"cp1_{hashlib.sha256(payload).hexdigest()}"
    return CanonicalPair(
        canonical_pair_id=pair_id,
        canonical_pair_id_version=CANONICAL_PAIR_ID_VERSION,
        doc_id_low=low_bytes.decode("utf-8"),
        doc_id_high=high_bytes.decode("utf-8"),
    )


def stable_record_id(domain: str, *parts: object) -> str:
    """Return a stable, domain-separated identifier for a provenance record."""

    encoded = [str(part).encode("utf-8") for part in parts]
    payload = domain.encode("utf-8") + b"".join(struct.pack(">Q", len(item)) + item for item in encoded)
    return hashlib.sha256(payload).hexdigest()
