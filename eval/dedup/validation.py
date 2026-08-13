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

"""Blocking validation helpers and immutable artifact I/O."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from eval.dedup.contracts import canonical_json_bytes


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    details: dict[str, Any]


class DedupEvaluationError(RuntimeError):
    """A named, machine-readable pipeline failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(f"{code}: {message}")
        self.issue = ValidationIssue(code=code, message=message, details=details)


class HumanQAPending(DedupEvaluationError):
    """Raised when a full run has reached the intentional human-QA gate."""


def fail(code: str, message: str, **details: Any) -> NoReturn:
    raise DedupEvaluationError(code, message, **details)


def require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        fail(code, message, **details)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DedupEvaluationError(
            "FILE_NOT_FOUND", "required JSON artifact does not exist", path=str(source)
        ) from exc
    except json.JSONDecodeError as exc:
        raise DedupEvaluationError(
            "INVALID_JSON", "JSON artifact could not be parsed", path=str(source), line=exc.lineno, column=exc.colno
        ) from exc
    require(isinstance(value, dict), "INVALID_JSON_ROOT", "JSON artifact must contain an object", path=str(source))
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        if path.exists():
            require(
                path.read_bytes() == payload,
                "IMMUTABLE_ARTIFACT_COLLISION",
                "existing immutable artifact has different content",
                path=str(path),
            )
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, value: Any) -> None:
    _atomic_write(Path(path), canonical_json_bytes(value) + b"\n")


def write_text_atomic(path: str | Path, value: str) -> None:
    _atomic_write(Path(path), value.encode("utf-8"))


def assert_exact_keys(
    value: dict[str, Any], required: set[str], *, context: str, optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    require(not missing, "MISSING_CONFIG_FIELDS", "required fields are missing", context=context, fields=missing)
    require(not unknown, "UNKNOWN_CONFIG_FIELDS", "unknown fields are not allowed", context=context, fields=unknown)


def validate_unique(values: list[Any], *, field: str, context: str) -> None:
    require(
        len(values) == len(set(values)),
        "DUPLICATE_KEY",
        "primary-key field contains duplicates",
        field=field,
        context=context,
        row_count=len(values),
        distinct_count=len(set(values)),
    )
