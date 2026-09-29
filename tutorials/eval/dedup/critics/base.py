# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fixed prepare/generate/apply interface for optional judge critics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import data_designer.config as dd


class Critic(Protocol):
    name: str
    output_columns: tuple[str, ...]
    prepared_columns: tuple[str, ...]
    applied_columns: tuple[str, ...]
    temporary_columns: tuple[str, ...]

    def prepare(self, record: dict[str, Any]) -> dict[str, Any]: ...

    def build_column(self, *, model_alias: str, prompt: str, system_prompt: str) -> dd.LLMStructuredColumnConfig: ...

    def apply(self, record: dict[str, Any]) -> dict[str, Any]: ...
