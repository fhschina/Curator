# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional critics for LLM judge workflows."""

from nemo_curator.eval.llm_judge.critics.coverage import CoverageCritic

BUILTIN_CRITICS = {"coverage": CoverageCritic}
