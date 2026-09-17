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

"""Source-checkout tooling for NeMo Curator deduplication evaluation."""

from eval.dedup.core.contracts import ARTIFACT_SCHEMA_VERSION, CANONICAL_PAIR_ID_VERSION
from eval.dedup.runtime import JUDGE_CONTRACT_VERSION, TOOL_VERSION

RECOMMENDED_JUDGE_VERSION = JUDGE_CONTRACT_VERSION

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_PAIR_ID_VERSION",
    "JUDGE_CONTRACT_VERSION",
    "RECOMMENDED_JUDGE_VERSION",
    "TOOL_VERSION",
]
