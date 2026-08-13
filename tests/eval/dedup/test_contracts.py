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

from __future__ import annotations

import hashlib
import struct

import pytest

from eval.dedup.contracts import PAIR_DOMAIN_TAG, cp1_pair


def test_cp1_known_vector_and_order_invariance() -> None:
    payload = PAIR_DOMAIN_TAG + struct.pack(">Q", 2) + b"10" + struct.pack(">Q", 1) + b"2"
    expected = "cp1_" + hashlib.sha256(payload).hexdigest()
    forward = cp1_pair(10, 2)
    reverse = cp1_pair(2, 10)
    assert forward == reverse
    assert forward.canonical_pair_id == expected
    assert forward.doc_id_low == "10"
    assert forward.doc_id_high == "2"


def test_cp1_rejects_self_pair() -> None:
    with pytest.raises(ValueError, match="self-pairs"):
        cp1_pair("12", 12)
