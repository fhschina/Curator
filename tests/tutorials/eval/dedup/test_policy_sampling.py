# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from policy_cases import digest
from policy_sampling import fetch_documents


def test_dense_manifest_physical_offsets_and_hashes(tmp_path: Path):
    path = tmp_path / "extracted_text/objects/source/text.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": ["first", "second"]}), path)
    manifest = {
        "shards": [
            {"start_id": 100, "end_id": 101, "rows": 2, "source_key": "source", "sha256": digest(path.read_bytes())}
        ]
    }
    (tmp_path / "manifest").mkdir()
    (tmp_path / "manifest/dense_10m_explicit.json").write_text(json.dumps(manifest))
    docs, _hashes = fetch_documents(tmp_path, {101})
    assert docs == {101: "second"}
    pq.write_table(pa.table({"text": ["tampered", "second"]}), path)
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch_documents(tmp_path, {101})
