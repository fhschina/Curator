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

"""Streaming access to manifest-selected corpus shards and text lookups."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from eval.dedup.config import TokenizerConfig
from eval.dedup.validation import require, sha256_file


def _arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = "pyarrow is required for corpus access"
        raise RuntimeError(msg) from exc
    return pa, pq


class TokenCounter:
    """Frozen tokenizer adapter used for lengths and long-document windows."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.tokenizer: Any | None = None
        self.resolved_revision = config.revision
        self.asset_checksums: dict[str, str] = {}
        if config.kind == "whitespace":
            return
        require(
            config.kind == "huggingface", "UNSUPPORTED_TOKENIZER", "tokenizer kind is not supported", kind=config.kind
        )
        try:
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer
        except ImportError as exc:
            msg = "transformers and huggingface_hub are required for the configured tokenizer"
            raise RuntimeError(msg) from exc
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.revision,
                cache_dir=config.cache_root,
                allow_patterns=[
                    "tokenizer*",
                    "vocab*",
                    "merges.txt",
                    "special_tokens_map.json",
                    "added_tokens.json",
                    "chat_template.jinja",
                ],
            )
        )
        self.resolved_revision = snapshot.name
        require(
            len(self.resolved_revision) == 40
            and all(character in "0123456789abcdef" for character in self.resolved_revision),
            "TOKENIZER_REVISION_NOT_IMMUTABLE",
            "Hugging Face tokenizer revision did not resolve to a commit SHA",
            revision=self.resolved_revision,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        for asset in sorted(snapshot.iterdir()):
            if asset.is_file():
                self.asset_checksums[asset.name] = sha256_file(asset)

    def contract(self) -> dict[str, Any]:
        return {
            "kind": self.config.kind,
            "model_id": self.config.model_id,
            "requested_revision": self.config.revision,
            "resolved_revision": self.resolved_revision,
            "asset_checksums": self.asset_checksums,
            "implementation": "whitespace-test-v1" if self.tokenizer is None else self.tokenizer.__class__.__name__,
        }

    def count_many(self, texts: Sequence[str], *, batch_size: int = 256) -> list[int]:
        if self.tokenizer is None:
            return [len(text.split()) for text in texts]
        lengths: list[int] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + batch_size]),
                add_special_tokens=False,
                return_length=True,
                truncation=False,
                padding=False,
            )
            lengths.extend(int(value) for value in encoded["length"])
        return lengths

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        if self.tokenizer is None:
            import re

            matches = list(re.finditer(r"\S+", text))
            return list(range(len(matches))), [(match.start(), match.end()) for match in matches]
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(encoded["input_ids"]), [tuple(item) for item in encoded["offset_mapping"]]


def iter_corpus_batches(
    corpus_manifest: dict[str, Any],
    *,
    columns: Sequence[str],
    batch_size: int = 16_384,
) -> Iterator[Any]:
    """Yield Arrow record batches with stable global and physical row IDs."""

    pa, pq = _arrow()
    for shard in corpus_manifest["shards"]:
        physical_offset = 0
        parquet_file = pq.ParquetFile(shard["resolved_path"])
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(columns)):
            length = batch.num_rows
            doc_ids = pa.array(
                range(int(shard["start_id"]) + physical_offset, int(shard["start_id"]) + physical_offset + length),
                type=pa.int64(),
            )
            shard_indices = pa.array([int(shard["shard_index"])] * length, type=pa.int32())
            row_indices = pa.array(range(physical_offset, physical_offset + length), type=pa.int32())
            arrays = [doc_ids, shard_indices, row_indices, *batch.columns]
            names = ["doc_id", "shard_index", "physical_row_index", *batch.schema.names]
            yield pa.RecordBatch.from_arrays(arrays, names=names)
            physical_offset += length


def load_documents_by_ids(
    corpus_manifest: dict[str, Any],
    doc_ids: Sequence[int],
    *,
    columns: Sequence[str] = ("text", "url", "timestamp", "language"),
) -> dict[int, dict[str, Any]]:
    """Load selected documents by grouping reads by their manifest shard."""

    _, pq = _arrow()
    sorted_ids = sorted(set(int(value) for value in doc_ids))
    require(len(sorted_ids) == len(set(sorted_ids)), "DUPLICATE_LOOKUP_ID", "document lookup IDs must be unique")
    shards = corpus_manifest["shards"]
    by_shard: dict[int, list[int]] = defaultdict(list)
    shard_index = 0
    for doc_id in sorted_ids:
        while shard_index < len(shards) and doc_id > int(shards[shard_index]["end_id"]):
            shard_index += 1
        require(shard_index < len(shards), "DOC_ID_OUT_OF_RANGE", "document ID is outside corpus", doc_id=doc_id)
        shard = shards[shard_index]
        require(doc_id >= int(shard["start_id"]), "DOC_ID_GAP", "document ID falls in a manifest gap", doc_id=doc_id)
        by_shard[shard_index].append(doc_id)
    result: dict[int, dict[str, Any]] = {}
    for index, ids in by_shard.items():
        shard = shards[index]
        table = pq.read_table(shard["resolved_path"], columns=list(columns))
        values = {name: table[name].to_pylist() for name in columns}
        start_id = int(shard["start_id"])
        for doc_id in ids:
            offset = doc_id - start_id
            result[doc_id] = {
                "doc_id": doc_id,
                "shard_index": index,
                "physical_row_index": offset,
                **{name: values[name][offset] for name in columns},
            }
    require(len(result) == len(sorted_ids), "DOCUMENT_JOIN_INCOMPLETE", "not all selected documents could be loaded")
    return result


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
