# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import subprocess
import sys

import pytest

from eval.dedup.analysis import retention_v4_execution as subject
from eval.dedup.validation import DedupEvaluationError


def test_cold_process_serial_warmup_makes_concurrent_tokenizer_access_safe():
    program = """
from concurrent.futures import ThreadPoolExecutor
from eval.dedup.analysis.retention_v4_execution import warmup
from eval.dedup.analysis.retention_v4_experiment import local_tokenizer
initialized = warmup()
with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda _: local_tokenizer(), range(8)))
assert all(value is initialized for value in results)
print('SERIAL_WARMUP_PARALLEL_ACCESS_OK')
"""
    result = subprocess.run(  # noqa: S603 - fixed test program in the existing interpreter
        [sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=60
    )
    assert "SERIAL_WARMUP_PARALLEL_ACCESS_OK" in result.stdout


def test_new_execution_requires_a_bound_manifest_before_key_loading_or_calls(tmp_path):
    with pytest.raises((FileNotFoundError, DedupEvaluationError)):
        subject.run(tmp_path, tmp_path / "never_read.env")
