# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make the standalone tutorial modules importable without installing a second package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "tutorials/eval/dedup"))
