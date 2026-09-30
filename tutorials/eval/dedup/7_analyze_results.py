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

"""
Summarize the fuzzy-dedup-eval LLM judge output (step 7): bucket the judge's
`relation_type` verdict into duplicate/not_duplicate/unresolved and compare
it against what each pair's `pair_type` implied fuzzy dedup decided
(`expected_duplicate`, written by 3_build_pair_dataset.py).

Example:
    python tutorials/eval/dedup/7_analyze_results.py \
        --judge-output-path output/dedup_eval/judged_pairs \
        --disagreements-output output/dedup_eval/disagreements.jsonl
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from loguru import logger

JUDGE_COLUMN = "pair_semantic_judgment"
SCORE_NAME = "relation_type"
DUPLICATE_RELATIONS = {"exact", "canonical_exact", "near_surface", "containment"}
NOT_DUPLICATE_RELATIONS = {"version_related", "related_non_duplicate", "unrelated"}
_DISAGREEMENT_COLUMNS = [
    "pair_id",
    "pair_type",
    "expected_duplicate",
    "doc_id_a",
    "doc_id_b",
    "truncated",
    "decision_source",
    "relation_type_raw",
    "relation_type",
    "relation_type_corrected",
    "relation_type_containment_corrected",
    "material_difference",
    "primary_material_difference",
    "confidence_tier",
    "primary_risk_factor",
    "dominant_overlap_source",
    JUDGE_COLUMN,
    "final_decision",
    "coverage_should_run",
    "coverage_review",
    "coverage_action",
    "coverage_reason",
    "coverage_evidence",
]


def _extract_field(row: dict | None, field_name: str) -> str | None:
    if not isinstance(row, dict):
        return None
    return row.get(field_name, {}).get("score")


def _correct_identical_text_relation(df: pd.DataFrame) -> pd.DataFrame:
    """Correct main relations only for complete, untruncated, equal nonempty original texts."""
    diff = df["semantic_diff"]
    is_complete = diff.map(lambda d: isinstance(d, dict) and d.get("status") == "COMPLETE")
    packet_untruncated = diff.map(
        lambda d: isinstance(d, dict)
        and all(d.get(key) is False for key in ("truncated", "truncated_a", "truncated_b"))
    )
    raw_equal = df.apply(
        lambda row: isinstance(row.get("text_a"), str)
        and bool(row["text_a"].strip())
        and row["text_a"] == row.get("text_b"),
        axis=1,
    )
    eligible = is_complete & packet_untruncated & raw_equal & ~df["truncated"] & df["relation_type"].notna()
    df["relation_type_corrected"] = eligible & (df["relation_type"] != "exact")
    df.loc[df["relation_type_corrected"], "relation_type"] = "exact"
    return df


def _correct_invalid_containment_relation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically correct `relation_type` away from `containment` when the judge violated
    one of its own preconditions for that verdict.
    """
    profile_a = df[JUDGE_COLUMN].map(lambda row: _extract_field(row, "span_content_profile_a"))
    profile_b = df[JUDGE_COLUMN].map(lambda row: _extract_field(row, "span_content_profile_b"))
    shared_basis = df[JUDGE_COLUMN].map(lambda row: _extract_field(row, "span_shared_basis"))
    profile_mismatch = profile_a.notna() & profile_b.notna() & (profile_a != profile_b)
    both_non_main = (profile_a == "non_main_only") & (profile_b == "non_main_only")
    no_shared_basis = shared_basis == "none"
    invalid_containment = profile_mismatch | both_non_main | no_shared_basis
    eligible = invalid_containment & (df["relation_type"] == "containment")
    df["relation_type_containment_corrected"] = eligible
    df.loc[eligible, "relation_type"] = "related_non_duplicate"
    return df


def annotate(df: pd.DataFrame, *, decision_source: str = "main", apply_main_corrections: bool = False) -> pd.DataFrame:
    """Add `relation_type`, `verdict`, and `is_disagreement` columns to the raw judge output."""

    if decision_source not in {"main", "final"}:
        msg = "decision_source must be 'main' or 'final'."
        raise ValueError(msg)
    if apply_main_corrections and decision_source != "main":
        msg = "Main analysis corrections cannot be applied to final decisions."
        raise ValueError(msg)
    decision_column = JUDGE_COLUMN if decision_source == "main" else "final_decision"
    if decision_column not in df.columns:
        msg = f"Expected judge output column {decision_column!r} not found; columns present: {list(df.columns)}"
        raise KeyError(msg)
    if "pair_type" not in df.columns or "expected_duplicate" not in df.columns:
        msg = (
            "Expected passthrough columns 'pair_type'/'expected_duplicate' not found in judge output. "
            "These come from 3_build_pair_dataset.py's pairs part files and must survive into the judge's output "
            "(Data Designer preserves original input columns alongside judge columns)."
        )
        raise KeyError(msg)

    df = df.copy()
    df["decision_source"] = decision_source
    for field_name in (
        SCORE_NAME,
        "material_difference",
        "primary_material_difference",
        "confidence_tier",
        "primary_risk_factor",
        "dominant_overlap_source",
    ):
        df[field_name] = df[decision_column].map(
            lambda row, f=field_name: (
                _extract_field(row, f) if decision_source == "main" else row.get(f) if isinstance(row, dict) else None
            )
        )
    df["relation_type_raw"] = df["relation_type"]

    df["truncated"] = df["truncated"].fillna(False).astype(bool) if "truncated" in df.columns else False

    df["relation_type_corrected"] = False
    df["relation_type_containment_corrected"] = False
    if apply_main_corrections:
        if "semantic_diff" in df.columns:
            df = _correct_identical_text_relation(df)
        df = _correct_invalid_containment_relation(df)

    df["verdict"] = df["relation_type"].map(
        lambda relation: (
            "duplicate"
            if relation in DUPLICATE_RELATIONS
            else "not_duplicate"
            if relation in NOT_DUPLICATE_RELATIONS
            else "unresolved"
        )
    )

    duplicate_expected = df["expected_duplicate"].astype(bool)
    df["is_disagreement"] = (duplicate_expected & (df["verdict"] != "duplicate")) | (
        ~duplicate_expected & (df["verdict"] == "duplicate")
    )

    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an already-`annotate`d dataframe into one row per `pair_type`.

    Truncated pairs are counted but excluded from the disagreement rate: a judge verdict on a
    pair where the judge never saw the full document isn't conclusive about the full documents.
    """
    rows = []

    for pair_type, group in df.groupby("pair_type"):
        total = len(group)
        num_truncated = int(group["truncated"].sum())
        conclusive = group[~group["truncated"]]
        relation_counts = conclusive["relation_type"].value_counts().to_dict()
        expected = bool(group["expected_duplicate"].iloc[0])
        disagreement_label = "judge_says_not_duplicate_rate" if expected else "judge_says_duplicate_rate"
        num_conclusive = len(conclusive)
        rows.append(
            {
                "pair_type": pair_type,
                "num_pairs": total,
                "num_truncated": num_truncated,
                "expected_duplicate": expected,
                **{f"relation_{k}": v for k, v in relation_counts.items()},
                disagreement_label: conclusive["is_disagreement"].sum() / num_conclusive
                if num_conclusive
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--judge-output-path",
        required=True,
        help="Directory of JSONL or Parquet part files written by step 5 or step 6.",
    )
    parser.add_argument(
        "--decision-source",
        choices=("main", "final"),
        default="main",
        help="Analyze the original main judgment or the coverage-adjusted final decision.",
    )
    parser.add_argument(
        "--apply-main-corrections",
        action="store_true",
        help="Opt in to exact-text and containment corrections for a separate main-only diagnostic report.",
    )
    parser.add_argument(
        "--disagreements-output",
        default=None,
        help=(
            "Optional JSONL path to write every disagreeing pair to, with its full rubric output, for manual "
            "review. Without this, the aggregate rate alone doesn't tell you which pairs to go read."
        ),
    )
    args = parser.parse_args()
    if args.apply_main_corrections and args.decision_source != "main":
        parser.error("--apply-main-corrections requires --decision-source main")
    return args


def main() -> None:
    args = _parse_args()

    jsonl_files = sorted(glob.glob(str(Path(args.judge_output_path) / "*.jsonl")))
    parquet_files = sorted(glob.glob(str(Path(args.judge_output_path) / "*.parquet")))
    if jsonl_files:
        frames = [pd.read_json(f, lines=True) for f in jsonl_files]
    elif parquet_files:
        frames = [pd.read_parquet(f) for f in parquet_files]
    else:
        msg = f"No .jsonl or .parquet part files found under {args.judge_output_path}"
        raise FileNotFoundError(msg)
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Loaded {len(df)} judged pairs from {args.judge_output_path}")

    df = annotate(df, decision_source=args.decision_source, apply_main_corrections=args.apply_main_corrections)
    logger.info(
        f"Analyzing {args.decision_source} decisions; main analysis corrections "
        f"{'enabled (do not use for raw main/final comparisons)' if args.apply_main_corrections else 'disabled'}."
    )

    num_corrected = int(df["relation_type_corrected"].sum())
    if num_corrected:
        logger.info(
            f"Deterministically corrected relation_type to 'exact' for {num_corrected}/{len(df)} pairs where "
            "the complete, untruncated original texts were identical but the judge said otherwise."
        )

    num_containment_corrected = int(df["relation_type_containment_corrected"].sum())
    if num_containment_corrected:
        logger.info(
            f"Deterministically corrected relation_type from 'containment' to 'related_non_duplicate' for "
            f"{num_containment_corrected}/{len(df)} pairs where the judge violated one of containment's own "
            "preconditions (profile mismatch, both sides non_main_only, or no verified shared basis)."
        )

    summary = summarize(df)
    logger.warning(
        "These rates are diagnostics from one unvalidated LLM judge on a minimal example config -- "
        "not calibrated fuzzy-dedup accuracy. Read a sample of disagreements before drawing conclusions."
    )

    num_truncated = int(df["truncated"].sum())
    if num_truncated:
        logger.warning(
            f"{num_truncated}/{len(df)} pairs had a truncated document (see 'num_truncated' per pair_type "
            "below) and were excluded from the disagreement rate -- the judge never saw the full text for "
            "those pairs, so its verdict isn't conclusive about them."
        )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))

    if args.disagreements_output:
        disagreements = df[df["is_disagreement"]]
        columns = [c for c in _DISAGREEMENT_COLUMNS if c in disagreements.columns]
        Path(args.disagreements_output).parent.mkdir(parents=True, exist_ok=True)
        disagreements[columns].to_json(args.disagreements_output, orient="records", lines=True, force_ascii=False)
        num_written = len(disagreements)
        logger.info(f"Wrote {num_written} disagreeing pairs to {args.disagreements_output}")


if __name__ == "__main__":
    main()
