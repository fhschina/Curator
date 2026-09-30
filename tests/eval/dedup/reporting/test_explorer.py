from pathlib import Path

from eval.dedup.reporting.explorer import _clip_visible_text, attach_group_context, pair_explorer_destination


def test_pair_explorer_output_is_run_scoped(tmp_path: Path) -> None:
    assert pair_explorer_destination(tmp_path / "final_report.md") == tmp_path / "pair_explorer.html"
    assert pair_explorer_destination(tmp_path / "final_report.sample.md") == tmp_path / "pair_explorer.sample.html"


def test_visible_text_is_bounded() -> None:
    value = _clip_visible_text("a" * 2_000, center=1_000, limit=100)
    assert len(value) == 102
    assert value.startswith("…")
    assert value.endswith("…")


def test_missing_hostnames_do_not_invent_cross_host_risk(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": index,
                    "predicted_group_id": 17,
                    "predicted_cluster_key": "group:17",
                    "action": "KEEP" if index == 0 else "REMOVE",
                    "final_keeper_id": 0,
                    "token_count": 10,
                    "hostname": None,
                    "language": None,
                    "url": None,
                }
                for index in range(21)
            ]
        ),
        tmp_path / "data/document_outcomes.parquet",
    )
    record = {
        "left": {"doc_id": "0", "predicted_group_id": 17, "predicted_group_size": 21},
        "right": {"doc_id": "1", "predicted_group_id": 17, "predicted_group_size": 21},
        "same_hostname": None,
        "judge_evidence_status": {"coverage": "UNAVAILABLE"},
        "reason_codes": [],
    }
    attach_group_context(tmp_path, [record])
    risks = {item["label"]: item["value"] for item in record["risk_indicators"]}
    assert risks["Cross-host pair"] == "UNAVAILABLE"
    assert risks["Possible template-driven grouping"] == "NO_SIGNAL"
