from pathlib import Path

from eval.dedup.runtime import contract, state


def test_use_collector_restores_contract_collector() -> None:
    original = contract.collect
    replacement = object()

    with state.use_collector(replacement):
        assert contract.collect is replacement

    assert contract.collect is original


def test_status_reports_latest_completed_session(tmp_path: Path) -> None:
    session = tmp_path / "recovery/sessions/001"
    (session / "heartbeats").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    state.write_json_atomic(session / "started.json", {"pid": -1, "process_start": None})
    state.write_json_atomic(session / "heartbeats/000000.json", {"completed": 0})
    state.write_json_atomic(session / "exit.json", {"status": "COMPLETE"})

    value = state.status(tmp_path)

    assert value["running"] is False
    assert value["exit"] == {"status": "COMPLETE"}
    assert value["last_heartbeat"] == {"completed": 0}
