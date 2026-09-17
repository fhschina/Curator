import json
from pathlib import Path

RELEASE = Path(__file__).parents[4] / "eval/dedup/release"


def test_release_manifest_points_only_to_current_assets() -> None:
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["tool_version"] == "v0.7.1"
    assert manifest["judge_contract_version"] == "v0.7"
    assert manifest["lineage"]["original_release_commit"] == "ef17d9b6527532521580781570942728509629bd"
    assert (RELEASE / manifest["schema"]).is_file()
    assert (RELEASE / manifest["smoke_panel"]).is_file()
    panel = json.loads((RELEASE / manifest["smoke_panel"]).read_text(encoding="utf-8"))
    assert len(panel) == len({row["canonical_pair_id"] for row in panel}) == 24
