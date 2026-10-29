"""Recompute the selected supplemental groups from restored native outputs."""

import argparse
import hashlib
import json
from pathlib import Path

from ledger_audit_core import strict_run
from policy_eval import verify_files
from replay_restored_evidence import require


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    expected = json.loads(args.summary.read_text())
    selected = session / expected["selected_candidate"]
    require(
        hashlib.sha256((selected / "manifest.json").read_bytes()).hexdigest() == expected["frozen_manifest_sha256"]
    )
    frozen = json.loads((selected / "frozen-code.json").read_text())
    all_passed, verified = True, 0
    for control in expected["groups"]:
        root = session / control["root"]
        require(hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == control["manifest_sha256"])
        manifest = json.loads((root / "manifest.json").read_text())
        verify_files(root, manifest["files"])
        for name, sha in frozen["nine_prompt_config_files"].items():
            require(hashlib.sha256((root / name).read_bytes()).hexdigest() == sha, name)
        for original in control["original_three_runs"]:
            audit = strict_run(root, original["run"])
            require(audit["groups"] == original["groups"], (root.name, original["run"]))
            require(audit["attempt_status_counts"] == original["attempt_status_counts"])
            require(audit["ledger_issues"] == original["ledger_issues"])
            technical = (
                not audit["integrity_failures"] and not audit["unmapped_attempts"] and not audit["ledger_issues"]
            )
            technical &= all(
                count == 0 for status, count in audit["attempt_status_counts"].items() if status != "valid"
            )
            passed = technical and all(
                value["correct_directions"] == value["directions"]
                and value["correct_pairs"] == value["pairs"]
                and value["invalid_pairs"] == 0
                and value["unresolved_pairs"] == 0
                and value["reference_wrong_allows"] == 0
                for layers in audit["groups"].values()
                for value in layers.values()
            )
            require(bool(technical) == original["technical_passed"])
            require(bool(passed) == original["all_layers_passed"])
            all_passed &= passed
            verified += 1
    require(bool(all_passed) == expected["all_controls_passed"])
    require(args.receipt.resolve().is_relative_to(session))
    require(not args.receipt.exists())
    args.receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selected_candidate": expected["selected_candidate"],
                "verified_original_runs": verified,
                "control_roots": len(expected["groups"]),
                "all_groups_and_recorded_gate_outcomes_equal": True,
                "recorded_control_gate": expected["control_gate"],
                "new_model_calls": 0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
