"""Project literal ledger invalidity through the existing frozen native comparison, without inference."""

import argparse
import copy
import csv
import hashlib
import html
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--session", type=Path, required=True)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--best-root", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
session, checkout = args.session.resolve(), args.checkout.resolve()
sys.path[:0] = [str(checkout / "tutorials/eval/dedup"), str(checkout)]
# Native imports follow the caller-selected checkout binding.
from ledger_audit_core import strict_run  # noqa: E402
from policy_eval import LAYERS, read_json, write_json  # noqa: E402
from policy_tuning import REPEATS, aggregate, nonregression_status, target_status  # noqa: E402

destination = session / args.output
destination.mkdir(exist_ok=False)
native_name = args.output + "-native"
env = {**os.environ, "PYTHONPATH": str(checkout / "tutorials/eval/dedup") + ":" + str(checkout)}
subprocess.run(  # noqa: S603
    [
        sys.executable,
        str(checkout / "tutorials/eval/dedup/policy_tuning.py"),
        "compare",
        "--session",
        str(session),
        "--best",
        "0",
        "--best-root",
        args.best_root,
        "--output",
        native_name,
    ],
    cwd=checkout,
    env=env,
    check=True,
    stdout=subprocess.DEVNULL,
)
native = read_json(session / native_name / "comparison.json")
result = copy.deepcopy(native)
evaluation = read_json(session / "baseline/evaluation.json")
group_plans = {
    **{name: (value["panel"], len(value["ids"])) for name, value in evaluation["hard_groups"].items()},
    "personal120": ("diagnostic", 120),
    "real_consensus6": ("diagnostic", len(evaluation["consensus_ids"])),
    "development99_original": ("holdout", 99),
    "development99_reviewed": ("holdout", 99),
    "development97_strict": ("holdout", 97),
}
changes = []
for version, values in native["runs"].items():
    root = session / version
    strict_values = [strict_run(root, value["run"], native=value) for value in values]
    result["runs"][version] = strict_values
    declared = read_json(root / "verification-plan.json")["runs"]
    result["summary"][version] = {
        group: aggregate(strict_values, group, declared[panel], pairs) for group, (panel, pairs) in group_plans.items()
    }
    for before, after in zip(values, strict_values, strict=True):
        if after["ledger_issues"]:
            changes.append(
                {
                    "version": version,
                    "run": after["run"],
                    "ledger_issues": after["ledger_issues"],
                    "native_all_input_validation": before["all_input_validation"],
                    "strict_all_input_validation": after["all_input_validation"],
                }
            )
best = result["summary"][args.best_root]
baseline = result["summary"]["baseline"]
hard_names = [*evaluation["hard_groups"], "real_consensus6"]
hard = {}
for group in hard_names:
    values = best[group]
    complete = all(values[layer]["completed_repeats"] == REPEATS for layer in LAYERS)
    passed = complete and all(
        values[layer]["pair_agreement"]["worst"] == 1 and values[layer]["valid_judgment_coverage"]["worst"] == 1
        for layer in LAYERS
    )
    hard[group] = "PASS" if passed else "FAIL" if complete else "NOT_COMPLETED"
personal = best["personal120"]["final"]
target = target_status(personal, Fraction(9, 10), Fraction(17, 20))
milestone = target_status(personal, Fraction(17, 20), Fraction(4, 5))
nonregression = {
    group: nonregression_status(best[group]["final"], baseline[group]["final"])
    for group in ["development99_original", "development99_reviewed", "development97_strict"]
}
complete = native["verification_completion"]["complete"]
technical = complete and all(
    not run["integrity_failures"]
    and not run["unmapped_attempts"]
    and not run["ledger_issues"]
    and all(count == 0 for status, count in run["attempt_status_counts"].items() if status != "valid")
    and all(counts.get("invalid", 0) == 0 for counts in run["all_input_validation"].values())
    for run in result["runs"][args.best_root]
)
result["requirement_status"] = {
    "milestone_85_80": milestone,
    "target_90_85": target,
    "hard_regressions": hard,
    "technical_validity": "PASS" if technical else "FAIL" if complete else "NOT_COMPLETED",
    "development_nonregression_mean": nonregression,
}
result["requirements"] = {
    "hard_regressions": {key: value == "PASS" for key, value in hard.items()},
    "personal_targets_mean": target == "PASS",
    "personal_targets_each_repeat": personal["completed_repeats"] == REPEATS
    and all(
        Fraction(value["correct_directions"], value["directions"]) >= Fraction(9, 10)
        and Fraction(value["correct_pairs"], value["pairs"]) >= Fraction(17, 20)
        for value in personal["each"]
    ),
    "development_nonregression_mean": {key: value == "PASS" for key, value in nonregression.items()},
}
result["acceptance_passed"] = (
    target == "PASS"
    and technical
    and all(value == "PASS" for value in hard.values())
    and all(value == "PASS" for value in nonregression.values())
)
result["audit_revision"] = {
    "id": "engineering-04-ledger-coherence",
    "new_model_calls": 0,
    "native_comparison": str(session / native_name / "comparison.json"),
    "native_comparison_sha256": hashlib.sha256((session / native_name / "comparison.json").read_bytes()).hexdigest(),
    "strict_core_sha256": hashlib.sha256((Path(__file__).parent / "ledger_audit_core.py").read_bytes()).hexdigest(),
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "labels_denominators_prompts_parameters_native_outputs_unchanged": True,
    "scope": "Literal positive-versus-uncovered-material/conflict/no-basis contradictions and negative-versus-covered-removed-side contradictions become invalid. Negative checks require a verified basis, no conflict and explicit not_translation; body-language retention stays protected. All references and arithmetic reuse policy_tuning functions. Native reports and outputs remain preserved.",
}
result["ledger_audit_changes"] = changes
result["created"] = time.time()
write_json(destination / "comparison.json", result)
flat = [
    {"version": version, "run": run["run"], "group": group, "layer": layer, **metrics}
    for version, runs in result["runs"].items()
    for run in runs
    for group, layers in run["groups"].items()
    for layer, metrics in layers.items()
]
with (destination / "comparison.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
(destination / "comparison.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>Ledger-valid retention comparison</title><pre>'
    + html.escape(
        json.dumps({key: value for key, value in result.items() if key != "runs"}, ensure_ascii=False, indent=2)
    )
    + "</pre>"
)
print(
    json.dumps(
        {
            "output": str(destination),
            "requirement_status": result["requirement_status"],
            "ledger_flagged_runs": len(changes),
            "acceptance_passed": result["acceptance_passed"],
        }
    )
)
