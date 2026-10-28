"""Replay the frozen round024 comparison through the native evaluator on one caller-owned service."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-session", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--endpoint", required=True)
parser.add_argument("--container", required=True)
parser.add_argument("--deadline-utc", required=True, help="Explicit future absolute UTC cutoff, ending in Z")
parser.add_argument("--best-root", help="Defaults to the immutable source final selection")
parser.add_argument("--plan-only", action="store_true")
parser.add_argument(
    "--prepare-only",
    action="store_true",
    help="Copy and verify the complete frozen replay inputs without any inference",
)
parser.add_argument(
    "--audit-tools",
    type=Path,
    default=None,
    help="Directory containing the frozen compare_strict.py and ledger_audit_core.py",
)
args = parser.parse_args()
if args.plan_only and args.prepare_only:
    parser.error("Choose one dry mode")
if sys.prefix != "/raid/hfang/llm_judge_env_pr2324_latest":
    parser.error("Use the existing /raid/hfang/llm_judge_env_pr2324_latest environment")
if not args.deadline_utc.endswith("Z"):
    parser.error("The deadline must be an explicit UTC timestamp ending in Z")
deadline = datetime.fromisoformat(args.deadline_utc).timestamp()
if deadline <= time.time() + 3600:
    parser.error(
        "The explicit new replay deadline must leave more than an hour; expired task deadlines are never extended implicitly"
    )
source, destination, checkout = args.source_session.resolve(), args.destination.resolve(), args.checkout.resolve()
args.audit_tools = (args.audit_tools or source).resolve()
selection = json.loads((source / "final-selection.json").read_text()) if not args.best_root else {}
best = args.best_root or selection["best_candidate"]
names = list(dict.fromkeys(["baseline", "candidate-00-start85", best]))
if not best.startswith("candidate-") or Path(best).name != best:
    parser.error("The selected candidate must be one frozen direct candidate root")
sys.path[:0] = [str(checkout / "tutorials/eval/dedup"), str(checkout)]
from policy_eval import verify_files  # noqa: E402
from policy_runtime import environment_receipt, existing_qwen_service  # noqa: E402
from policy_tuning import declare_verification  # noqa: E402

environment_receipt(checkout)
for name in names:
    manifest = json.loads((source / name / "manifest.json").read_text())
    verify_files(source / name, manifest["files"])
    code_path = source / name / "frozen-code.json"
    if code_path.exists():
        for filename, expected in json.loads(code_path.read_text())["python_files"].items():
            if hashlib.sha256((checkout / filename).read_bytes()).hexdigest() != expected:
                parser.error("The replay checkout differs from the evaluated native source: " + filename)
for filename in ["compare_strict.py", "ledger_audit_core.py"]:
    if not (args.audit_tools / filename).is_file():
        parser.error("Missing frozen shared audit source: " + filename)
plan = {
    "source_session": str(source),
    "destination": str(destination),
    "versions": names,
    "comparison_roles": {"baseline": "baseline", "starting": "candidate-00-start85", "selected": best},
    "identical_start_and_best_evaluated_once": best == "candidate-00-start85",
    "repeats": [1, 2, 3],
    "panels": ["gate", "coverage-regression", "preflight", "diagnostic", "holdout"],
    "deadline_utc": args.deadline_utc,
    "gpu_stop_utc": datetime.fromtimestamp(deadline - 900, UTC).isoformat(),
    "service_lifecycle": "The explicitly supplied service is caller-owned. Install its absolute shutdown guard before running this replay and release it at the stated GPU cutoff.",
}
if args.plan_only:
    print(json.dumps(plan, indent=2))
    raise SystemExit(0)
if not args.prepare_only:
    existing_qwen_service(args.container, args.endpoint)
destination.mkdir(parents=True, exist_ok=False)
(destination / "replay-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
(destination / "SESSION_STATUS.json").write_text(
    json.dumps(
        {
            "start_time": time.time(),
            "deadline": deadline,
            "gpu_stop_at": deadline - 900,
            "max_gpus": 1,
            "source_session": str(source),
            "purpose": "frozen_replay_no_semantic_search",
        },
        indent=2,
    )
    + "\n"
)
(destination / "service.json").write_text(
    json.dumps({"endpoint": args.endpoint, "container": args.container, "caller_owned": True}, indent=2) + "\n"
)


env = {**os.environ, "PYTHONPATH": str(checkout / "tutorials/eval/dedup") + ":" + str(checkout)}
for name in names:
    root = destination / name
    manifest = json.loads((source / name / "manifest.json").read_text())
    for filename in sorted(set(manifest["files"]) | {"manifest.json", "private_index.json"}):
        target = root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name / filename, target)
    variant = "baseline" if name == "baseline" else "candidate"
    declare_verification(root, variant)
    for filename in ["frozen-code.json", "hypothesis.json"]:
        if (source / name / filename).exists():
            shutil.copy2(source / name / filename, root / filename)
    if args.prepare_only:
        continue
    command = [
        sys.executable,
        str(checkout / "tutorials/eval/dedup/policy_tuning.py"),
        "execute",
        "--session",
        str(destination),
        "--root",
        str(root),
        "--variant",
        variant,
        "--panels",
        *plan["panels"],
        "--repeats",
        "1",
        "2",
        "3",
        "--reserve-seconds",
        "900",
    ]
    with (destination / f"{name}.log").open("w") as stream:
        subprocess.run(  # noqa: S603
            command,
            cwd=checkout,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=max(1, deadline - 900 - time.time()),
        )
if args.prepare_only:
    print(
        json.dumps(
            {
                "prepared": time.time(),
                "destination": str(destination),
                "model_calls": 0,
                "inference_status": "NOT_COMPLETED",
            }
        )
    )
    raise SystemExit(0)
subprocess.run(  # noqa: S603
    [
        sys.executable,
        str(args.audit_tools.resolve() / "compare_strict.py"),
        "--session",
        str(destination),
        "--checkout",
        str(checkout),
        "--best-root",
        best,
        "--output",
        "comparison",
    ],
    cwd=checkout,
    env=env,
    check=True,
)
print(
    json.dumps(
        {
            "completed": time.time(),
            "destination": str(destination),
            "service_shutdown": "Caller must stop its owned service and preserve its lifecycle receipt.",
        }
    )
)
