"""Replay relocated evidence with original project access and network connections forbidden."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def require(condition: bool, detail: object = "Restored replay requirement failed") -> None:
    if not condition:
        raise ValueError(str(detail))


def main() -> None:  # noqa: PLR0915
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--best-root", required=True)
    p.add_argument("--output", default="restored-score-replay")
    p.add_argument("--original-session", type=Path)
    p.add_argument("--access-log-directory", type=Path)
    p.add_argument("--timeout-seconds", type=int, default=600)
    args = p.parse_args()
    session, checkout = (args.session.resolve(), args.checkout.resolve())
    require(Path(args.best_root).name == args.best_root, "Restored replay requirement failed")
    require(
        Path(args.output).name == args.output and args.output not in ["", ".", ".."],
        "Restored replay requirement failed",
    )
    require(not (session / args.output).exists(), "Restored replay requirement failed")
    root = session / args.best_root
    location = session / ".restored-evidence-location.json"
    if args.original_session:
        original = Path(os.path.abspath(args.original_session))
    elif location.exists():
        original = Path(json.loads(location.read_text())["original_session"])
    else:
        original = Path(json.loads((root / "runs/candidate-gate-1/tuning_audit-v2.json").read_text())["root"]).parent
    require(
        original != session and (not session.is_relative_to(original)) and (not original.is_relative_to(session)),
        "Restored replay requirement failed",
    )
    frozen = json.loads((root / "frozen-code.json").read_text())
    for name, expected in frozen["python_files"].items():
        require(hashlib.sha256((checkout / name).read_bytes()).hexdigest() == expected, name)
    for name, expected in frozen["nine_prompt_config_files"].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name)
    strict_reference = json.loads(
        (root / "runs/candidate-gate-1/audits/engineering04-ledger-coherence/strict_audit.json").read_text()
    )
    require(
        hashlib.sha256((session / "ledger_audit_core.py").read_bytes()).hexdigest()
        == strict_reference["strict_audit_sha256"],
        "Restored replay requirement failed",
    )
    location.write_text(
        json.dumps({"original_session": str(original), "restored_session": str(session)}, indent=2) + "\n"
    )
    state = json.loads((session / "SESSION_STATUS.json").read_text())
    old_checkout = Path(state["workspace"])
    forbidden = [
        str(old_checkout),
        str(old_checkout.with_name("Curator")),
        str(old_checkout.with_name("Curator-retention-policy-v4-gpu-checks")),
    ]
    logs = (args.access_log_directory or session / (args.output + "-access")).resolve()
    logs.mkdir(parents=True, exist_ok=False)
    guard = Path(__file__).resolve().parent / "replay_location_guard"
    require((guard / "sitecustomize.py").is_file(), "Restored replay requirement failed")
    env = {
        **os.environ,
        "PYTHONPATH": f"{guard}:{session}:{checkout}/tutorials/eval/dedup:{checkout}",
        "RETENTION_REPLAY_GUARD_REQUIRED": "1",
        "RETENTION_REPLAY_ORIGINAL_SESSION": str(original),
        "RETENTION_REPLAY_RESTORED_SESSION": str(session),
        "RETENTION_REPLAY_RESTORED_CHECKOUT": str(checkout),
        "RETENTION_REPLAY_FORBIDDEN_CHECKOUTS": json.dumps(forbidden),
        "RETENTION_REPLAY_ACCESS_LOG_DIRECTORY": str(logs),
    }
    command = [
        sys.executable,
        str(session / "compare_strict.py"),
        "--session",
        str(session),
        "--checkout",
        str(checkout),
        "--best-root",
        args.best_root,
        "--output",
        args.output,
    ]
    started = time.time()
    process = subprocess.Popen(command, cwd=checkout, env=env, start_new_session=True)  # noqa: S603
    try:
        returncode = process.wait(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    require(returncode == 0, returncode)
    additional_replay = None
    verification_path = session / "verification-final.json"
    if verification_path.exists():
        additional_pointer = (
            json.loads(verification_path.read_text()).get("actual_additional_controls", {}).get("receipt")
        )
        if additional_pointer:
            additional_receipt = session / (args.output + "-additional-controls.json")
            subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    str(Path(__file__).with_name("replay_supplemental_outputs.py")),
                    "--session",
                    str(session),
                    "--summary",
                    str(session / additional_pointer),
                    "--receipt",
                    str(additional_receipt),
                ],
                cwd=checkout,
                env=env,
                check=True,
                timeout=max(1, args.timeout_seconds - (time.time() - started)),
            )
            additional_replay = json.loads(additional_receipt.read_text())
    records = [json.loads(path.read_text()) for path in sorted(logs.glob("*.json"))]
    require(records and all(item["enabled"] for item in records), "Restored replay requirement failed")
    rebased = sum(len(item["rebased_reads"]) for item in records)
    require(rebased > 0, "Native absolute checksum reads were not exercised")
    require(
        all(
            not item[key]
            for item in records
            for key in ["denied_original_reads", "denied_checkout_reads", "denied_network_connections"]
        ),
        "Restored replay requirement failed",
    )
    for item in records:
        for name, filename in item["native_module_files"].items():
            expected_root = session if name == "ledger_audit_core" else checkout
            require(Path(filename).resolve().is_relative_to(expected_root), (name, filename))
    receipt = {
        "status": "PASS",
        "started": started,
        "completed": time.time(),
        "original_session": str(original),
        "restored_session": str(session),
        "restored_checkout": str(checkout),
        "output": args.output,
        "guarded_python_processes": len(records),
        "native_absolute_checksum_reads_relocated": rebased,
        "original_session_and_source_checkout_reads_blocked": True,
        "network_connections_blocked": True,
        "raw_native_outputs_frozen_inputs_and_native_source_not_rewritten": True,
        "selected_additional_control_replay": additional_replay,
        "new_model_calls": 0,
        "access_log_directory": str(logs),
        "guard_sha256": hashlib.sha256((guard / "sitecustomize.py").read_bytes()).hexdigest(),
    }
    (session / (args.output + "-location-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
