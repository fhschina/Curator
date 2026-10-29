"""Relocate native checksum reads without changing frozen evidence or native code."""

import atexit
import builtins
import io
import json
import os
import sys
from pathlib import Path


def install() -> None:  # noqa: C901
    if os.environ.get("RETENTION_REPLAY_GUARD_REQUIRED") != "1":
        return
    original = os.path.abspath(os.environ["RETENTION_REPLAY_ORIGINAL_SESSION"])
    restored = os.path.abspath(os.environ["RETENTION_REPLAY_RESTORED_SESSION"])
    checkout = os.path.abspath(os.environ["RETENTION_REPLAY_RESTORED_CHECKOUT"])
    forbidden = json.loads(os.environ["RETENTION_REPLAY_FORBIDDEN_CHECKOUTS"])
    log_dir = Path(os.environ["RETENTION_REPLAY_ACCESS_LOG_DIRECTORY"])
    log_dir.mkdir(parents=True, exist_ok=True)

    def under(path: str, prefix: str) -> bool:
        return path == prefix or path.startswith(prefix + os.sep)

    if under(restored, original) or under(original, restored):
        message = "Independent replay locations must not overlap"
        raise ValueError(message)
    record = {
        "enabled": True,
        "pid": os.getpid(),
        "original_session": original,
        "restored_session": restored,
        "restored_checkout": checkout,
        "rebased_reads": [],
        "denied_original_reads": [],
        "denied_checkout_reads": [],
        "denied_network_connections": [],
    }
    original_builtin_open, original_io_open = builtins.open, io.open

    def path_text(value: object) -> str | None:
        try:
            return os.path.abspath(os.fsdecode(os.fspath(value)))
        except TypeError:
            return None

    def relocated(value: object, mode: str) -> object:
        path = path_text(value)
        if path is not None and under(path, original):
            if any(character in mode for character in "wa+"):
                message = "Replay cannot write through an original-session path"
                raise PermissionError(message)
            target = restored + path[len(original) :]
            record["rebased_reads"].append({"original": path, "restored": target})
            return target
        return value

    def builtin_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        return original_builtin_open(relocated(file, mode), mode, *args, **kwargs)

    def io_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        return original_io_open(relocated(file, mode), mode, *args, **kwargs)

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open":
            path = path_text(args[0])
            resolved = os.path.realpath(path) if path is not None else None
            if path is not None and (under(path, original) or under(resolved, original)):
                record["denied_original_reads"].append(path)
                message = "Original-session filesystem access is forbidden during restored replay"
                raise PermissionError(message)
            if path is not None and any(under(path, prefix) or under(resolved, prefix) for prefix in forbidden):
                record["denied_checkout_reads"].append(path)
                message = "Original-checkout filesystem access is forbidden during restored replay"
                raise PermissionError(message)
        elif event == "socket.connect":
            record["denied_network_connections"].append(str(args[1]))
            message = "Restored score replay cannot use the network"
            raise PermissionError(message)

    def save() -> None:
        record["native_module_files"] = {
            name: str(module.__file__)
            for name, module in list(sys.modules.items())
            if (name in ["policy_eval", "policy_tuning", "ledger_audit_core"] or name.startswith("nemo_curator"))
            and getattr(module, "__file__", None)
        }
        with original_builtin_open(log_dir / f"{os.getpid()}.json", "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")

    builtins.open, io.open = builtin_open, io_open
    sys.addaudithook(audit)
    atexit.register(save)


try:
    install()
# Python ignores sitecustomize exceptions; a required guard must fail closed.
except Exception as error:  # noqa: BLE001
    sys.stderr.write(f"Required restored-replay guard did not activate: {error}\n")
    os._exit(97)
