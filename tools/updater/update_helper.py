"""
Standalone helper that swaps a staged macOS app bundle into place.

This script is copied to the user-data updater directory and launched with the
system Python so it can continue running after the main app exits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _wait_for_pid(pid: int, timeout_seconds: int = 30) -> None:
    """Wait for a PID to exit."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.2)


def _write_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip() + "\n")


def main() -> int:
    if len(sys.argv) < 4:
        return 1

    pending_path = Path(sys.argv[1])
    launcher_pid = int(sys.argv[2])
    log_path = Path(sys.argv[3])

    if not pending_path.exists():
        _write_log(log_path, "Pending update marker missing; nothing to do.")
        return 0

    with open(pending_path, "r", encoding="utf-8") as pending_file:
        data = json.load(pending_file)

    install_target = Path(data["install_target"])
    staged_bundle = Path(data["staged_bundle_path"])
    staging_dir = Path(data["staging_dir"])
    download_path = Path(data["download_path"])
    backup_bundle = install_target.with_name(f"{install_target.stem}.previous.app")

    _write_log(log_path, f"Waiting for launcher PID {launcher_pid} to exit")
    _wait_for_pid(launcher_pid)

    if not staged_bundle.exists():
        _write_log(log_path, "Staged bundle does not exist; aborting update.")
        return 1

    try:
        if backup_bundle.exists():
            shutil.rmtree(backup_bundle)

        if install_target.exists():
            shutil.move(str(install_target), str(backup_bundle))

        shutil.move(str(staged_bundle), str(install_target))
        pending_path.unlink(missing_ok=True)

        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        if download_path.exists():
            download_path.unlink(missing_ok=True)
        if backup_bundle.exists():
            shutil.rmtree(backup_bundle, ignore_errors=True)

        _write_log(log_path, f"Installed update to {install_target}")
        subprocess.Popen(["open", str(install_target)])
        return 0
    except Exception as exc:
        _write_log(log_path, f"Update helper failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
