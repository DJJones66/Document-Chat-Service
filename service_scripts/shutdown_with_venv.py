#!/usr/bin/env python3
"""
Stop the FastAPI service started with the local venv.

Usage:
  python service_scripts/shutdown_with_venv.py

Environment overrides:
  PID_FILE           PID file to check before searching (default: .run/uvicorn.pid)
  PROCESS_NAME       Process name to match when PID file is absent (default: uvicorn)
  PROCESS_CMD_MATCH  Command-line substring to validate targets (default: app.main:app)
  PROCESS_PORT       Expected service port (falls back to API_PORT when set)
  SHUTDOWN_TIMEOUT   Seconds to wait after SIGTERM before force killing (default: 10)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

from venv_utils import REPO_ROOT

DEFAULT_PID_FILE = Path(os.environ.get("PID_FILE", REPO_ROOT / ".run" / "uvicorn.pid"))
DEFAULT_PROCESS_NAME = os.environ.get("PROCESS_NAME", "uvicorn")
DEFAULT_CMD_MATCH = os.environ.get("PROCESS_CMD_MATCH", "app.main:app")
DEFAULT_PROCESS_PORT = os.environ.get("PROCESS_PORT") or os.environ.get("API_PORT")
DEFAULT_TIMEOUT = float(os.environ.get("SHUTDOWN_TIMEOUT", "10"))


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def _read_cmdline(pid: int) -> str:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # First line is the header; the rest is the command.
            return " ".join(out.splitlines()[1:]).strip()
        except Exception:
            return ""

    try:
        out = subprocess.check_output(["ps", "-o", "pid=,command=", "-p", str(pid)], text=True)
        return " ".join(out.splitlines()).strip()
    except Exception:
        return ""


def _read_cwd(pid: int) -> str:
    if os.name == "nt":
        return ""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def _looks_like_service(
    cmdline: str,
    process_name: str,
    cmd_match: str,
    *,
    cwd: str = "",
    port: str | None = None,
) -> bool:
    if not cmdline:
        return False
    lowered = cmdline.lower()
    name_ok = not process_name or process_name.lower() in lowered
    cmd_ok = not cmd_match or cmd_match.lower() in lowered
    port_token = str(port).strip() if port else ""
    if port_token:
        port_lower = port_token.lower()
        port_ok = f"--port {port_lower}" in lowered or f"--port={port_lower}" in lowered
    else:
        port_ok = True
    cwd_ok = False
    if cwd:
        try:
            cwd_path = Path(cwd).resolve()
            cwd_ok = cwd_path == REPO_ROOT or REPO_ROOT in cwd_path.parents
        except Exception:
            cwd_ok = False
    if cwd or port_token:
        scope_ok = cwd_ok or (bool(port_token) and port_ok)
    else:
        scope_ok = True
    return name_ok and cmd_ok and port_ok and scope_ok


def _scan_processes(process_name: str, cmd_match: str, port: str | None) -> Iterable[Tuple[int, str]]:
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["wmic", "process", "get", "ProcessId,CommandLine"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return []
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        results = []
        for line in lines[1:]:
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            cmdline, pid_str = parts
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            if _looks_like_service(cmdline, process_name, cmd_match, port=port):
                results.append((pid, cmdline))
        return results

    try:
        out = subprocess.check_output(["ps", "-eo", "pid,command"], text=True)
    except Exception:
        return []

    results = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_str, cmdline = parts
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        if pid == os.getpid():
            continue
        cwd = _read_cwd(pid)
        if _looks_like_service(cmdline, process_name, cmd_match, cwd=cwd, port=port):
            results.append((pid, cmdline))
    return results


def _gather_targets(pid_file: Path, process_name: str, cmd_match: str, port: str | None) -> Dict[int, str]:
    targets: Dict[int, str] = {}

    file_pid = _read_pid(pid_file)
    if file_pid:
        if _pid_is_running(file_pid):
            cmdline = _read_cmdline(file_pid)
            cwd = _read_cwd(file_pid)
            if _looks_like_service(cmdline, process_name, cmd_match, cwd=cwd, port=port):
                targets[file_pid] = f"pid file ({pid_file})"
            else:
                print(
                    f"PID {file_pid} from {pid_file} is running but does not look like this service.\n"
                    f" Command line: {cmdline or 'unknown'}"
                )
                # Do not stop a suspicious PID automatically.
        else:
            print(f"PID {file_pid} from {pid_file} is not running. Removing stale pid file.")
            pid_file.unlink(missing_ok=True)

    for pid, cmdline in _scan_processes(process_name, cmd_match, port):
        targets.setdefault(pid, f"process search: {cmdline}")

    return targets


def _stop_pid(pid: int, timeout: float) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        print(f"Permission denied when trying to stop PID {pid}.")
        return False
    except OSError as exc:
        print(f"Error sending SIGTERM to PID {pid}: {exc}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.25)

    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.kill(pid, sigkill)
    except ProcessLookupError:
        return True
    except Exception as exc:
        print(f"Error sending SIGKILL to PID {pid}: {exc}")
        return False

    return not _pid_is_running(pid)


def shutdown_service(
    *,
    pid_file: Path = DEFAULT_PID_FILE,
    process_name: str = DEFAULT_PROCESS_NAME,
    cmd_match: str = DEFAULT_CMD_MATCH,
    port: str | None = DEFAULT_PROCESS_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    quiet: bool = False,
) -> bool:
    """
    Attempt to stop the running service. Returns True if the service is no longer running.
    """
    targets = _gather_targets(pid_file, process_name, cmd_match, port)
    if not targets:
        if not quiet:
            print("No running service processes found.")
        return True

    if not quiet:
        print(f"Found {len(targets)} target(s) to stop:")
        for pid, source in targets.items():
            print(f" - PID {pid} ({source})")

    success = True
    for pid, source in targets.items():
        if not quiet:
            print(f"Stopping PID {pid} ({source}) ...")
        if _stop_pid(pid, timeout):
            if not quiet:
                print(f"Stopped PID {pid}.")
            if pid_file.exists() and _read_pid(pid_file) == pid:
                pid_file.unlink(missing_ok=True)
        else:
            success = False
            print(f"PID {pid} may still be running; please verify manually.")

    # Double-check no matching processes remain.
    remaining = list(_scan_processes(process_name, cmd_match, port))
    if remaining:
        success = False
        if not quiet:
            print("Some matching processes are still running:")
            for pid, cmdline in remaining:
                print(f" - PID {pid}: {cmdline}")

    return success


def main() -> None:
    ok = shutdown_service()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
