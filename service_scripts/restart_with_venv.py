#!/usr/bin/env python3
"""
Gracefully restart the FastAPI service using the local venv. Useful after updating .env or settings.

Usage:
  python service_scripts/restart_with_venv.py

Environment overrides:
  PID_FILE           PID file to check before searching (default: .run/uvicorn.pid)
  PROCESS_NAME       Process name to match when PID file is absent (default: uvicorn)
  PROCESS_CMD_MATCH  Command-line substring to validate targets (default: app.main:app)
  SHUTDOWN_TIMEOUT   Seconds to wait after SIGTERM before force killing (default: 10)
  RESTART_DELAY      Seconds to wait after shutdown before starting again (default: 1.0)

  VENV_PATH, API_HOST, API_PORT, UVICORN_RELOAD behave the same as in start_with_venv.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import start_with_venv
from shutdown_with_venv import (
    DEFAULT_CMD_MATCH,
    DEFAULT_PID_FILE,
    DEFAULT_PROCESS_NAME,
    DEFAULT_TIMEOUT,
    shutdown_service,
)
from venv_utils import DEFAULT_VENV_DIR, REPO_ROOT, venv_exists


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def main() -> None:
    _load_env_file(REPO_ROOT / ".env")
    pid_file = Path(os.environ.get("PID_FILE", DEFAULT_PID_FILE))
    process_name = os.environ.get("PROCESS_NAME", DEFAULT_PROCESS_NAME)
    cmd_match = os.environ.get("PROCESS_CMD_MATCH", DEFAULT_CMD_MATCH)
    process_port = os.environ.get("PROCESS_PORT") or os.environ.get("API_PORT")
    timeout = float(os.environ.get("SHUTDOWN_TIMEOUT", str(DEFAULT_TIMEOUT)))
    restart_delay = float(os.environ.get("RESTART_DELAY", "1.0"))
    venv_dir = Path(os.environ.get("VENV_PATH", DEFAULT_VENV_DIR))

    if not venv_exists(venv_dir):
        sys.exit(f"Venv not found at {venv_dir}. Run service_scripts/create_venv.py first.")

    print("Stopping running service (if any)...")
    stopped = shutdown_service(
        pid_file=pid_file,
        process_name=process_name,
        cmd_match=cmd_match,
        port=process_port,
        timeout=timeout,
        quiet=False,
    )
    if not stopped:
        sys.exit("Unable to stop running service cleanly. Aborting restart.")

    if restart_delay > 0:
        time.sleep(restart_delay)

    print("Starting service...")
    start_with_venv.main()


if __name__ == "__main__":
    main()
