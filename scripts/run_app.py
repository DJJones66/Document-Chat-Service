"""Start the FastAPI app using the local .venv."""

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_ops import (  # noqa: E402
    PID_FILE,
    clear_pid,
    ensure_venv,
    pid_is_running,
    read_pid,
    start_background,
    venv_python,
    write_pid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastAPI service from .venv.")
    parser.add_argument("--host", default=os.getenv("API_HOST", "0.0.0.0"))
    parser.add_argument("--port", default=os.getenv("API_PORT", "8000"))
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev only).")
    parser.add_argument(
        "--python",
        help="Path to python used to create the venv if it does not exist yet.",
    )
    args = parser.parse_args()

    ensure_venv(args.python)
    existing_pid = read_pid()
    if existing_pid and pid_is_running(existing_pid):
        raise SystemExit(f"Service already running with PID {existing_pid} (pid file {PID_FILE})")
    clear_pid()

    py = venv_python()
    cmd = [
        str(py),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    proc = start_background(cmd)
    write_pid(proc.pid)
    print(f"Service started with PID {proc.pid} (pid file {PID_FILE})")
    print("Stop with: python scripts/shutdown_app.py")


if __name__ == "__main__":
    main()
