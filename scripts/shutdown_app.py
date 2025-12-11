"""Stop the FastAPI app started via run_app.py."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_ops import PID_FILE, clear_pid, pid_is_running, read_pid, stop_pid  # noqa: E402


def main() -> None:
    pid = read_pid()
    if not pid:
        print(f"No pid file found at {PID_FILE}. Nothing to stop.")
        return
    if not pid_is_running(pid):
        print(f"PID {pid} is not running. Removing stale pid file.")
        clear_pid()
        return
    if stop_pid(pid):
        print(f"Stopped service with PID {pid}.")
        clear_pid()
    else:
        print(f"Attempted to stop PID {pid} but it may still be running. Check manually.")


if __name__ == "__main__":
    main()
