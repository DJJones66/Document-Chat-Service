import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
RUN_DIR = ROOT / ".run"
PID_FILE = RUN_DIR / "uvicorn.pid"


def _bin_dir() -> Path:
    return VENV_DIR / ("Scripts" if os.name == "nt" else "bin")


def venv_python() -> Path:
    return _bin_dir() / ("python.exe" if os.name == "nt" else "python")


def _check_version(python_cmd: str) -> None:
    version_info = subprocess.check_output(
        [python_cmd, "-c", "import sys;print(sys.version_info.major);print(sys.version_info.minor)"],
        text=True,
    ).strip().splitlines()
    major, minor = int(version_info[0]), int(version_info[1])
    if major != 3 or minor < 11 or minor >= 14:
        raise SystemExit(f"Python 3.11-3.13 required; got {major}.{minor} at {python_cmd}")


def create_venv(python_cmd: Optional[str] = None, wipe: bool = False) -> None:
    python_cmd = python_cmd or sys.executable
    _check_version(python_cmd)
    if VENV_DIR.exists():
        if wipe:
            print(f"Removing existing venv at {VENV_DIR}")
            shutil.rmtree(VENV_DIR)
        else:
            print(f"venv already exists at {VENV_DIR}")
            return
    print(f"Creating venv with {python_cmd} ...")
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([python_cmd, "-m", "venv", str(VENV_DIR)])


def ensure_venv(python_cmd: Optional[str] = None, wipe: bool = False) -> None:
    if wipe or not VENV_DIR.exists():
        create_venv(python_cmd, wipe=wipe)
    py = venv_python()
    if not py.exists():
        raise SystemExit(f"Expected venv python at {py}, but it is missing.")
    _check_version(str(py))


def run_cmd(cmd: Iterable[str], label: str) -> None:
    print(label)
    subprocess.check_call(list(cmd), cwd=ROOT)


def start_background(cmd: Iterable[str]) -> subprocess.Popen:
    kwargs = {"cwd": ROOT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(cmd), **kwargs)


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def read_pid() -> Optional[int]:
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            return None
    return None


def write_pid(pid: int) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))


def clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def stop_pid(pid: int, timeout: float = 10.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_is_running(pid):
            return True
        time.sleep(0.25)
    return not pid_is_running(pid)
