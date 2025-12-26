#!/usr/bin/env python3
"""
Helpers for managing a local venv (cross-platform).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV_DIR = REPO_ROOT / os.environ.get("VENV_PATH", ".venv")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3.11")


def _is_windows_store_stub(path: str) -> bool:
    if os.name != "nt":
        return False
    return "windowsapps" in path.lower()


def _resolve_candidate(candidate: str) -> str | None:
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if candidate_path.is_file():
        return str(candidate_path)
    return shutil.which(str(candidate))


def find_python() -> str:
    """
    Resolve the Python executable to use for creating the venv.
    Falls back to python3/python if python3.11 is not found.
    Only accepts Python 3.11-3.13.
    """
    candidates = [
        sys.executable,
        PYTHON_BIN,
        "python3.11",
        "python3",
        "python",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        resolved = _resolve_candidate(str(candidate))
        if not resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_windows_store_stub(resolved):
            continue
        if _is_supported_python(resolved):
            return resolved
    sys.exit("No suitable Python 3.11-3.13 executable found. Set PYTHON_BIN to a supported Python path.")


def _is_supported_python(python_cmd: str) -> bool:
    try:
        out = subprocess.check_output(
            [python_cmd, "-c", "import sys;print(sys.version_info.major);print(sys.version_info.minor)"],
            text=True,
        ).strip().splitlines()
        major, minor = int(out[0]), int(out[1])
        return major == 3 and 11 <= minor < 14
    except Exception:
        return False


def venv_python(venv_dir: Path = DEFAULT_VENV_DIR) -> Path:
    """
    Return the python executable inside the venv.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_exists(venv_dir: Path = DEFAULT_VENV_DIR) -> bool:
    """
    Check whether the venv python exists.
    """
    return venv_python(venv_dir).exists()
