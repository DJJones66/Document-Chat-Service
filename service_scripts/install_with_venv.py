#!/usr/bin/env python3
"""
Install project dependencies into the local venv for the Document Chat Service.

Usage:
  python tests/install_with_venv.py             # editable install from pyproject
  python tests/install_with_venv.py --full      # install pinned requirements.txt (torch/docling/etc.)

Environment overrides:
  VENV_PATH        Path to venv directory (default: .venv at repo root)
  FULL_INSTALL     Set to 1/true to force requirements.txt install
  UPGRADE_PIP      Set to 0/false to skip pip upgrade
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from venv_utils import DEFAULT_VENV_DIR, REPO_ROOT, venv_exists, venv_python


def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _run_pip(cmd: list[str], *, retry_on_windows: bool = False) -> None:
    env = os.environ.copy()
    env["PIP_NO_INPUT"] = "1"
    try:
        subprocess.check_call(cmd, cwd=REPO_ROOT, env=env)
        return
    except subprocess.CalledProcessError:
        if os.name != "nt" or not retry_on_windows:
            raise
    time.sleep(2)
    retry_cmd = cmd + ["--no-cache-dir"]
    subprocess.check_call(retry_cmd, cwd=REPO_ROOT, env=env)


def _pyproject_disallows_editable() -> bool:
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return False
    try:
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("package-mode") and "false" in stripped:
                return True
    except OSError:
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Install dependencies into the local venv.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use requirements.txt (includes heavy deps like torch/docling/easyocr).",
    )
    parser.add_argument(
        "--no-upgrade-pip",
        action="store_true",
        help="Skip pip upgrade step.",
    )
    args = parser.parse_args()

    venv_dir = Path(os.environ.get("VENV_PATH", DEFAULT_VENV_DIR))
    if not venv_exists(venv_dir):
        sys.exit(f"Venv not found at {venv_dir}. Run tests/create_venv.py first.")

    full_install = args.full or _as_bool(os.environ.get("FULL_INSTALL"))
    if not full_install and _pyproject_disallows_editable():
        print("pyproject.toml uses package-mode = false; falling back to requirements.txt install.")
        full_install = True
    upgrade_pip = not args.no_upgrade_pip and _as_bool(os.environ.get("UPGRADE_PIP"), default=True)

    venv_py = venv_python(venv_dir)

    if upgrade_pip:
        _run_pip([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"], retry_on_windows=True)

    if full_install:
        requirements = REPO_ROOT / "requirements.txt"
        if not requirements.exists():
            sys.exit(f"requirements.txt not found at {requirements}")
        # Use PyTorch CPU wheels index to avoid pulling CUDA dependencies by default.
        torch_index = "https://download.pytorch.org/whl/cpu"
        cmd = [
            str(venv_py),
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
            "--extra-index-url",
            torch_index,
        ]
        print("Installing pinned requirements.txt (with CPU torch index)...")
        _run_pip(cmd, retry_on_windows=True)
    else:
        cmd = [str(venv_py), "-m", "pip", "install", "-e", "."]
        print("Installing editable project dependencies (pyproject)...")
        _run_pip(cmd, retry_on_windows=True)

    print("\nDependencies installed into the venv.")


if __name__ == "__main__":
    main()
