#!/usr/bin/env python3
"""
Create a local Python virtual environment for the Document Chat Service.

Usage:
  python tests/create_venv.py
  python tests/create_venv.py --wipe

Environment overrides:
  VENV_PATH   Path to venv directory (default: .venv at repo root)
  PYTHON_BIN  Python executable to use (default: python3.11, falls back to python3/python; must be 3.11-3.13)
  VENV_FORCE_RECREATE Set to 1/true to delete and recreate the venv
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from venv_utils import DEFAULT_VENV_DIR, REPO_ROOT, find_python, venv_exists, venv_python


def main() -> None:
    venv_dir = Path(os.environ.get("VENV_PATH", DEFAULT_VENV_DIR))
    force_recreate = (
        os.environ.get("VENV_FORCE_RECREATE", "").lower() in {"1", "true", "yes", "on"}
        or "--wipe" in sys.argv
    )

    if venv_exists(venv_dir):
        if force_recreate:
            print(f"Removing existing venv at {venv_dir}")
            shutil.rmtree(venv_dir)
        else:
            print(f"Venv already exists at {venv_dir}. Nothing to do. Set VENV_FORCE_RECREATE=1 to rebuild.")
            return

    python_bin = find_python()
    print(f"Creating venv at {venv_dir} with {python_bin}")
    subprocess.check_call([python_bin, "-m", "venv", str(venv_dir)], cwd=REPO_ROOT)

    # Upgrade core tooling inside the venv for smoother installs.
    venv_py = venv_python(venv_dir)
    subprocess.check_call([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=REPO_ROOT)

    print(f"Done. Activate with: source {venv_dir}/bin/activate (Linux/macOS) or {venv_dir}\\Scripts\\activate (Windows)")


if __name__ == "__main__":
    main()
