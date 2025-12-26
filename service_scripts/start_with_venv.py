#!/usr/bin/env python3
"""
Start the FastAPI service using the local venv.

Usage:
  python tests/start_with_venv.py

Environment overrides:
  VENV_PATH       Path to venv directory (default: .venv at repo root)
  API_HOST        Host for uvicorn (default: 0.0.0.0)
  API_PORT        Port for uvicorn (default: 8000)
  UVICORN_RELOAD  Set to '1'/'true' to enable reload
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from venv_utils import DEFAULT_VENV_DIR, REPO_ROOT, venv_exists, venv_python


def _as_bool(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _ensure_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        return
    example_path = REPO_ROOT / ".env.local.example"
    if example_path.exists():
        shutil.copyfile(example_path, env_path)
        print(f"Created {env_path} from {example_path}")
        return
    sys.exit("Missing .env and .env.local.example. Create .env or add .env.local.example to continue.")


def _set_windows_symlink_defenses() -> None:
    if os.name != "nt":
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def main() -> None:
    env_name = os.environ.get("VENV_PATH", DEFAULT_VENV_DIR)
    api_host = os.environ.get("API_HOST", "0.0.0.0")
    api_port = os.environ.get("API_PORT", "18000")
    reload_flag = _as_bool(os.environ.get("UVICORN_RELOAD"))

    _ensure_env_file()
    _set_windows_symlink_defenses()

    venv_dir = Path(env_name)
    if not venv_exists(venv_dir):
        sys.exit(f"Venv not found at {venv_dir}. Run tests/create_venv.py first.")

    venv_py = venv_python(venv_dir)
    cmd = [
        str(venv_py),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        api_host,
        "--port",
        str(api_port),
    ]
    if reload_flag:
        cmd.append("--reload")

    print("Starting service with:\n ", " ".join(cmd))
    try:
        subprocess.check_call(cmd, cwd=REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (-signal.SIGTERM, -signal.SIGINT):
            print("Service stopped.")
            return
        raise


if __name__ == "__main__":
    main()
