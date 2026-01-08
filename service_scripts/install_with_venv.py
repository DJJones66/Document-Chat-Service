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
  AUTO_CREATE_VENV Set to 0/false to disable auto-creating the venv when missing
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from venv_utils import DEFAULT_VENV_DIR, REPO_ROOT, venv_exists, venv_python

DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_MODEL_KEYS = (
    "OLLAMA_EMBEDDING_MODEL",
    "OLLAMA_LLM_MODEL",
    "OLLAMA_CONTEXTUAL_LLM_MODEL",
)


def _parse_env_file(path: Path) -> dict:
    if not path.exists():
        return {}
    values: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def _resolve_ollama_models() -> list[str]:
    env_values: dict = {}
    for candidate in (REPO_ROOT / ".env", REPO_ROOT / ".env.local.example", REPO_ROOT / ".env.example"):
        env_values.update(_parse_env_file(candidate))
    models: list[str] = []
    for key in OLLAMA_MODEL_KEYS:
        value = os.environ.get(key) or env_values.get(key)
        if not value and key == "OLLAMA_EMBEDDING_MODEL":
            value = DEFAULT_OLLAMA_EMBED_MODEL
        if not value:
            continue
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "null", "false", "disabled"}:
            continue
        models.append(normalized)
    # Deduplicate while preserving order
    seen = set()
    ordered: list[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def _maybe_pull_ollama_models() -> None:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return
    for model_name in _resolve_ollama_models():
        show_result = subprocess.run(
            [ollama_bin, "show", model_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if show_result.returncode == 0:
            continue
        try:
            print(f"Ollama model missing; pulling {model_name}...")
            subprocess.check_call([ollama_bin, "pull", model_name], cwd=REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            print(f"Warning: Ollama model pull failed (will retry at runtime): {exc}")


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
        auto_create = _as_bool(os.environ.get("AUTO_CREATE_VENV"), default=True)
        if auto_create:
            print(f"Venv not found at {venv_dir}; creating it now.")
            create_script = REPO_ROOT / "service_scripts" / "create_venv.py"
            subprocess.check_call([sys.executable, str(create_script)], cwd=REPO_ROOT)
        if not venv_exists(venv_dir):
            sys.exit(f"Venv not found at {venv_dir}. Run service_scripts/create_venv.py first.")

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

    _maybe_pull_ollama_models()

    print("\nDependencies installed into the venv.")


if __name__ == "__main__":
    main()
