"""Create a local virtual environment at .venv (Python 3.11-3.13)."""

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_ops import ROOT, VENV_DIR, create_venv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a .venv for the service.")
    parser.add_argument(
        "--python",
        help="Path to the python executable to use (defaults to the current interpreter).",
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Delete any existing .venv before creating a new one.",
    )
    args = parser.parse_args()

    create_venv(args.python, wipe=args.wipe)
    print(f"Venv ready at {VENV_DIR}")
    print(f"Activate with: {VENV_DIR / ('Scripts/activate' if os.name == 'nt' else 'bin/activate')}")
    print(f"Project root: {ROOT}")


if __name__ == "__main__":
    main()
