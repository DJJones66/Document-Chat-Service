"""Install dependencies into the .venv."""

import argparse
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from env_ops import ensure_venv, run_cmd, venv_python  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Install project dependencies into .venv.")
    parser.add_argument(
        "--python",
        help="Path to python used to create the venv if it does not exist yet.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install the full pinned set from requirements.txt (torch/docling/easyocr).",
    )
    args = parser.parse_args()

    ensure_venv(args.python)
    py = venv_python()
    run_cmd([str(py), "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip...")
    if args.full:
        run_cmd([str(py), "-m", "pip", "install", "-r", "requirements.txt"], "Installing full requirements.txt ...")
    else:
        run_cmd([str(py), "-m", "pip", "install", "-e", "."], "Installing editable project deps ...")
    print("Install complete.")


if __name__ == "__main__":
    main()
