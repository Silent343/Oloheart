"""Run OloHeart from a source checkout and prefer its isolated environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def _inside_project_environment() -> bool:
    return Path(sys.executable).resolve() == VENV_PYTHON.resolve()


if VENV_PYTHON.exists() and not _inside_project_environment():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oloheart.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

