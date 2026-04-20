"""Desktop Briefcase launcher adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    # Briefcase copies sources into app package resources; this keeps imports stable.
    return here.parents[2]


def main() -> None:
    root = _project_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    os.chdir(root_str)

    from run_app import main as run_main

    run_main()
