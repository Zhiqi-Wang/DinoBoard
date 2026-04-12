from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_root(current_file: str, *, levels_up: int = 3) -> Path:
    """Ensure project root is available on sys.path for script-style entrypoints."""
    resolved_file = Path(current_file).resolve()
    project_root = resolved_file.parents[levels_up]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root

