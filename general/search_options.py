from __future__ import annotations

from copy import deepcopy
from typing import Any


def clone_search_options(search_options: dict[str, Any] | None) -> dict[str, Any]:
    """Return a detached dict for passthrough-only search options."""
    if not isinstance(search_options, dict) or not search_options:
        return {}
    return deepcopy(search_options)
