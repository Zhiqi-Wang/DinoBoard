from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_config_hash(payload: dict[str, Any]) -> str:
    dumped = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
