from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_candidate(artifacts_dir: Path, payload: dict[str, Any]) -> Path:
    out = artifacts_dir / "candidate_manifest.json"
    write_json(out, payload)
    return out


def promote_best_if_accepted(artifacts_dir: Path, candidate_manifest: dict[str, Any], accepted: bool) -> Path | None:
    if not accepted:
        return None
    best_payload = {
        "promoted_at": utc_now(),
        "source": "candidate_manifest",
        "candidate": candidate_manifest,
    }
    out = artifacts_dir / "best_manifest.json"
    write_json(out, best_payload)
    return out

