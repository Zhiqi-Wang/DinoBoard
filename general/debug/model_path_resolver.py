from __future__ import annotations

from pathlib import Path


def resolve_default_model_path(project_dir: Path, game: str) -> Path | None:
    """Resolve best-effort default ONNX model path for a game.

    Priority:
    1) games/<game>/model/best_model.onnx
    2) games/<game>/model/model_best.onnx
    3) newest games/<game>/train/runs/**/artifacts/models/best_model.onnx
    4) newest games/<game>/train/out/**/artifacts/models/best_model.onnx
    5) newest games/<game>/train/runs/**/artifacts/models/candidate_model.onnx
    6) newest games/<game>/train/out/**/artifacts/models/candidate_model.onnx
    """
    stable_best = project_dir / "games" / game / "model" / "best_model.onnx"
    if stable_best.exists():
        return stable_best
    stable_legacy = project_dir / "games" / game / "model" / "model_best.onnx"
    if stable_legacy.exists():
        return stable_legacy

    train_root = project_dir / "games" / game / "train"
    runs_root = train_root / "runs"
    out_root = train_root / "out"

    def _pick_newest(root: Path, relative_pattern: str) -> Path | None:
        if not root.exists():
            return None
        hits = list(root.glob(relative_pattern))
        if not hits:
            return None
        return max(hits, key=lambda p: p.stat().st_mtime)

    for search_root, pattern in (
        (runs_root, "**/artifacts/models/best_model.onnx"),
        (out_root, "**/artifacts/models/best_model.onnx"),
        (runs_root, "**/artifacts/models/candidate_model.onnx"),
        (out_root, "**/artifacts/models/candidate_model.onnx"),
    ):
        resolved = _pick_newest(search_root, pattern)
        if resolved is not None:
            return resolved

    return None
