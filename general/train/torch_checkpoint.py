from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple


def try_load_checkpoint(*, checkpoint_path: str | None, net: Any, optimizer: Any, torch_module: Any) -> Tuple[bool, str]:
    if not checkpoint_path:
        return False, ""
    path = Path(checkpoint_path)
    if not path.exists():
        return False, ""
    try:
        ckpt = torch_module.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only arg.
        ckpt = torch_module.load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        return False, f"invalid checkpoint format: {path}"
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        return False, f"invalid checkpoint missing model_state: {path}"
    try:
        net.load_state_dict(model_state, strict=True)
    except Exception as e:
        return False, f"failed to load model_state from checkpoint {path}: {e}"
    opt_state = ckpt.get("optimizer_state")
    if opt_state is not None and not isinstance(opt_state, dict):
        return False, f"invalid checkpoint optimizer_state type: {path}"
    if isinstance(opt_state, dict):
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            return False, f"failed to load optimizer_state from checkpoint {path}: {e}"
    return True, str(path)


def save_checkpoint(
    *,
    checkpoint_path: Path,
    net: Any,
    optimizer: Any,
    torch_module: Any,
    extra: dict[str, Any] | None = None,
) -> str:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if isinstance(extra, dict) and extra:
        payload.update(extra)
    torch_module.save(payload, str(checkpoint_path))
    return str(checkpoint_path)
