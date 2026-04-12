from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Hashable

from .torch_checkpoint import try_load_checkpoint

_TRAINER_RUNTIME_CACHE: dict[str, dict[str, Any]] = {}
_MAX_RUNTIME_CACHE_ENTRIES = 4


def _checkpoint_descriptor(checkpoint_path: str | None) -> tuple[str, int, int] | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def _evict_if_needed() -> None:
    while len(_TRAINER_RUNTIME_CACHE) > _MAX_RUNTIME_CACHE_ENTRIES:
        oldest_key = next(iter(_TRAINER_RUNTIME_CACHE))
        _TRAINER_RUNTIME_CACHE.pop(oldest_key, None)


def get_or_create_torch_runtime(
    *,
    cache_key: str,
    runtime_signature: Hashable,
    build_net: Callable[[], Any],
    build_optimizer: Callable[[Any], Any],
    resume_checkpoint_path: str | None,
    torch_module: Any,
) -> tuple[Any, Any, bool, str]:
    ckpt_desc = _checkpoint_descriptor(resume_checkpoint_path)
    entry = _TRAINER_RUNTIME_CACHE.get(cache_key)
    if entry is not None and entry.get("signature") == runtime_signature and entry.get("checkpoint_desc") == ckpt_desc:
        # Refresh insertion order for simple LRU behavior.
        _TRAINER_RUNTIME_CACHE.pop(cache_key, None)
        _TRAINER_RUNTIME_CACHE[cache_key] = entry
        return entry["net"], entry["optimizer"], False, ""

    net = build_net()
    optimizer = build_optimizer(net)
    resumed, resumed_from = try_load_checkpoint(
        checkpoint_path=resume_checkpoint_path,
        net=net,
        optimizer=optimizer,
        torch_module=torch_module,
    )
    if resume_checkpoint_path and not resumed and resumed_from:
        raise RuntimeError(resumed_from)
    _TRAINER_RUNTIME_CACHE[cache_key] = {
        "signature": runtime_signature,
        "checkpoint_desc": ckpt_desc,
        "net": net,
        "optimizer": optimizer,
    }
    _evict_if_needed()
    return net, optimizer, resumed, resumed_from
