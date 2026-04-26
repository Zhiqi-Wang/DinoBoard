from __future__ import annotations

from typing import Any


def normalize_sparse_policy(
    policy_action_ids: Any,
    policy_probs: Any,
    policy_action_visits: Any = None,
    *,
    fallback_action: int | None = None,
) -> tuple[list[int], list[float]]:
    """Normalize sparse policy target lists and optionally fallback to one-hot."""
    ids: list[int] = []
    probs: list[float] = []
    if isinstance(policy_action_ids, list) and isinstance(policy_probs, list):
        if len(policy_action_ids) == len(policy_probs) and len(policy_action_ids) > 0:
            for aid, p in zip(policy_action_ids, policy_probs):
                try:
                    ids.append(int(aid))
                    probs.append(float(p))
                except (TypeError, ValueError):
                    continue
            if ids and len(ids) == len(probs):
                total = float(sum(max(0.0, v) for v in probs))
                if total > 1e-12:
                    return ids, [float(max(0.0, v) / total) for v in probs]
    if isinstance(policy_action_ids, list) and isinstance(policy_action_visits, list):
        if len(policy_action_ids) == len(policy_action_visits) and len(policy_action_ids) > 0:
            ids = []
            visits: list[float] = []
            for aid, v in zip(policy_action_ids, policy_action_visits):
                try:
                    ids.append(int(aid))
                    visits.append(float(v))
                except (TypeError, ValueError):
                    continue
            if ids and len(ids) == len(visits):
                total = float(sum(max(0.0, v) for v in visits))
                if total > 1e-12:
                    return ids, [float(max(0.0, v) / total) for v in visits]
    if fallback_action is not None:
        return [int(fallback_action)], [1.0]
    return [], []

