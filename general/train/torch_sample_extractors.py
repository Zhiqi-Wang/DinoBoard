from __future__ import annotations

from typing import Any

from .policy_target_utils import normalize_sparse_policy

SparsePolicyTrainRow = tuple[list[float], list[int], list[float], float, float, int]


def extract_sparse_policy_train_rows(rows: list[dict[str, Any]]) -> list[SparsePolicyTrainRow]:
    """Extract rows with dense features + sparse policy targets + scalar value labels."""
    out: list[SparsePolicyTrainRow] = []
    for row in rows:
        feats = row.get("features")
        act = row.get("action_id")
        if not isinstance(feats, list) or len(feats) <= 0 or act is None:
            continue
        action = int(act)
        ids, probs = normalize_sparse_policy(
            row.get("policy_action_ids"),
            row.get("policy_probs"),
            row.get("policy_action_visits"),
            fallback_action=action,
        )
        phase = float(row.get("phase", 0.0))
        phase = max(0.0, min(1.0, phase))
        out.append(([float(v) for v in feats], ids, probs, float(row.get("z", 0.0)), phase, action))
    return out

