from __future__ import annotations

from pathlib import Path
from typing import Any

from general.train.policy_target_utils import normalize_sparse_policy
from general.train.torch_pvnet import (
    export_initial_policy_onnx_from_config,
)
from general.train.torch_sparse_trainer import SparseTrainRow, run_sparse_policy_value_train

from .constants import DEFAULT_RETURN_PHASE_FACTOR, SPLENDOR_INPUT_DIM, SPLENDOR_POLICY_DIM

_RUNTIME_REPLAY_CACHE: dict[str, list[SparseTrainRow]] = {}


def export_initial_policy_onnx(
    *,
    config,
    output_path: Path,
    input_dim: int = SPLENDOR_INPUT_DIM,
    policy_dim: int = SPLENDOR_POLICY_DIM,
    seed: int = 20260323,
) -> str:
    """Export a randomly initialized dual-head policy-value ONNX for self-play bootstrap."""
    return export_initial_policy_onnx_from_config(
        config=config,
        output_path=output_path,
        input_dim=input_dim,
        policy_dim=policy_dim,
        seed=seed,
    )


def run_torch_train(
    config,
    artifacts_dir: Path,
    *,
    resume_checkpoint_path: str | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
    incremental_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def _extract_train_rows(rows: list[dict[str, Any]]) -> list[SparseTrainRow]:
        out: list[SparseTrainRow] = []
        for row in rows:
            feats = row.get("features")
            act = row.get("action_id")
            if not isinstance(feats, list) or len(feats) < SPLENDOR_INPUT_DIM or act is None:
                continue
            action = int(act)
            if action < 0 or action >= SPLENDOR_POLICY_DIM:
                continue
            ids, probs = normalize_sparse_policy(
                row.get("policy_action_ids"),
                row.get("policy_probs"),
                row.get("policy_action_visits"),
                fallback_action=action,
            )
            phase = max(0.0, min(1.0, float(row.get("phase", 0.0))))
            return_phase = 1.0 if float(row.get("return_phase", 0.0)) > 0.5 else 0.0
            choose_noble_phase = 1.0 if float(row.get("choose_noble_phase", 0.0)) > 0.5 else 0.0
            out.append(
                (
                    [float(v) for v in feats[:SPLENDOR_INPUT_DIM]],
                    ids,
                    probs,
                    float(row.get("z", 0.0)),
                    phase,
                    action,
                    {
                        "return_phase": return_phase,
                        "choose_noble_phase": choose_noble_phase,
                    },
                )
            )
        return out

    return_phase_factor = float(getattr(config.trainer, "return_phase_factor", DEFAULT_RETURN_PHASE_FACTOR))
    return_phase_factor = max(0.0, min(1.0, return_phase_factor))

    def _weight_builder(phase_i, extras, value_late_weight: float, torch):
        return_phase_i = extras.get("return_phase")
        choose_noble_phase_i = extras.get("choose_noble_phase")
        if return_phase_i is None:
            return_phase_i = torch.zeros_like(phase_i)
        if choose_noble_phase_i is None:
            choose_noble_phase_i = torch.zeros_like(phase_i)
        w_i = 1.0 + value_late_weight * phase_i
        w_i = w_i * (1.0 - return_phase_i + return_phase_i * return_phase_factor)
        metrics = {
            "return_phase_sample_ratio": float(return_phase_i.mean().item()),
            "choose_noble_phase_sample_ratio": float(choose_noble_phase_i.mean().item()),
            "subphase_sample_ratio": float(torch.maximum(return_phase_i, choose_noble_phase_i).mean().item()),
            "mean_value_weight": float(w_i.mean().item()),
        }
        return w_i, metrics

    return run_sparse_policy_value_train(
        config=config,
        artifacts_dir=artifacts_dir,
        runtime_replay_cache=_RUNTIME_REPLAY_CACHE,
        extract_rows=_extract_train_rows,
        policy_dim=SPLENDOR_POLICY_DIM,
        empty_reason="no_training_samples_with_features",
        resume_checkpoint_path=resume_checkpoint_path,
        step_index=step_index,
        total_steps=total_steps,
        incremental_samples=incremental_samples,
        min_hidden=32,
        weight_builder=_weight_builder,
        fixed_metrics={"return_phase_factor": return_phase_factor},
    )

