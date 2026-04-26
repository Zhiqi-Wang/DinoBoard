from __future__ import annotations

from pathlib import Path
from typing import Any

from games.azul.train.constants import AZUL_INPUT_DIM, AZUL_POLICY_DIM
from general.train.torch_pvnet import (
    export_initial_policy_onnx_from_config,
)
from general.train.torch_sample_extractors import SparsePolicyTrainRow, extract_sparse_policy_train_rows
from general.train.torch_sparse_trainer import SparseTrainRow, run_sparse_policy_value_train

_RUNTIME_REPLAY_CACHE: dict[
    str,
    list[SparseTrainRow],
] = {}


def export_initial_policy_onnx(
    *,
    config,
    output_path: Path,
    input_dim: int = AZUL_INPUT_DIM,
    policy_dim: int = AZUL_POLICY_DIM,
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
    def _extract_rows(rows: list[dict[str, Any]]) -> list[SparseTrainRow]:
        base_rows: list[SparsePolicyTrainRow] = extract_sparse_policy_train_rows(rows)
        return [(f, ids, probs, z, phase, action, {}) for f, ids, probs, z, phase, action in base_rows]

    return run_sparse_policy_value_train(
        config=config,
        artifacts_dir=artifacts_dir,
        runtime_replay_cache=_RUNTIME_REPLAY_CACHE,
        extract_rows=_extract_rows,
        policy_dim=AZUL_POLICY_DIM,
        empty_reason="no_training_samples_with_features",
        resume_checkpoint_path=resume_checkpoint_path,
        step_index=step_index,
        total_steps=total_steps,
        incremental_samples=incremental_samples,
        min_hidden=32,
    )
