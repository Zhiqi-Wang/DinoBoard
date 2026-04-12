from __future__ import annotations

from pathlib import Path
from typing import Any

from general.train.torch_pvnet import export_initial_policy_onnx_from_config
from general.train.torch_sample_extractors import SparsePolicyTrainRow, extract_sparse_policy_train_rows
from general.train.torch_sparse_trainer import SparseTrainRow, run_sparse_policy_value_train

from .constants import QUORIDOR_INPUT_DIM, QUORIDOR_POLICY_DIM

_RUNTIME_REPLAY_CACHE: dict[str, list[SparseTrainRow]] = {}


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
        out: list[SparseTrainRow] = []
        base_rows: list[SparsePolicyTrainRow] = extract_sparse_policy_train_rows(rows)
        for feats, ids, probs, z, phase, action in base_rows:
            if len(feats) < QUORIDOR_INPUT_DIM:
                continue
            clipped_ids: list[int] = []
            clipped_probs: list[float] = []
            for aid, prob in zip(ids, probs):
                if 0 <= int(aid) < QUORIDOR_POLICY_DIM:
                    clipped_ids.append(int(aid))
                    clipped_probs.append(float(prob))
            if action < 0 or action >= QUORIDOR_POLICY_DIM:
                continue
            out.append(
                (
                    [float(v) for v in feats[:QUORIDOR_INPUT_DIM]],
                    clipped_ids,
                    clipped_probs,
                    float(z),
                    float(phase),
                    int(action),
                    {},
                )
            )
        return out

    return run_sparse_policy_value_train(
        config=config,
        artifacts_dir=artifacts_dir,
        runtime_replay_cache=_RUNTIME_REPLAY_CACHE,
        extract_rows=_extract_rows,
        policy_dim=QUORIDOR_POLICY_DIM,
        empty_reason="no_training_samples_with_features",
        resume_checkpoint_path=resume_checkpoint_path,
        step_index=step_index,
        total_steps=total_steps,
        incremental_samples=incremental_samples,
        min_hidden=32,
    )


def export_initial_policy_onnx(config, output_path: Path, *, seed: int = 0) -> str:
    return export_initial_policy_onnx_from_config(
        config=config,
        output_path=output_path,
        input_dim=QUORIDOR_INPUT_DIM,
        policy_dim=QUORIDOR_POLICY_DIM,
        seed=seed,
    )

