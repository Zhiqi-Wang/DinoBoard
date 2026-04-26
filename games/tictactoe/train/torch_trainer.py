from __future__ import annotations

from pathlib import Path
from typing import Any

from general.train.torch_pvnet import export_initial_policy_onnx_from_config
from general.train.torch_simple_trainer import SimpleTrainRow, run_simple_policy_value_train

_INPUT_DIM = 27
_POLICY_DIM = 9
_RUNTIME_REPLAY_CACHE: dict[str, list[SimpleTrainRow]] = {}


def _to_features(board: list[int], player: int) -> list[float]:
    me = float(player)
    op = float(1 - player)
    feat: list[float] = []
    for c in board:
        v = float(c)
        feat.append(1.0 if v == me else 0.0)
        feat.append(1.0 if v == op else 0.0)
        feat.append(1.0 if v < 0 else 0.0)
    return feat


def run_torch_train(
    config,
    artifacts_dir: Path,
    *,
    resume_checkpoint_path: str | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
    incremental_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def _extract_train_rows(rows: list[dict[str, Any]]) -> list[SimpleTrainRow]:
        out: list[SimpleTrainRow] = []
        for row in rows:
            board = row.get("board")
            action_id = row.get("action_id")
            if not isinstance(board, list) or len(board) != 9 or action_id is None:
                continue
            out.append(
                (
                    _to_features([int(v) for v in board], int(row.get("player", 0))),
                    int(action_id),
                    float(row.get("z", 0.0)),
                )
            )
        return out

    return run_simple_policy_value_train(
        config=config,
        artifacts_dir=artifacts_dir,
        runtime_replay_cache=_RUNTIME_REPLAY_CACHE,
        extract_rows=_extract_train_rows,
        input_dim=_INPUT_DIM,
        policy_dim=_POLICY_DIM,
        empty_reason="no_training_samples_with_board",
        resume_checkpoint_path=resume_checkpoint_path,
        step_index=step_index,
        total_steps=total_steps,
        incremental_samples=incremental_samples,
    )


def export_initial_policy_onnx(config, output_path: Path, *, seed: int = 0) -> str:
    return export_initial_policy_onnx_from_config(
        config=config,
        output_path=output_path,
        input_dim=_INPUT_DIM,
        policy_dim=_POLICY_DIM,
        seed=seed,
    )

