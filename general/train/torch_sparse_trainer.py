from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .torch_checkpoint import save_checkpoint
from .torch_pvnet import create_pvnet, export_pvnet_onnx, mean_last
from .torch_runtime import get_or_create_torch_runtime

SparseTrainRow = tuple[list[float], list[int], list[float], float, float, int, dict[str, float]]
WeightBuilder = Callable[[Any, dict[str, Any], float, Any], tuple[Any, dict[str, float] | None]]


def _default_weight_builder(phase_i, _extras, value_late_weight: float, _torch):
    return 1.0 + value_late_weight * phase_i, None


def run_sparse_policy_value_train(
    *,
    config: Any,
    artifacts_dir: Path,
    runtime_replay_cache: dict[str, list[SparseTrainRow]],
    extract_rows: Callable[[list[dict[str, Any]]], list[SparseTrainRow]],
    policy_dim: int,
    empty_reason: str,
    resume_checkpoint_path: str | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
    incremental_samples: list[dict[str, Any]] | None = None,
    min_hidden: int = 32,
    weight_builder: WeightBuilder | None = None,
    fixed_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as e:  # pragma: no cover - runtime env dependent
        return {
            "status": "skipped",
            "framework": "torch",
            "reason": "torch_not_installed",
            "error": str(e),
        }

    weight_builder_fn = weight_builder or _default_weight_builder
    buffer_size = max(1, int(getattr(config.trainer, "buffer_size", 100000)))
    cache_key = str(artifacts_dir.resolve())
    replay_buffer = runtime_replay_cache.setdefault(cache_key, [])

    if incremental_samples is not None:
        replay_buffer.extend(extract_rows(incremental_samples))
    if len(replay_buffer) > buffer_size:
        del replay_buffer[: len(replay_buffer) - buffer_size]
    if not replay_buffer:
        return {
            "status": "skipped",
            "framework": "torch",
            "reason": empty_reason,
        }

    feature_rows = [rec[0] for rec in replay_buffer]
    policy_target_ids = [rec[1] for rec in replay_buffer]
    policy_target_probs = [rec[2] for rec in replay_buffer]
    value_targets = [rec[3] for rec in replay_buffer]
    value_phases = [rec[4] for rec in replay_buffer]
    played_action_ids = [rec[5] for rec in replay_buffer]
    sample_extras = [rec[6] for rec in replay_buffer]

    input_dim = len(feature_rows[0])
    hidden = max(min_hidden, int(getattr(config.trainer, "hidden", 512)))
    mlp_layers = max(1, int(getattr(config.trainer, "mlp_layers", 4)))
    updates_per_step = max(1, int(getattr(config.trainer, "updates_per_step", 1)))
    steps = max(0, int(getattr(config.trainer, "steps", 0)))
    batch = max(16, int(config.trainer.batch_size))
    epochs = max(1, int(config.trainer.epochs))
    value_late_weight = max(0.0, float(getattr(config.trainer, "value_late_weight", 0.0)))

    runtime_signature = (
        int(input_dim),
        int(policy_dim),
        int(hidden),
        int(mlp_layers),
        float(config.trainer.learning_rate),
    )
    net, opt, resumed, resumed_from = get_or_create_torch_runtime(
        cache_key=cache_key,
        runtime_signature=runtime_signature,
        build_net=lambda: create_pvnet(input_dim, policy_dim, hidden, mlp_layers, nn),
        build_optimizer=lambda model: optim.Adam(model.parameters(), lr=float(config.trainer.learning_rate)),
        resume_checkpoint_path=resume_checkpoint_path,
        torch_module=torch,
    )
    net.train()
    if resumed:
        if step_index is not None and total_steps is not None:
            print(
                f"[train] trainer resume: step={step_index}/{total_steps} checkpoint={resumed_from}",
                flush=True,
            )
        else:
            print(f"[train] trainer resume: checkpoint={resumed_from}", flush=True)

    total_losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    custom_metric_values: dict[str, list[float]] = {}

    def _train_update(x_i, y_soft_i, z_i, w_i) -> None:
        logits, val = net(x_i)
        log_probs = torch.log_softmax(logits, dim=1)
        p_loss = -(y_soft_i * log_probs).sum(dim=1).mean()
        sq_err = (val - z_i) ** 2
        v_loss = (sq_err * w_i).sum() / w_i.sum().clamp_min(1e-6)
        loss = p_loss + 0.5 * v_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total_losses.append(float(loss.detach().cpu().item()))
        policy_losses.append(float(p_loss.detach().cpu().item()))
        value_losses.append(float(v_loss.detach().cpu().item()))

    n = len(feature_rows)
    extra_keys = sorted({k for row in sample_extras for k in row.keys()})

    def _build_minibatch(sample_idx: list[int]):
        x_i = torch.tensor([feature_rows[i] for i in sample_idx], dtype=torch.float32)
        y_soft_i = torch.zeros((len(sample_idx), policy_dim), dtype=torch.float32)
        for row, i in enumerate(sample_idx):
            ids = policy_target_ids[i]
            probs = policy_target_probs[i]
            if ids and probs and len(ids) == len(probs):
                for aid, prob in zip(ids, probs):
                    if 0 <= aid < policy_dim:
                        y_soft_i[row, aid] += float(prob)
            row_sum = float(y_soft_i[row].sum().item())
            if row_sum <= 1e-12:
                fallback_aid = int(played_action_ids[i])
                if 0 <= fallback_aid < policy_dim:
                    y_soft_i[row, fallback_aid] = 1.0
            else:
                y_soft_i[row] /= row_sum
        z_i = torch.tensor([[value_targets[i]] for i in sample_idx], dtype=torch.float32)
        phase_i = torch.tensor([[value_phases[i]] for i in sample_idx], dtype=torch.float32)
        extras = {
            key: torch.tensor(
                [[float(sample_extras[i].get(key, 0.0))] for i in sample_idx],
                dtype=torch.float32,
            )
            for key in extra_keys
        }
        w_i, extra_metrics = weight_builder_fn(phase_i, extras, value_late_weight, torch)
        if extra_metrics:
            for key, value in extra_metrics.items():
                custom_metric_values.setdefault(str(key), []).append(float(value))
        return x_i, y_soft_i, z_i, w_i

    if steps > 0:
        sgd_passes = max(1, int(epochs))
        for step_idx in range(steps):
            before = len(total_losses)
            for _ in range(sgd_passes):
                for _ in range(updates_per_step):
                    idx = torch.randint(0, n, (min(batch, n),), dtype=torch.int64).tolist()
                    x_i, y_soft_i, z_i, w_i = _build_minibatch(idx)
                    _train_update(x_i, y_soft_i, z_i, w_i)
            done = len(total_losses) - before
            print(
                f"[train] trainer step {step_idx + 1}/{steps}: "
                f"updates={done} "
                f"loss={mean_last(total_losses, done):.6f} "
                f"policy={mean_last(policy_losses, done):.6f} "
                f"value={mean_last(value_losses, done):.6f}",
                flush=True,
            )
    else:
        for epoch_idx in range(epochs):
            order = torch.randperm(n, dtype=torch.int64).tolist()
            before = len(total_losses)
            for i in range(0, n, batch):
                idx = order[i : i + batch]
                x_i, y_soft_i, z_i, w_i = _build_minibatch(idx)
                _train_update(x_i, y_soft_i, z_i, w_i)
            done = len(total_losses) - before
            print(
                f"[train] trainer epoch {epoch_idx + 1}/{epochs}: "
                f"updates={done} "
                f"loss={mean_last(total_losses, done):.6f} "
                f"policy={mean_last(policy_losses, done):.6f} "
                f"value={mean_last(value_losses, done):.6f}",
                flush=True,
            )

    net.eval()
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "candidate_model.pt"
    checkpoint_saved = save_checkpoint(
        checkpoint_path=checkpoint_path,
        net=net,
        optimizer=opt,
        torch_module=torch,
        extra={
            "input_dim": input_dim,
            "policy_dim": policy_dim,
            "hidden": hidden,
            "mlp_layers": mlp_layers,
        },
    )
    onnx_path = model_dir / "candidate_model.onnx"
    export_pvnet_onnx(
        net=net,
        output_path=onnx_path,
        input_dim=input_dim,
        torch_module=torch,
    )

    metrics: dict[str, float] = {
        "final_total_loss": float(total_losses[-1]) if total_losses else 0.0,
        "mean_total_loss": float(sum(total_losses) / len(total_losses)) if total_losses else 0.0,
        "final_policy_loss": float(policy_losses[-1]) if policy_losses else 0.0,
        "mean_policy_loss": float(sum(policy_losses) / len(policy_losses)) if policy_losses else 0.0,
        "final_value_loss": float(value_losses[-1]) if value_losses else 0.0,
        "mean_value_loss": float(sum(value_losses) / len(value_losses)) if value_losses else 0.0,
        "train_samples": float(len(replay_buffer)),
        "updates": float(len(total_losses)),
    }
    for key, values in custom_metric_values.items():
        if values:
            metrics[key] = float(sum(values) / len(values))
    if fixed_metrics:
        for key, value in fixed_metrics.items():
            metrics[str(key)] = float(value)

    return {
        "status": "completed",
        "framework": "torch",
        "model_path": str(onnx_path),
        "checkpoint_path": checkpoint_saved,
        "metrics": metrics,
        "trainer_config": asdict(config.trainer),
    }

