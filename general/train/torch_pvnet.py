from __future__ import annotations

from pathlib import Path


def create_pvnet(input_dim: int, policy_dim: int, hidden: int, mlp_layers: int, nn):
    """Build a generic MLP backbone + policy/value dual-head network."""

    class PvNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.ReLU()]
            for _ in range(max(1, mlp_layers) - 1):
                layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
            self.backbone = nn.Sequential(*layers)
            self.policy = nn.Linear(hidden, policy_dim)
            self.value = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())

        def forward(self, t):
            h = self.backbone(t)
            return self.policy(h), self.value(h)

    return PvNet()


def export_pvnet_onnx(*, net, output_path: Path, input_dim: int, torch_module) -> str:
    """Export a policy-value network to ONNX with standard names/axes."""
    net.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch_module.zeros((1, input_dim), dtype=torch_module.float32)
    torch_module.onnx.export(
        net,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["policy", "value"],
        dynamic_axes={"features": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=13,
    )
    return str(output_path)


def export_initial_policy_onnx_from_config(
    *,
    config,
    output_path: Path,
    input_dim: int,
    policy_dim: int,
    seed: int = 20260323,
) -> str:
    """Export a randomly initialized PVNet ONNX from trainer config."""
    import torch

    hidden = max(32, int(getattr(config.trainer, "hidden", 512)))
    mlp_layers = max(1, int(getattr(config.trainer, "mlp_layers", 4)))
    torch.manual_seed(int(seed))
    net = create_pvnet(input_dim, policy_dim, hidden, mlp_layers, torch.nn)
    return export_pvnet_onnx(
        net=net,
        output_path=output_path,
        input_dim=input_dim,
        torch_module=torch,
    )


def mean_last(values: list[float], count: int) -> float:
    if not values:
        return 0.0
    n_recent = max(1, min(int(count), len(values)))
    recent = values[-n_recent:]
    return float(sum(recent) / len(recent))

