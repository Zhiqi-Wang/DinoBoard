from __future__ import annotations

from .config import PolicyConfig


def apply_policy_search_options_bridge(policy_cfg: PolicyConfig, ctx: dict[str, object]) -> PolicyConfig:
    """Bridge PolicyConfig fields into search_options with sane defaults.

    By default:
    - selfplay/warm_start keep configured Dirichlet exploration fields;
    - periodic_eval/gating candidate/opponent roles force deterministic Dirichlet=0.
    """
    phase = str(ctx.get("phase", "") or "")
    role = str(ctx.get("role", "") or "")
    opts = dict(policy_cfg.search_options or {})
    if phase in {"periodic_eval", "gating"} and role in {"candidate", "benchmark", "history_best"}:
        dir_alpha = 0.0
        dir_epsilon = 0.0
        dir_nplies = 0
    else:
        dir_alpha = float(policy_cfg.dirichlet_alpha)
        dir_epsilon = float(policy_cfg.dirichlet_epsilon)
        dir_nplies = int(policy_cfg.dirichlet_on_first_n_plies)
    opts["dirichlet_alpha"] = dir_alpha
    opts["dirichlet_epsilon"] = dir_epsilon
    opts["dirichlet_on_first_n_plies"] = dir_nplies
    policy_cfg.search_options = opts
    return policy_cfg

