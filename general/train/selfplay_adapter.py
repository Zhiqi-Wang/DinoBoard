from __future__ import annotations

import os
from typing import Any, Callable

from .policy_target_utils import normalize_sparse_policy


def outcome_value(player_id: int, winner: int | None, shared_victory: bool) -> float:
    if winner is None or shared_victory:
        return 0.0
    return 1.0 if player_id == winner else -1.0


def data_source_from_engine(engine: str, *, netmcts_label: str | None = None) -> str:
    if engine == "heuristic":
        return "heuristic_selfplay"
    if engine == "netmcts" and netmcts_label:
        return netmcts_label
    return "mcts_selfplay"


def build_selfplay_episode_payload(
    *,
    raw: dict[str, Any],
    episode_index: int,
    seed: int,
    policy: Any,
    search_params_hash: str,
    game_type: str,
    ruleset: str,
    read_shared_victory_from_raw: bool,
    netmcts_data_source: str | None = None,
    enrich_sample: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    winner_raw = raw.get("winner")
    if winner_raw is None:
        winner = None
    else:
        try:
            winner = int(winner_raw)
        except (TypeError, ValueError):
            winner = None
    shared_victory = bool(raw.get("shared_victory", False)) if read_shared_victory_from_raw else False
    data_source = data_source_from_engine(policy.engine, netmcts_label=netmcts_data_source)
    samples: list[dict[str, Any]] = []

    for rec in list(raw.get("samples", [])):
        player = int(rec.get("player", 0))
        sample: dict[str, Any] = {
            "game_type": game_type,
            "ruleset": ruleset,
            "seed": seed,
            "episode": episode_index,
            "ply": int(rec.get("ply", 0)),
            "player": player,
            "action_id": int(rec.get("action_id", -1)),
            "state_version": int(rec.get("state_version", 0)),
            "search_params_hash": search_params_hash,
            "data_source": data_source,
            "search_params": {
                "engine": policy.engine,
                "simulations": policy.simulations,
                "temperature": policy.temperature,
                "dirichlet_alpha": policy.dirichlet_alpha,
                "dirichlet_epsilon": policy.dirichlet_epsilon,
                "dirichlet_on_first_n_plies": policy.dirichlet_on_first_n_plies,
            },
            "z": outcome_value(player, winner, shared_victory),
        }
        ids, probs = normalize_sparse_policy(
            rec.get("policy_action_ids"),
            rec.get("policy_probs"),
            rec.get("policy_action_visits"),
        )
        if ids and probs:
            sample["policy_action_ids"] = ids
            sample["policy_probs"] = probs
        if enrich_sample is not None:
            enrich_sample(sample, rec)
        samples.append(sample)

    return {
        "winner": winner,
        "shared_victory": shared_victory,
        "plies": int(raw.get("plies", len(samples))),
        "scores": list(raw.get("scores", [])) if isinstance(raw.get("scores"), list) else None,
        "win_margin_steps": int(raw.get("win_margin_steps", 0))
        if isinstance(raw.get("win_margin_steps"), (int, float))
        else 0,
        "samples": samples,
    }


def build_arena_match_payload(*, seed: int, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": seed,
        "winner": raw.get("winner"),
        "shared_victory": bool(raw.get("shared_victory", False)),
        "scores": list(raw.get("scores", [])),
    }


def parse_value_margin_params(
    search_options: dict[str, Any] | None,
    *,
    default_weight: float,
    default_scale: float,
) -> dict[str, float]:
    opts = dict(search_options or {})
    try:
        margin_weight = float(opts.get("value_margin_weight", default_weight))
    except (TypeError, ValueError):
        margin_weight = float(default_weight)
    margin_weight = max(0.0, min(1.0, margin_weight))
    try:
        margin_scale = float(opts.get("value_margin_scale", default_scale))
    except (TypeError, ValueError):
        margin_scale = float(default_scale)
    margin_scale = max(1e-6, margin_scale)
    return {
        "value_margin_weight": margin_weight,
        "value_margin_scale": margin_scale,
    }


def build_worker_selfplay_episode_result(
    *,
    raw: dict[str, Any],
    payload: dict[str, Any],
    label_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    scores = raw.get("scores")
    out: dict[str, Any] = {
        "worker_pid": int(os.getpid()),
        "winner": payload.get("winner"),
        "shared_victory": bool(payload.get("shared_victory", False)),
        "scores": list(scores) if isinstance(scores, list) else None,
        "win_margin_steps": int(payload.get("win_margin_steps", 0))
        if isinstance(payload.get("win_margin_steps"), (int, float))
        else 0,
        "plies": int(payload.get("plies", len(payload.get("samples", [])))),
        "samples": list(payload.get("samples", [])),
    }
    if label_params is not None:
        out["label_params"] = dict(label_params)
    return out
