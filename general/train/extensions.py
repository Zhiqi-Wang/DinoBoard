from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from .config import PolicyConfig
from .selfplay_adapter import parse_value_margin_params

PolicyPrepareHook = Callable[[PolicyConfig, dict[str, Any]], PolicyConfig]
EvalSummaryHook = Callable[[dict[str, Any], dict[str, Any]], None]
EpisodeContextHook = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
SampleLabelHook = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]
EpisodePayloadHook = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class TrainPipelineHooks:
    """Optional game-provided hooks consumed by the general pipeline."""

    prepare_selfplay_policy: PolicyPrepareHook | None = None
    prepare_eval_policy: PolicyPrepareHook | None = None
    enrich_periodic_eval_summary: EvalSummaryHook | None = None
    build_episode_context: EpisodeContextHook | None = None
    label_sample: SampleLabelHook | None = None
    postprocess_episode_payload: EpisodePayloadHook | None = None


def build_score_margin_pipeline_hooks(
    *,
    default_label_version: str,
    default_margin_weight: float,
    default_margin_scale: float,
) -> TrainPipelineHooks:
    """Build reusable hooks for score-margin value target shaping."""

    def _build_episode_context(raw_episode: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        label_params = raw_episode.get("label_params") or {}
        margin_params = parse_value_margin_params(
            label_params,
            default_weight=float(default_margin_weight),
            default_scale=float(default_margin_scale),
        )
        margin_weight = float(margin_params["value_margin_weight"])
        margin_scale = float(margin_params["value_margin_scale"])

        winner_raw = raw_episode.get("winner")
        winner = int(winner_raw) if winner_raw is not None else None
        shared = bool(raw_episode.get("shared_victory", False))
        scores = raw_episode.get("scores")
        score_diff_p0 = 0.0
        if isinstance(scores, list) and len(scores) >= 2:
            try:
                score_diff_p0 = float(scores[0]) - float(scores[1])
            except (TypeError, ValueError):
                score_diff_p0 = 0.0
        if shared or winner is None:
            base = {0: 0.0, 1: 0.0}
        else:
            base = {winner: 1.0, 1 - winner: -1.0}
        margin_p0 = math.tanh(score_diff_p0 / margin_scale)
        margin = {0: margin_p0, 1: -margin_p0}
        value_targets = {
            0: float((1.0 - margin_weight) * base[0] + margin_weight * margin[0]),
            1: float((1.0 - margin_weight) * base[1] + margin_weight * margin[1]),
        }
        plies = max(1, int(raw_episode.get("plies", 1)))
        return {
            "plies": plies,
            "value_targets": value_targets,
            "label_version": default_label_version,
        }

    def _label_sample(sample: dict[str, Any], episode_ctx: dict[str, Any], _ctx: dict[str, Any]) -> dict[str, Any]:
        value_targets = episode_ctx.get("value_targets", {})
        player = int(sample.get("player", 0))
        plies = max(1, int(episode_ctx.get("plies", 1)))
        denom = max(1, plies - 1)
        ply = int(sample.get("ply", 0))
        return {
            "z": float(value_targets.get(player, 0.0)),
            "phase": float(max(0.0, min(1.0, float(ply) / float(denom)))),
            "label_version": str(episode_ctx.get("label_version", default_label_version)),
        }

    return TrainPipelineHooks(
        build_episode_context=_build_episode_context,
        label_sample=_label_sample,
    )
