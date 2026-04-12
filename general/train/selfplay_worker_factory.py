from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .selfplay_adapter import (
    build_arena_match_payload,
    build_selfplay_episode_payload,
    build_worker_selfplay_episode_result,
    parse_value_margin_params,
)

EpisodeRunner = Callable[[int, int, Any, str], dict[str, Any]]
ArenaRunner = Callable[[int, Any, Any], dict[str, Any]]


@dataclass(slots=True)
class WorkerSelfplayEpisodeRunner:
    backend_factory: Callable[[], Any]
    game_type: str
    ruleset: str
    read_shared_victory_from_raw: bool
    enrich_sample: Callable[[dict[str, Any], dict[str, Any]], None] | None = None
    netmcts_data_source: str | None = None
    default_value_margin_weight: float | None = None
    default_value_margin_scale: float | None = None

    def __call__(
        self,
        episode_index: int,
        seed: int,
        policy: Any,
        search_params_hash: str,
    ) -> dict[str, Any]:
        backend = self.backend_factory()
        raw = backend.run_selfplay_episode_fast(seed=seed, policy=policy)
        payload = build_selfplay_episode_payload(
            raw=raw,
            episode_index=episode_index,
            seed=seed,
            policy=policy,
            search_params_hash=search_params_hash,
            game_type=self.game_type,
            ruleset=self.ruleset,
            read_shared_victory_from_raw=self.read_shared_victory_from_raw,
            netmcts_data_source=self.netmcts_data_source,
            enrich_sample=self.enrich_sample,
        )
        with_margin = self.default_value_margin_weight is not None and self.default_value_margin_scale is not None
        if not with_margin:
            return payload
        label_params = parse_value_margin_params(
            policy.search_options,
            default_weight=float(self.default_value_margin_weight),
            default_scale=float(self.default_value_margin_scale),
        )
        return build_worker_selfplay_episode_result(
            raw=raw,
            payload=payload,
            label_params=label_params,
        )


@dataclass(slots=True)
class WorkerArenaRunner:
    backend_factory: Callable[[], Any]

    def __call__(self, seed: int, p0: Any, p1: Any) -> dict[str, Any]:
        backend = self.backend_factory()
        raw = backend.run_arena_match_fast(seed=seed, p0=p0, p1=p1)
        return build_arena_match_payload(seed=seed, raw=raw)


def create_worker_runners(
    *,
    backend_factory: Callable[[], Any],
    game_type: str,
    ruleset: str,
    read_shared_victory_from_raw: bool,
    enrich_sample: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    netmcts_data_source: str | None = None,
    default_value_margin_weight: float | None = None,
    default_value_margin_scale: float | None = None,
) -> tuple[EpisodeRunner, ArenaRunner]:
    return (
        WorkerSelfplayEpisodeRunner(
            backend_factory=backend_factory,
            game_type=game_type,
            ruleset=ruleset,
            read_shared_victory_from_raw=read_shared_victory_from_raw,
            enrich_sample=enrich_sample,
            netmcts_data_source=netmcts_data_source,
            default_value_margin_weight=default_value_margin_weight,
            default_value_margin_scale=default_value_margin_scale,
        ),
        WorkerArenaRunner(backend_factory=backend_factory),
    )

