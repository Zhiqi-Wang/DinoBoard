from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ServiceSession:
    session_id: str
    handle: int
    human_player: int
    owner_key: str


@dataclass(slots=True)
class ServiceError(Exception):
    code: str
    message: str
    details: dict[str, Any]


class IGameSessionBackend(Protocol):
    def session_new(self, seed: int, human_player: int) -> int: ...
    def session_delete(self, handle: int) -> None: ...
    def session_payload(self, handle: int) -> dict[str, Any]: ...
    def session_legal_actions(self, handle: int) -> list[dict[str, Any]]: ...
    def session_apply_action(self, handle: int, action_id: int, forced: bool) -> dict[str, Any] | None: ...
    def session_ai_move(
        self,
        handle: int,
        engine: str,
        simulations: int,
        temperature: float,
        model_path: str | None = None,
        time_budget_ms: int = 0,
        search_options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...
    def session_can_step_back(self, handle: int) -> bool: ...
    def session_step_back(self, handle: int) -> bool: ...
    def session_replay_events(self, handle: int) -> list[dict[str, Any]]: ...
    def session_frames_range(self, handle: int, from_ply: int, to_ply: int) -> list[dict[str, Any]] | None: ...
    def session_frame_at(self, handle: int, ply: int) -> dict[str, Any] | None: ...
    def session_seek(self, handle: int, ply: int) -> bool: ...
    def session_rebuild_frames(self, handle: int) -> int: ...
