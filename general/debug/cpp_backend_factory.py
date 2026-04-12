from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from ..search_options import clone_search_options
from .service_interfaces import IGameSessionBackend


class CppModuleSessionBackend(IGameSessionBackend):
    """Adapter for cpp_*_engine modules exposing session_* functions."""

    def __init__(self, cpp_module: ModuleType) -> None:
        self._m = cpp_module

    def session_new(self, seed: int, human_player: int) -> int:
        return int(self._m.session_new(seed, human_player))

    def session_delete(self, handle: int) -> None:
        self._m.session_delete(handle)

    def session_payload(self, handle: int) -> dict[str, Any]:
        return self._m.session_payload(handle)

    def session_legal_actions(self, handle: int) -> list[dict[str, Any]]:
        return self._m.session_legal_actions(handle)

    def session_apply_action(self, handle: int, action_id: int, forced: bool) -> dict[str, Any] | None:
        return self._m.session_apply_action(handle, action_id, 1 if forced else 0)

    def session_ai_move(
        self,
        handle: int,
        engine: str,
        simulations: int,
        temperature: float,
        model_path: str | None = None,
        time_budget_ms: int = 0,
        search_options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        options_map = clone_search_options(search_options)
        options = options_map if options_map else None
        if model_path:
            if options is not None:
                try:
                    return self._m.session_ai_move(
                        handle, engine, simulations, temperature, time_budget_ms, model_path, options
                    )
                except TypeError:
                    pass
            return self._m.session_ai_move(handle, engine, simulations, temperature, time_budget_ms, model_path)
        if options is not None:
            try:
                return self._m.session_ai_move(handle, engine, simulations, temperature, time_budget_ms, None, options)
            except TypeError:
                pass
        return self._m.session_ai_move(handle, engine, simulations, temperature, time_budget_ms)

    def session_can_step_back(self, handle: int) -> bool:
        return bool(self._m.session_can_step_back(handle))

    def session_step_back(self, handle: int) -> bool:
        return bool(self._m.session_step_back(handle))

    def session_replay_events(self, handle: int) -> list[dict[str, Any]]:
        return self._m.session_replay_events(handle)

    def session_frames_range(self, handle: int, from_ply: int, to_ply: int) -> list[dict[str, Any]] | None:
        return self._m.session_frames_range(handle, from_ply, to_ply)

    def session_frame_at(self, handle: int, ply: int) -> dict[str, Any] | None:
        return self._m.session_frame_at(handle, ply)

    def session_seek(self, handle: int, ply: int) -> bool:
        return bool(self._m.session_seek(handle, ply))

    def session_rebuild_frames(self, handle: int) -> int:
        return int(self._m.session_rebuild_frames(handle))


def create_cpp_session_backend(cpp_module: ModuleType) -> IGameSessionBackend:
    return CppModuleSessionBackend(cpp_module)


def create_cpp_session_backend_from_module_name(cpp_module_name: str) -> IGameSessionBackend:
    name = str(cpp_module_name).strip()
    if not name:
        raise ValueError("cpp_module_name must be non-empty")
    cpp_module = importlib.import_module(name)
    return create_cpp_session_backend(cpp_module)
