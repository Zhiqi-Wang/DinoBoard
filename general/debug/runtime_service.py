from __future__ import annotations

import uuid
import secrets
from typing import Any

from ..search_options import clone_search_options
from .service_interfaces import IGameSessionBackend, ServiceError, ServiceSession


class DebugServiceRuntime:
    def __init__(self, backend: IGameSessionBackend) -> None:
        self._backend = backend
        self._sessions: dict[str, ServiceSession] = {}
        self._owner_to_session_id: dict[str, str] = {}

    def _must_session(self, session_id: str, owner_key: str) -> ServiceSession:
        session = self._sessions.get(session_id)
        if not session:
            raise ServiceError("SESSION_NOT_FOUND", "session does not exist", {"session_id": session_id})
        if session.owner_key != owner_key:
            raise ServiceError(
                "SESSION_NOT_FOUND",
                "session does not exist",
                {"session_id": session_id},
            )
        return session

    def _payload(self, session: ServiceSession) -> dict[str, Any]:
        return self._backend.session_payload(session.handle)

    def _delete_session(self, session: ServiceSession) -> None:
        try:
            self._backend.session_delete(int(session.handle))
        except Exception as e:
            raise ServiceError(
                "SESSION_DELETE_FAILED",
                "failed to delete backend session",
                {"session_id": session.session_id, "handle": int(session.handle), "error": str(e)},
            ) from e

    def _drop_session_record(self, session: ServiceSession) -> None:
        self._sessions.pop(session.session_id, None)
        if self._owner_to_session_id.get(session.owner_key) == session.session_id:
            self._owner_to_session_id.pop(session.owner_key, None)

    def _terminate_owner_session(self, owner_key: str) -> bool:
        session_id = self._owner_to_session_id.get(owner_key)
        if not session_id:
            return False
        session = self._sessions.get(session_id)
        if not session:
            self._owner_to_session_id.pop(owner_key, None)
            return False
        self._delete_session(session)
        self._drop_session_record(session)
        return True

    def _ensure_state_version(self, session: ServiceSession, state_version: int) -> None:
        payload = self._payload(session)
        current = int(payload.get("state_version", -1))
        if current != state_version:
            raise ServiceError(
                "STATE_VERSION_MISMATCH",
                "state_version does not match current session state",
                {"session_id": session.session_id, "expected": current, "got": state_version},
            )

    @staticmethod
    def _to_prob01_from_mcts_value(value: Any) -> float | None:
        if value is None:
            return None
        try:
            x = float(value)
        except (TypeError, ValueError):
            return None
        if x != x:  # NaN
            return None
        # If already in [0,1], keep; otherwise treat as [-1,1] value.
        if 0.0 <= x <= 1.0:
            p = x
        else:
            p = (x + 1.0) * 0.5
        return max(0.0, min(1.0, p))

    def _normalize_ai_event(self, event: dict[str, Any] | None) -> dict[str, Any] | None:
        if event is None:
            return None
        # Unified field for frontend/general hint panel.
        if "estimated_winrate" in event:
            event["estimated_winrate"] = self._to_prob01_from_mcts_value(event.get("estimated_winrate"))
            return event
        # Backward-compatible fallback keys; convert in general once.
        for key in ("best_action_value", "mcts_best_action_value", "search_best_action_value"):
            if key in event:
                event["estimated_winrate"] = self._to_prob01_from_mcts_value(event.get(key))
                break
        else:
            event["estimated_winrate"] = None
        return event

    def create_game(self, seed: int | None, human_player: int, owner_key: str) -> dict[str, Any]:
        # Keep one active debug game per client.
        # Starting a new game only tears down the caller's previous session.
        self._terminate_owner_session(owner_key)
        session_id = f"g_{uuid.uuid4().hex[:10]}"
        # Keep deterministic behavior when caller explicitly provides seed.
        # When seed is omitted/null, use a random non-zero 64-bit seed so
        # consecutive "start game" actions do not always open the same position.
        resolved_seed = int(seed) if seed is not None else int(secrets.randbits(63) + 1)
        handle = int(self._backend.session_new(resolved_seed, int(human_player)))
        self._sessions[session_id] = ServiceSession(
            session_id=session_id,
            handle=handle,
            human_player=human_player,
            owner_key=owner_key,
        )
        self._owner_to_session_id[owner_key] = session_id
        return {"session_id": session_id, "human_player": human_player, **self._backend.session_payload(handle)}

    def delete_game(self, session_id: str, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        self._delete_session(session)
        self._drop_session_record(session)
        return {"session_id": session_id, "deleted": True}

    def get_state(self, session_id: str, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        return {"session_id": session_id, **self._payload(session)}

    def get_legal_actions(self, session_id: str, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        payload = self._payload(session)
        actions = self._backend.session_legal_actions(session.handle)
        return {"session_id": session_id, "state_version": payload["state_version"], "actions": actions}

    def post_action(self, session_id: str, action_id: int, state_version: int, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        self._ensure_state_version(session, state_version)
        event = self._backend.session_apply_action(session.handle, action_id, False)
        if event is None:
            payload = self._payload(session)
            raise ServiceError(
                "ILLEGAL_ACTION",
                "action is not legal in current state",
                {"session_id": session_id, "state_version": payload["state_version"], "action_id": action_id},
            )
        return {"session_id": session_id, "event": self._normalize_ai_event(event), **self._payload(session)}

    def force_opponent_move(self, session_id: str, action_id: int, state_version: int, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        self._ensure_state_version(session, state_version)
        payload = self._payload(session)
        current_player = int(payload["public_state"]["common"]["current_player"])
        if current_player == session.human_player:
            raise ServiceError(
                "NOT_OPPONENT_TURN",
                "force-opponent-move can only be used on opponent turn",
                {"session_id": session_id, "current_player": current_player},
            )
        event = self._backend.session_apply_action(session.handle, action_id, True)
        if event is None:
            payload2 = self._payload(session)
            raise ServiceError(
                "ILLEGAL_ACTION",
                "action is not legal in current state",
                {"session_id": session_id, "state_version": payload2["state_version"], "action_id": action_id},
            )
        return {"session_id": session_id, "event": self._normalize_ai_event(event), **self._payload(session)}

    def post_ai_move(
        self,
        session_id: str,
        state_version: int,
        engine: str,
        simulations: int,
        temperature: float,
        model_path: str | None = None,
        time_budget_ms: int = 0,
        search_options: dict[str, Any] | None = None,
        owner_key: str = "",
    ) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        self._ensure_state_version(session, state_version)
        options = clone_search_options(search_options)
        event = self._backend.session_ai_move(
            session.handle,
            engine,
            simulations,
            temperature,
            model_path,
            time_budget_ms,
            options,
        )
        if event is None:
            raise ServiceError("NO_LEGAL_ACTION", "no legal action available", {"session_id": session_id})
        return {
            "session_id": session_id,
            "policy": {
                "engine": engine,
                "simulations": simulations,
                "temperature": temperature,
                "model_path": model_path,
                "time_budget_ms": time_budget_ms,
                "search_options": options,
            },
            "event": self._normalize_ai_event(event),
            **self._payload(session),
        }

    def post_ai_hint(
        self,
        session_id: str,
        state_version: int,
        engine: str,
        simulations: int,
        temperature: float,
        model_path: str | None = None,
        time_budget_ms: int = 0,
        search_options: dict[str, Any] | None = None,
        owner_key: str = "",
    ) -> dict[str, Any]:
        """Return AI-recommended action without changing session state."""
        session = self._must_session(session_id, owner_key)
        self._ensure_state_version(session, state_version)
        options = clone_search_options(search_options)
        event = self._backend.session_ai_move(
            session.handle,
            engine,
            simulations,
            temperature,
            model_path,
            time_budget_ms,
            options,
        )
        if event is None:
            raise ServiceError("NO_LEGAL_ACTION", "no legal action available", {"session_id": session_id})

        if not self._backend.session_can_step_back(session.handle):
            raise ServiceError(
                "HINT_ROLLBACK_UNAVAILABLE",
                "cannot rollback after ai hint move; session state may have changed",
                {"session_id": session_id},
            )
        if not self._backend.session_step_back(session.handle):
            raise ServiceError(
                "HINT_ROLLBACK_FAILED",
                "failed to rollback ai hint move",
                {"session_id": session_id},
            )
        payload = self._payload(session)
        restored_version = int(payload.get("state_version", -1))
        if restored_version != state_version:
            raise ServiceError(
                "HINT_STATE_NOT_RESTORED",
                "state was not restored after ai hint move",
                {"session_id": session_id, "expected_state_version": state_version, "restored_state_version": restored_version},
            )
        return {
            "session_id": session_id,
            "policy": {
                "engine": engine,
                "simulations": simulations,
                "temperature": temperature,
                "model_path": model_path,
                "time_budget_ms": time_budget_ms,
                "search_options": options,
            },
            "event": self._normalize_ai_event(event),
            **payload,
        }

    def step_back(self, session_id: str, state_version: int, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        self._ensure_state_version(session, state_version)
        if not self._backend.session_can_step_back(session.handle):
            raise ServiceError("AT_INITIAL_STATE", "already at initial state", {"session_id": session_id})
        self._backend.session_step_back(session.handle)
        return {"session_id": session_id, **self._payload(session)}

    def get_replay(self, session_id: str, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        events = self._backend.session_replay_events(session.handle)
        payload = self._payload(session)
        return {
            "session_id": session_id,
            "state_version": payload["state_version"],
            "event_count": len(events),
            "events": events,
        }

    def get_frames(self, session_id: str, from_ply: int, to_ply: int | None, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        frames = self._backend.session_frames_range(session.handle, from_ply, -1 if to_ply is None else to_ply)
        if frames is None:
            raise ServiceError("INVALID_RANGE", "from ply is greater than to ply", {"from": from_ply, "to": to_ply})
        return {"session_id": session_id, "frames": frames}

    def get_frame_at(self, session_id: str, ply: int, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        frame = self._backend.session_frame_at(session.handle, ply)
        if frame is None:
            raise ServiceError("INVALID_PLY", "requested ply out of range", {"ply": ply})
        return {"session_id": session_id, "frame": frame}

    def seek(self, session_id: str, ply: int, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        if not self._backend.session_seek(session.handle, ply):
            raise ServiceError("INVALID_PLY", "requested ply out of range", {"ply": ply})
        return {"session_id": session_id, **self._payload(session)}

    def rebuild_frames(self, session_id: str, owner_key: str) -> dict[str, Any]:
        session = self._must_session(session_id, owner_key)
        frame_count = int(self._backend.session_rebuild_frames(session.handle))
        return {"session_id": session_id, "frame_count": frame_count}
