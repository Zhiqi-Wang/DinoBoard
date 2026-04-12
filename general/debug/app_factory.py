from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runtime_service import DebugServiceRuntime
from .service_interfaces import IGameSessionBackend, ServiceError


def _error_response(code: str, message: str, details: dict[str, Any] | None = None) -> HTTPException:
    payload = {"error": {"code": code, "message": message, "details": details or {}}}
    return HTTPException(status_code=400, detail=payload)


def _to_http_error(err: ServiceError) -> HTTPException:
    return _error_response(err.code, err.message, err.details)


class CreateGameRequest(BaseModel):
    seed: int | None = None
    human_player: int = Field(default=0, ge=0, le=1)


class ActionRequest(BaseModel):
    action_id: int
    state_version: int = Field(ge=0)


class AiMoveRequest(BaseModel):
    state_version: int = Field(ge=0)
    engine: str = "heuristic"
    simulations: int | None = Field(default=None, ge=0)
    time_budget_ms: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.0, ge=0.0)
    model_path: str | None = None
    search_options: dict[str, Any] | None = None


class StepBackRequest(BaseModel):
    state_version: int = Field(ge=0)


class SeekRequest(BaseModel):
    ply: int = Field(ge=0)


def create_debug_service_app(
    *,
    title: str,
    version: str,
    project_dir: Path,
    web_dir: Path,
    backend: IGameSessionBackend,
    action_id_min: int,
    action_id_max: int,
    allowed_engines: tuple[str, ...],
    default_engine: str,
    default_simulations: int,
    default_model_path: Path | None = None,
    require_model_exists_for_engines: tuple[str, ...] = (),
) -> FastAPI:
    app = FastAPI(title=title, version=version)
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    app.mount("/general-web", StaticFiles(directory=str(project_dir / "general" / "web")), name="general-web")
    runtime = DebugServiceRuntime(backend=backend)
    required_model_engine_set = set(require_model_exists_for_engines)
    allowed_engine_set = set(allowed_engines)
    client_cookie_name = "dino_debug_client_id"

    def _extract_client_ip(request: Request) -> str:
        forwarded_for = (request.headers.get("x-forwarded-for") or "").strip()
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            if first:
                return first
        host = request.client.host if request.client else ""
        return host or "unknown"

    def _resolve_client_key(
        request: Request,
        response: Response | None = None,
        *,
        ensure_cookie: bool = False,
    ) -> str:
        cookie_client_id = (request.cookies.get(client_cookie_name) or "").strip()
        if cookie_client_id:
            return f"cid:{cookie_client_id}"
        if ensure_cookie:
            client_id = f"c_{secrets.token_urlsafe(18)}"
            if response is not None:
                response.set_cookie(
                    key=client_cookie_name,
                    value=client_id,
                    httponly=False,
                    samesite="lax",
                )
            return f"cid:{client_id}"
        return f"ip:{_extract_client_ip(request)}"

    def _resolve_ai_request(req: AiMoveRequest) -> tuple[str, str | None, int]:
        if req.engine not in allowed_engine_set:
            raise _error_response(
                "INVALID_ENGINE",
                "engine is not supported for this game",
                {"engine": req.engine, "allowed_engines": list(allowed_engines)},
            )
        effective_engine = req.engine
        model_path = req.model_path
        # Guardrail: if caller sends mcts but game supports netmcts and has a default model,
        # promote to netmcts so trained model can actually participate in decision.
        if (
            req.engine == "mcts"
            and "netmcts" in allowed_engine_set
            and default_model_path is not None
            and model_path is None
        ):
            effective_engine = "netmcts"
        if (
            not model_path
            and default_model_path is not None
            and (effective_engine == default_engine or effective_engine == "netmcts")
        ):
            model_path = str(default_model_path)
        if effective_engine in required_model_engine_set and model_path and not Path(model_path).exists():
            raise _error_response(
                "MODEL_NOT_FOUND",
                "model file does not exist",
                {"model_path": model_path},
            )
        simulations = default_simulations if req.simulations is None else req.simulations
        return effective_engine, model_path, simulations

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(str(web_dir / "index.html"))

    @app.post("/api/v1/games")
    def create_game(req: CreateGameRequest, request: Request, response: Response) -> dict[str, Any]:
        owner_key = _resolve_client_key(request, response, ensure_cookie=True)
        try:
            return runtime.create_game(seed=req.seed, human_player=req.human_player, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.delete("/api/v1/games/{session_id}")
    def delete_game(session_id: str, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.delete_game(session_id, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.get("/api/v1/games/{session_id}/state")
    def get_state(session_id: str, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.get_state(session_id, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.get("/api/v1/games/{session_id}/legal-actions")
    def get_legal_actions(session_id: str, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.get_legal_actions(session_id, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/actions")
    def post_human_action(session_id: str, req: ActionRequest, request: Request) -> dict[str, Any]:
        if req.action_id < action_id_min or req.action_id > action_id_max:
            raise _error_response(
                "INVALID_ACTION_ID",
                "action_id is out of allowed range",
                {"min": action_id_min, "max": action_id_max, "got": req.action_id},
            )
        owner_key = _resolve_client_key(request)
        try:
            return runtime.post_action(session_id, action_id=req.action_id, state_version=req.state_version, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/force-opponent-move")
    def force_opponent_move(session_id: str, req: ActionRequest, request: Request) -> dict[str, Any]:
        if req.action_id < action_id_min or req.action_id > action_id_max:
            raise _error_response(
                "INVALID_ACTION_ID",
                "action_id is out of allowed range",
                {"min": action_id_min, "max": action_id_max, "got": req.action_id},
            )
        owner_key = _resolve_client_key(request)
        try:
            return runtime.force_opponent_move(
                session_id,
                action_id=req.action_id,
                state_version=req.state_version,
                owner_key=owner_key,
            )
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/ai-move")
    def post_ai_move(session_id: str, req: AiMoveRequest, request: Request) -> dict[str, Any]:
        effective_engine, model_path, simulations = _resolve_ai_request(req)
        owner_key = _resolve_client_key(request)
        try:
            return runtime.post_ai_move(
                session_id,
                state_version=req.state_version,
                engine=effective_engine,
                simulations=simulations,
                time_budget_ms=req.time_budget_ms,
                temperature=req.temperature,
                model_path=model_path,
                search_options=req.search_options,
                owner_key=owner_key,
            )
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/ai-hint")
    def post_ai_hint(session_id: str, req: AiMoveRequest, request: Request) -> dict[str, Any]:
        effective_engine, model_path, simulations = _resolve_ai_request(req)
        owner_key = _resolve_client_key(request)
        try:
            return runtime.post_ai_hint(
                session_id,
                state_version=req.state_version,
                engine=effective_engine,
                simulations=simulations,
                time_budget_ms=req.time_budget_ms,
                temperature=req.temperature,
                model_path=model_path,
                search_options=req.search_options,
                owner_key=owner_key,
            )
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/step-back")
    def step_back(session_id: str, req: StepBackRequest, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.step_back(session_id, state_version=req.state_version, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.get("/api/v1/games/{session_id}/replay")
    def get_replay(session_id: str, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.get_replay(session_id, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.get("/api/v1/games/{session_id}/frames")
    def get_frames(session_id: str, request: Request, from_ply: int = 0, to_ply: int | None = None) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.get_frames(session_id, from_ply=from_ply, to_ply=to_ply, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.get("/api/v1/games/{session_id}/frames/{ply}")
    def get_frame_by_ply(session_id: str, ply: int, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.get_frame_at(session_id, ply=ply, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/seek")
    def seek_frame(session_id: str, req: SeekRequest, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.seek(session_id, ply=req.ply, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    @app.post("/api/v1/games/{session_id}/rebuild-frames")
    def rebuild_frames(session_id: str, request: Request) -> dict[str, Any]:
        owner_key = _resolve_client_key(request)
        try:
            return runtime.rebuild_frames(session_id, owner_key=owner_key)
        except ServiceError as e:
            raise _to_http_error(e)

    return app
