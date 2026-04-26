from __future__ import annotations

import importlib
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .registry import GAME_REGISTRY

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

app = FastAPI(title="DinoBoard Intelligence Debug Hub", version="0.1.0")

for game in GAME_REGISTRY:
    module = importlib.import_module(game.app_module)
    sub_app = getattr(module, "app", None)
    if sub_app is None:
        raise RuntimeError(f"missing 'app' in module: {game.app_module}")
    app.mount(game.mount_path, sub_app)


@app.get("/")
def root() -> RedirectResponse:
    default = GAME_REGISTRY[0]
    return RedirectResponse(url=default.mount_url)


def _redirect_no_slash_factory(game_id: str, mount_url: str):
    def _redirect() -> RedirectResponse:
        return RedirectResponse(url=mount_url)

    _redirect.__name__ = f"redirect_{game_id}_no_slash"
    return _redirect


for game in GAME_REGISTRY:
    app.add_api_route(game.mount_path, _redirect_no_slash_factory(game.game_id, game.mount_url), methods=["GET"])


@app.get("/api/games")
def list_games() -> list[dict[str, str]]:
    return [{"id": g.game_id, "label": g.label, "url": g.mount_url} for g in GAME_REGISTRY]


