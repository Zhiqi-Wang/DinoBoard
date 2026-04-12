from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameRegistration:
    game_id: str
    label: str
    app_module: str

    @property
    def mount_path(self) -> str:
        return f"/games/{self.game_id}"

    @property
    def mount_url(self) -> str:
        return f"{self.mount_path}/"


GAME_REGISTRY: tuple[GameRegistration, ...] = (
    GameRegistration(game_id="azul", label="花砖物语", app_module="games.azul.debug_service.app"),
    GameRegistration(game_id="quoridor", label="步步为营", app_module="games.quoridor.debug_service.app"),
    GameRegistration(game_id="splendor", label="璀璨宝石", app_module="games.splendor.debug_service.app"),
    GameRegistration(game_id="tictactoe", label="井字棋", app_module="games.tictactoe.debug_service.app"),
)

