from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib
import os
from typing import Any, Callable


@dataclass(slots=True)
class MovePolicy:
    engine: str
    simulations: int
    temperature: float
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0
    dirichlet_on_first_n_plies: int = 0
    model_path: str | None = None
    search_options: dict[str, Any] | None = None


def validate_netmcts_model_path(
    *,
    policy: MovePolicy,
    onnx_enabled_getter: Any = None,
    module_name: str = "cpp_engine",
) -> None:
    if str(policy.engine).lower() != "netmcts":
        return
    if not policy.model_path:
        raise RuntimeError("netmcts requires non-empty model_path")
    if not os.path.exists(str(policy.model_path)):
        raise RuntimeError(f"netmcts model_path not found: {policy.model_path}")
    if onnx_enabled_getter is not None:
        if not callable(onnx_enabled_getter):
            raise RuntimeError(f"{module_name} does not expose onnx_enabled(); please rebuild extension")
        if not bool(onnx_enabled_getter()):
            raise RuntimeError(
                f"{module_name} was built without ONNX support (BOARD_AI_WITH_ONNX=0). "
                "Benchmark/eval would ignore model weights."
            )


class ClassicCppTrainingBackend:
    """Shared wrapper for engines with classic positional C++ signatures."""

    def __init__(self, cpp_module: Any) -> None:
        self._cpp = cpp_module

    def run_selfplay_episode_fast(self, seed: int, policy: MovePolicy) -> dict[str, Any]:
        return self._cpp.run_selfplay_episode_fast(
            int(seed),
            str(policy.engine),
            int(policy.simulations),
            float(policy.temperature),
            str(policy.model_path) if policy.model_path else None,
            float(policy.dirichlet_alpha),
            float(policy.dirichlet_epsilon),
            int(policy.dirichlet_on_first_n_plies),
        )

    def run_arena_match_fast(self, seed: int, p0: MovePolicy, p1: MovePolicy) -> dict[str, Any]:
        return self._cpp.run_arena_match_fast(
            int(seed),
            str(p0.engine),
            int(p0.simulations),
            float(p0.temperature),
            str(p0.model_path) if p0.model_path else None,
            str(p1.engine),
            int(p1.simulations),
            float(p1.temperature),
            str(p1.model_path) if p1.model_path else None,
        )


class SearchOptionsCppTrainingBackend:
    """Shared wrapper for engines requiring dirichlet + search_options arguments."""

    def __init__(self, cpp_module: Any, *, module_name: str = "cpp_engine") -> None:
        self._cpp = cpp_module
        self._module_name = module_name

    def _validate_policy(self, policy: MovePolicy) -> None:
        validate_netmcts_model_path(
            policy=policy,
            onnx_enabled_getter=getattr(self._cpp, "onnx_enabled", None),
            module_name=self._module_name,
        )

    def run_selfplay_episode_fast(self, seed: int, policy: MovePolicy) -> dict[str, Any]:
        self._validate_policy(policy)
        return self._cpp.run_selfplay_episode_fast(
            int(seed),
            str(policy.engine),
            int(policy.simulations),
            float(policy.temperature),
            str(policy.model_path) if policy.model_path else None,
            float(policy.dirichlet_alpha),
            float(policy.dirichlet_epsilon),
            int(policy.dirichlet_on_first_n_plies),
            dict(policy.search_options or {}),
        )

    def run_arena_match_fast(self, seed: int, p0: MovePolicy, p1: MovePolicy) -> dict[str, Any]:
        self._validate_policy(p0)
        self._validate_policy(p1)
        return self._cpp.run_arena_match_fast(
            int(seed),
            str(p0.engine),
            int(p0.simulations),
            float(p0.temperature),
            str(p0.model_path) if p0.model_path else None,
            str(p1.engine),
            int(p1.simulations),
            float(p1.temperature),
            str(p1.model_path) if p1.model_path else None,
            dict(p0.search_options or {}),
            dict(p1.search_options or {}),
        )


class FlexibleSearchOptionsCppTrainingBackend:
    """Shared wrapper for engines with optional search_options arguments."""

    def __init__(self, cpp_module: Any, *, module_name: str = "cpp_engine") -> None:
        self._cpp = cpp_module
        self._module_name = module_name

    def _validate_policy(self, policy: MovePolicy) -> None:
        validate_netmcts_model_path(
            policy=policy,
            onnx_enabled_getter=getattr(self._cpp, "onnx_enabled", None),
            module_name=self._module_name,
        )

    def run_selfplay_episode_fast(self, seed: int, policy: MovePolicy) -> dict[str, Any]:
        self._validate_policy(policy)
        options = dict(policy.search_options or {})
        if policy.model_path:
            if options:
                return self._cpp.run_selfplay_episode_fast(
                    int(seed),
                    str(policy.engine),
                    int(policy.simulations),
                    float(policy.temperature),
                    str(policy.model_path),
                    options,
                )
            return self._cpp.run_selfplay_episode_fast(
                int(seed),
                str(policy.engine),
                int(policy.simulations),
                float(policy.temperature),
                str(policy.model_path),
            )
        if options:
            return self._cpp.run_selfplay_episode_fast(
                int(seed),
                str(policy.engine),
                int(policy.simulations),
                float(policy.temperature),
                None,
                options,
            )
        return self._cpp.run_selfplay_episode_fast(
            int(seed),
            str(policy.engine),
            int(policy.simulations),
            float(policy.temperature),
        )

    def run_arena_match_fast(self, seed: int, p0: MovePolicy, p1: MovePolicy) -> dict[str, Any]:
        self._validate_policy(p0)
        self._validate_policy(p1)
        p0_options = dict(p0.search_options or {})
        p1_options = dict(p1.search_options or {})
        if p0_options or p1_options:
            return self._cpp.run_arena_match_fast(
                int(seed),
                str(p0.engine),
                int(p0.simulations),
                float(p0.temperature),
                str(p0.model_path) if p0.model_path else None,
                str(p1.engine),
                int(p1.simulations),
                float(p1.temperature),
                str(p1.model_path) if p1.model_path else None,
                p0_options if p0_options else None,
                p1_options if p1_options else None,
            )
        return self._cpp.run_arena_match_fast(
            int(seed),
            str(p0.engine),
            int(p0.simulations),
            float(p0.temperature),
            str(p0.model_path) if p0.model_path else None,
            str(p1.engine),
            int(p1.simulations),
            float(p1.temperature),
            str(p1.model_path) if p1.model_path else None,
        )


def _create_classic_backend(cpp_module_name: str) -> Any:
    return ClassicCppTrainingBackend(importlib.import_module(cpp_module_name))


def _create_search_options_backend(cpp_module_name: str) -> Any:
    return SearchOptionsCppTrainingBackend(
        importlib.import_module(cpp_module_name),
        module_name=cpp_module_name,
    )


def _create_flexible_search_options_backend(cpp_module_name: str) -> Any:
    return FlexibleSearchOptionsCppTrainingBackend(
        importlib.import_module(cpp_module_name),
        module_name=cpp_module_name,
    )


def build_classic_backend_factory(cpp_module_name: str) -> Callable[[], Any]:
    return partial(_create_classic_backend, cpp_module_name)


def build_search_options_backend_factory(cpp_module_name: str) -> Callable[[], Any]:
    return partial(_create_search_options_backend, cpp_module_name)


def build_flexible_search_options_backend_factory(cpp_module_name: str) -> Callable[[], Any]:
    return partial(_create_flexible_search_options_backend, cpp_module_name)

