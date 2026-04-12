from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

from setuptools import Extension
from setuptools.command.build_ext import build_ext


class WindowsNoLtcgBuildExt(build_ext):
    """Avoid MSVC LTCG/whole-program optimization ICEs in debug-service wheels."""

    @staticmethod
    def _strip_flags(flags: Iterable[str] | None, blocked: set[str]) -> list[str]:
        return [arg for arg in list(flags or []) if arg.upper() not in blocked]

    @staticmethod
    def _switch_hostx86_to_hostx64(cmd: str) -> str:
        return re.sub(r"HostX86([/\\\\])x64", r"HostX64\1x64", cmd, flags=re.IGNORECASE)

    @classmethod
    def _switch_command_to_hostx64(cls, value):
        if isinstance(value, str):
            return cls._switch_hostx86_to_hostx64(value)
        if isinstance(value, (list, tuple)) and value:
            first = value[0]
            if isinstance(first, str):
                switched = cls._switch_hostx86_to_hostx64(first)
                if switched != first:
                    if isinstance(value, tuple):
                        value = list(value)
                    value[0] = switched
            return value
        return value

    def _prefer_hostx64_msvc_tools(self) -> None:
        if not sys.platform.startswith("win"):
            return

        attr_names = (
            "cc",
            "linker",
            "linker_so",
            "linker_exe",
            "lib",
            "rc",
            "mc",
        )
        for attr_name in attr_names:
            current = getattr(self.compiler, attr_name, None)
            if current is not None:
                setattr(self.compiler, attr_name, self._switch_command_to_hostx64(current))

        executables = getattr(self.compiler, "executables", None)
        if isinstance(executables, dict):
            for key, value in list(executables.items()):
                executables[key] = self._switch_command_to_hostx64(value)

    def build_extensions(self) -> None:
        if sys.platform.startswith("win"):
            if hasattr(self.compiler, "initialize") and not getattr(self.compiler, "initialized", False):
                self.compiler.initialize()

            # Prefer 64-bit hosted MSVC tools to reduce random C1001 ICEs on large C++ TUs.
            self._prefer_hostx64_msvc_tools()

            for ext in self.extensions:
                extra_compile_args = list(getattr(ext, "extra_compile_args", []) or [])
                ext.extra_compile_args = [arg for arg in extra_compile_args if arg.upper() != "/GL"]

                extra_link_args = list(getattr(ext, "extra_link_args", []) or [])
                filtered_link_args = self._strip_flags(extra_link_args, {"/LTCG", "/LTCG:INCREMENTAL", "/GL"})
                ext.extra_link_args = filtered_link_args + ["/LTCG:OFF"]

            compile_opts = getattr(self.compiler, "compile_options", None)
            if compile_opts is not None:
                stripped = self._strip_flags(compile_opts, {"/GL", "/O1", "/O2", "/OX", "/OT"})
                stripped.append("/Od")
                self.compiler.compile_options = stripped

            compile_opts_debug = getattr(self.compiler, "compile_options_debug", None)
            if compile_opts_debug is not None:
                self.compiler.compile_options_debug = self._strip_flags(compile_opts_debug, {"/GL"})

            link_flag_attrs = (
                "ldflags_shared",
                "ldflags_shared_debug",
                "ldflags_exe",
                "ldflags_exe_debug",
                "ldflags_static",
                "ldflags_static_debug",
            )
            for attr_name in link_flag_attrs:
                attr_value = getattr(self.compiler, attr_name, None)
                if attr_value is not None:
                    stripped = self._strip_flags(attr_value, {"/LTCG", "/LTCG:INCREMENTAL"})
                    if "/LTCG:OFF" not in {arg.upper() for arg in stripped}:
                        stripped.append("/LTCG:OFF")
                    setattr(self.compiler, attr_name, stripped)

            ldflags_map = getattr(self.compiler, "_ldflags", None)
            if isinstance(ldflags_map, dict):
                for key, value in list(ldflags_map.items()):
                    stripped = self._strip_flags(value, {"/LTCG", "/LTCG:INCREMENTAL"})
                    if "/LTCG:OFF" not in {arg.upper() for arg in stripped}:
                        stripped.append("/LTCG:OFF")
                    ldflags_map[key] = stripped

        super().build_extensions()


def build_cpp_extension(
    *,
    root: Path,
    game: str,
    extension_name: str,
    module_source: str,
    game_sources: Iterable[str],
    extra_general_sources: Iterable[Path] = (),
) -> Extension:
    if sys.platform.startswith("win"):
        extra_compile_args = ["/std:c++17", "/Od", "/Ob0"]
        extra_link_args = ["/LTCG:OFF"]
    else:
        extra_compile_args = ["-std=c++17", "-O3"]
        extra_link_args = []
    with_onnx = os.environ.get("BOARD_AI_WITH_ONNX", "0") == "1"
    onnx_root = os.environ.get("BOARD_AI_ONNXRUNTIME_ROOT", "")

    include_dirs = [
        str(root / "general" / "include"),
        str(root / "games" / game / "include"),
    ]
    library_dirs: list[str] = []
    libraries: list[str] = []
    define_macros = [("BOARD_AI_WITH_ONNX", "1" if with_onnx else "0")]

    if with_onnx:
        if not onnx_root:
            raise RuntimeError("BOARD_AI_WITH_ONNX=1 requires BOARD_AI_ONNXRUNTIME_ROOT")
        include_dirs.append(str(Path(onnx_root) / "include"))
        library_dirs.append(str(Path(onnx_root) / "lib"))
        libraries.append("onnxruntime")

    sources = [module_source]
    sources.extend(str(p) for p in extra_general_sources)
    sources.extend(str(root / "games" / game / "src" / source_name) for source_name in game_sources)

    return Extension(
        name=extension_name,
        sources=sources,
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        library_dirs=library_dirs,
        libraries=libraries,
    )


def build_cpp_extension_setup_kwargs(
    *,
    root: Path,
    game: str,
    extension_name: str,
    module_source: str,
    game_sources: Iterable[str],
    package_name: str,
    package_version: str = "0.1.0",
    extra_general_sources: Iterable[Path] = (),
) -> dict[str, object]:
    ext = build_cpp_extension(
        root=root,
        game=game,
        extension_name=extension_name,
        module_source=module_source,
        game_sources=game_sources,
        extra_general_sources=extra_general_sources,
    )
    return {
        "name": package_name,
        "version": package_version,
        "ext_modules": [ext],
        "cmdclass": {"build_ext": WindowsNoLtcgBuildExt},
    }


def build_standard_game_cpp_setup_kwargs(
    *,
    root: Path,
    game: str,
    extension_name: str,
    package_name: str,
    package_version: str = "0.1.0",
    include_action_constraint: bool = False,
    module_source: str | None = None,
    game_sources: Iterable[str] | None = None,
) -> dict[str, object]:
    source_module = module_source or f"cpp_{game}_engine_module.cpp"
    source_files = tuple(
        game_sources
        or (
            f"{game}_state.cpp",
            f"{game}_rules.cpp",
            f"{game}_net_adapter.cpp",
        )
    )
    extra_general_sources = [
        root / "general" / "src" / "search" / "net_mcts.cpp",
        root / "general" / "src" / "infer" / "onnx_policy_value_evaluator.cpp",
    ]
    if include_action_constraint:
        extra_general_sources.insert(0, root / "general" / "src" / "core" / "action_constraint.cpp")
    return build_cpp_extension_setup_kwargs(
        root=root,
        game=game,
        extension_name=extension_name,
        module_source=source_module,
        game_sources=source_files,
        package_name=package_name,
        package_version=package_version,
        extra_general_sources=tuple(extra_general_sources),
    )

