from pathlib import Path
import sys

from setuptools import setup

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from general.debug.plugin_loader import build_cpp_setup_kwargs_from_current_game  # noqa: E402

setup(
    **build_cpp_setup_kwargs_from_current_game(current_file=__file__, root=ROOT)
)

