from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .config import TrainJobConfig

TrainerResultDict = dict[str, Any]


class ITrainer(Protocol):
    """Normalized trainer contract used by general/train pipeline.

    Game trainers (for example Azul/Splendor torch trainers) should return a
    dict-like payload so the pipeline can merge fields directly into
    `train_summary` without adapter glue.
    """

    def run(
        self,
        config: TrainJobConfig,
        artifacts_dir: Path,
        *,
        resume_checkpoint_path: str | None = None,
        step_index: int | None = None,
        total_steps: int | None = None,
        incremental_samples: list[dict[str, Any]] | None = None,
    ) -> TrainerResultDict: ...

