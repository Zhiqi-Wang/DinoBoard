from __future__ import annotations

from .config import MctsSchedule


def resolve_simulations(
    training_step_index: int,
    default_simulations: int,
    schedule: MctsSchedule | None,
    total_training_steps: int,
) -> int:
    """Resolve MCTS simulation count for a fixed linear schedule range.

    - start step is always 0
    - end step is always ``total_training_steps``
    """
    if schedule is None:
        return max(1, int(default_simulations))
    if schedule.type != "linear":
        return max(1, int(default_simulations))
    if training_step_index <= 0:
        return max(1, int(schedule.start_simulations))
    span = max(1, int(total_training_steps) - 1)
    if training_step_index >= span:
        return max(1, int(schedule.end_simulations))
    ratio = float(training_step_index) / float(span)
    sims = schedule.start_simulations + ratio * (schedule.end_simulations - schedule.start_simulations)
    return max(1, int(round(sims)))
