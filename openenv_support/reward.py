from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .graders import grade_state
from .models import EnvironmentState, TaskSpec


@dataclass
class RewardSignal:
    shaped_delta: float
    penalties: Dict[str, float]
    score: float


def compute_reward(
    previous_state: EnvironmentState,
    current_state: EnvironmentState,
    task: TaskSpec,
    invalid_action: bool,
    repeated_action: bool,
) -> RewardSignal:
    previous_score = grade_state(previous_state, task)
    current_score = grade_state(current_state, task)
    progress_delta = current_score - previous_score

    penalties: Dict[str, float] = {}
    penalty_total = 0.0

    if invalid_action:
        penalties["invalid_action"] = -0.03
        penalty_total += 0.03

    if repeated_action:
        penalties["looping"] = -0.02
        penalty_total += 0.02

    if current_state.step_count >= current_state.max_steps and not current_state.done:
        penalties["timeout"] = -0.05
        penalty_total += 0.05

    shaped_delta = max(-0.20, min(0.20, progress_delta - penalty_total))
    return RewardSignal(shaped_delta=shaped_delta, penalties=penalties, score=current_score)
