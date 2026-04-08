from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    Reward,
    StepInfo,
    TaskSpec,
    TicketAgentState,
    TicketCategory,
    TicketGroundTruth,
    TicketObservation,
    TicketPriority,
    SupportTeam,
)
from .reward import compute_reward
from .tasks import get_task


class CustomerSupportEnv:
    """OpenEnv-style environment for customer support triage workflows."""

    def __init__(self, task_id: str = "easy", seed: int = 7):
        self.seed = seed
        self.task: TaskSpec = get_task(task_id)
        self._state: EnvironmentState = self._build_initial_state(self.task)

    def _build_initial_state(self, task: TaskSpec) -> EnvironmentState:
        tickets = {
            t.ticket_id: TicketAgentState(ticket_id=t.ticket_id)
            for t in task.tickets
        }
        ground_truth = {t.ticket_id: t for t in task.tickets}
        queue_order = [t.ticket_id for t in task.tickets]
        return EnvironmentState(
            task_id=task.task_id,
            step_count=0,
            max_steps=task.max_steps,
            queue_order=queue_order,
            ground_truth=ground_truth,
            tickets=tickets,
            total_reward=0.0,
            done=False,
            action_history=[],
        )

    def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed = seed
        if task_id is not None:
            self.task = get_task(task_id)
        self._state = self._build_initial_state(self.task)
        return self._observation()

    def state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state.done:
            obs = self._observation()
            reward = Reward(value=0.0, shaped_delta=0.0, task_score=0.0, penalties={"done": -0.0})
            return obs, reward, True, {"message": "Episode already complete"}

        prev_state = self.state()
        invalid_action = False
        message = ""

        if action.action_type == ActionType.noop:
            message = "No-op action received"
        else:
            invalid_action, message = self._apply_action(action)

        self._state.step_count += 1
        signature = self._action_signature(action)
        self._state.action_history.append(signature)

        repeated_action = len(self._state.action_history) >= 3 and len(
            set(self._state.action_history[-3:])
        ) == 1

        if self._state.step_count >= self._state.max_steps:
            self._state.done = True

        if all(t.resolved for t in self._state.tickets.values()):
            self._state.done = True

        signal = compute_reward(
            previous_state=prev_state,
            current_state=self._state,
            task=self.task,
            invalid_action=invalid_action,
            repeated_action=repeated_action,
        )

        self._state.total_reward += signal.shaped_delta

        norm_total = max(0.0, min(1.0, signal.score + self._state.total_reward * 0.1))
        reward = Reward(
            value=norm_total,
            shaped_delta=signal.shaped_delta,
            task_score=signal.score,
            penalties=signal.penalties,
        )

        info = StepInfo(
            task_score=signal.score,
            penalties=signal.penalties,
            invalid_action=invalid_action,
            message=message,
        )
        return self._observation(), reward, self._state.done, info.model_dump()

    def _action_signature(self, action: Action) -> str:
        return "|".join(
            [
                action.action_type.value,
                action.ticket_id or "",
                action.value or "",
                str(action.target_index if action.target_index is not None else ""),
            ]
        )

    def _apply_action(self, action: Action) -> Tuple[bool, str]:
        if action.action_type == ActionType.reorder_queue:
            return self._reorder_queue(action)

        if not action.ticket_id or action.ticket_id not in self._state.tickets:
            return True, "ticket_id is required for this action"

        ticket = self._state.tickets[action.ticket_id]
        truth = self._state.ground_truth[action.ticket_id]

        if action.action_type == ActionType.classify:
            if not action.value:
                return True, "Missing category value"
            try:
                ticket.predicted_category = TicketCategory(action.value)
            except ValueError:
                return True, "Invalid category value"
            return False, "Category updated"

        if action.action_type == ActionType.set_priority:
            if not action.value:
                return True, "Missing priority value"
            try:
                ticket.predicted_priority = TicketPriority(action.value)
            except ValueError:
                return True, "Invalid priority value"
            return False, "Priority updated"

        if action.action_type == ActionType.assign_team:
            if not action.value:
                return True, "Missing team value"
            try:
                ticket.predicted_team = SupportTeam(action.value)
            except ValueError:
                return True, "Invalid team value"
            return False, "Team assigned"

        if action.action_type == ActionType.escalate:
            ticket.escalated = True
            return False, "Ticket escalated"

        if action.action_type == ActionType.draft_response:
            if not action.response_text:
                return True, "response_text required"
            ticket.drafted_response = action.response_text.strip()
            return False, "Response drafted"

        if action.action_type == ActionType.resolve:
            if (
                ticket.predicted_category is None
                or ticket.predicted_priority is None
                or ticket.predicted_team is None
            ):
                return True, "Cannot resolve before triage fields are set"
            if truth.requires_escalation and not ticket.escalated:
                return True, "Ticket requires escalation before resolution"
            ticket.resolved = True
            return False, "Ticket resolved"

        return True, "Unsupported action"

    def _reorder_queue(self, action: Action) -> Tuple[bool, str]:
        if not action.ticket_id or action.ticket_id not in self._state.queue_order:
            return True, "ticket_id missing or not found in queue"
        if action.target_index is None:
            return True, "target_index required for reorder"
        if action.target_index < 0 or action.target_index >= len(self._state.queue_order):
            return True, "target_index out of bounds"

        current = self._state.queue_order.index(action.ticket_id)
        if current == action.target_index:
            return False, "Queue unchanged"

        self._state.queue_order.pop(current)
        self._state.queue_order.insert(action.target_index, action.ticket_id)
        return False, "Queue reordered"

    def _observation(self) -> Observation:
        tickets: List[TicketObservation] = []
        for tid in self._state.queue_order:
            truth: TicketGroundTruth = self._state.ground_truth[tid]
            state = self._state.tickets[tid]
            tickets.append(
                TicketObservation(
                    ticket_id=tid,
                    subject=truth.subject,
                    customer_message=truth.customer_message,
                    predicted_category=state.predicted_category,
                    predicted_priority=state.predicted_priority,
                    predicted_team=state.predicted_team,
                    escalated=state.escalated,
                    resolved=state.resolved,
                )
            )

        return Observation(
            task_id=self._state.task_id,
            step_count=self._state.step_count,
            remaining_steps=self._state.max_steps - self._state.step_count,
            queue_order=self._state.queue_order,
            tickets=tickets,
            available_actions=[
                ActionType.classify,
                ActionType.set_priority,
                ActionType.assign_team,
                ActionType.reorder_queue,
                ActionType.escalate,
                ActionType.draft_response,
                ActionType.resolve,
                ActionType.noop,
            ],
            instruction=(
                "Complete ticket triage and resolution safely. Maximize grader score while "
                "avoiding invalid or repetitive actions."
            ),
        )
