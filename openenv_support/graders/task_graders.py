from __future__ import annotations

from typing import Dict, List

from ..models import EnvironmentState, TaskSpec, TicketPriority


PRIORITY_RANK = {
    TicketPriority.low: 1,
    TicketPriority.medium: 2,
    TicketPriority.high: 3,
    TicketPriority.urgent: 4,
}


def _queue_score(state: EnvironmentState, task: TaskSpec) -> float:
    expected = sorted(
        task.tickets,
        key=lambda t: PRIORITY_RANK[t.correct_priority],
        reverse=True,
    )
    expected_ids = [t.ticket_id for t in expected]
    current = state.queue_order
    hits = sum(1 for i, tid in enumerate(current) if i < len(expected_ids) and tid == expected_ids[i])
    return hits / max(1, len(expected_ids))


def _response_quality(text: str, required_keywords: List[str]) -> float:
    if not required_keywords:
        return 1.0
    if not text.strip():
        return 0.0
    lower = text.lower()
    hit = sum(1 for kw in required_keywords if kw.lower() in lower)
    return hit / len(required_keywords)


def grade_state(state: EnvironmentState, task: TaskSpec) -> float:
    gt = state.ground_truth
    total = 0.0
    per_ticket_scores: Dict[str, float] = {}

    for ticket_id, ticket_truth in gt.items():
        ticket_state = state.tickets[ticket_id]
        components = []

        components.append(1.0 if ticket_state.predicted_category == ticket_truth.correct_category else 0.0)
        components.append(1.0 if ticket_state.predicted_priority == ticket_truth.correct_priority else 0.0)
        components.append(1.0 if ticket_state.predicted_team == ticket_truth.correct_team else 0.0)

        if task.task_id == "hard":
            components.append(
                1.0 if ticket_state.escalated == ticket_truth.requires_escalation else 0.0
            )
            components.append(
                _response_quality(ticket_state.drafted_response, ticket_truth.required_response_keywords)
            )

        components.append(1.0 if ticket_state.resolved else 0.0)
        per_ticket_scores[ticket_id] = sum(components) / len(components)
        total += per_ticket_scores[ticket_id]

    score = total / max(1, len(gt))

    if task.task_id in {"medium", "hard"}:
        score = (score * 0.8) + (_queue_score(state, task) * 0.2)

    return max(0.0, min(1.0, score))
