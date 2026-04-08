from __future__ import annotations

from typing import Dict

from openenv_support.models import Observation


def system_prompt() -> str:
    return (
        "You are an operations agent for customer support triage. "
        "Return exactly one JSON object with fields: "
        "action_type, ticket_id, value, target_index, response_text. "
        "Only output valid actions and keep text concise."
    )


def user_prompt(observation: Observation) -> str:
    return (
        "Current observation as JSON:\n"
        f"{observation.model_dump_json(indent=2)}\n\n"
        "Choose the best next action to maximize final score."
    )
