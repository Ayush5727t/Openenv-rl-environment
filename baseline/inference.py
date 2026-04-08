from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from openenv_support.models import Action
from .prompts import system_prompt, user_prompt


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def request_action(
    client: OpenAI,
    model: str,
    observation_json_prompt: str,
) -> Action:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": observation_json_prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = {"action_type": "noop"}

    try:
        return Action(**payload)
    except Exception:
        return Action(action_type="noop")


def observation_to_prompt(observation) -> str:
    return user_prompt(observation)
