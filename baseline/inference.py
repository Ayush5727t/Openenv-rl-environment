from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from openenv_support.models import Action
from .prompts import system_prompt, user_prompt


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"


def build_client() -> OpenAI:
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    # Keep credentials optional for local/open endpoints.
    # OpenAI client requires a non-empty api_key value, so use a non-secret placeholder.
    return OpenAI(base_url=API_BASE_URL, api_key=api_key or "no-auth")


def request_action(
    client: OpenAI,
    model: Optional[str],
    observation_json_prompt: str,
) -> Action:
    selected_model = model or MODEL_NAME
    response = client.chat.completions.create(
        model=selected_model,
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
