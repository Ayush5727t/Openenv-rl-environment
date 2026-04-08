"""
Submission inference entrypoint.

This script follows the required stdout contract:
  [START] ...
  [STEP] ...
  [END] ...
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from openai import OpenAI

from baseline.prompts import system_prompt, user_prompt
from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "customer-support-triage-v1")
MAX_STEPS = int(os.getenv("MAX_STEPS", "64"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_client() -> OpenAI:
    # Keep credentials optional for local/open endpoints.
    # OpenAI client requires a non-empty api_key value, so use a non-secret placeholder.
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-auth")


def request_action(client: OpenAI, observation_json_prompt: str) -> Action:
    response = client.chat.completions.create(
        model=MODEL_NAME,
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


def action_to_str(action: Action) -> str:
    payload = action.model_dump(exclude_none=True)
    return json.dumps(payload, separators=(",", ":"))


def maybe_close_env(env: CustomerSupportEnv) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def main() -> None:
    env = CustomerSupportEnv(task_id=TASK_NAME)
    client = build_client()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=TASK_NAME)

        for step in range(1, MAX_STEPS + 1):
            prompt = user_prompt(observation)
            action = request_action(client=client, observation_json_prompt=prompt)
            observation, reward, done, info = env.step(action)

            step_reward = float(reward.value)
            rewards.append(step_reward)
            steps_taken = step

            error = info.get("message") if info.get("invalid_action") else None
            log_step(
                step=step,
                action=action_to_str(action),
                reward=step_reward,
                done=done,
                error=error,
            )

            score = float(info.get("task_score", score))
            score = max(0.0, min(1.0, score))
            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Keep strict output format: emit END in finally, without synthetic STEP lines.
        success = False
        if not rewards:
            score = 0.0
        _ = exc

    finally:
        maybe_close_env(env)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
