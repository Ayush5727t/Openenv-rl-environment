from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Dict, List

from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action

from .inference import build_client, observation_to_prompt, request_action


TASKS = ["easy", "medium", "hard"]


def run_episode(env: CustomerSupportEnv, model: str, task_id: str, max_steps: int) -> Dict[str, float]:
    client = build_client()
    observation = env.reset(task_id=task_id)

    final_task_score = 0.0
    cumulative_reward = 0.0

    for _ in range(max_steps):
        prompt = observation_to_prompt(observation)
        action = request_action(client=client, model=model, observation_json_prompt=prompt)
        observation, reward, done, info = env.step(action)
        final_task_score = info.get("task_score", final_task_score)
        cumulative_reward += reward.shaped_delta
        if done:
            break

    return {
        "task_score": round(final_task_score, 4),
        "trajectory_reward": round(cumulative_reward, 4),
        "steps": observation.step_count,
    }


def run_baseline(model: str, episodes: int, max_steps: int) -> Dict[str, Dict[str, float]]:
    env = CustomerSupportEnv()
    report: Dict[str, Dict[str, float]] = {}

    for task_id in TASKS:
        scores: List[float] = []
        rewards: List[float] = []
        steps: List[int] = []

        for _ in range(episodes):
            result = run_episode(env=env, model=model, task_id=task_id, max_steps=max_steps)
            scores.append(result["task_score"])
            rewards.append(result["trajectory_reward"])
            steps.append(result["steps"])

        report[task_id] = {
            "avg_task_score": round(mean(scores), 4),
            "avg_trajectory_reward": round(mean(rewards), 4),
            "avg_steps": round(mean(steps), 2),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI baseline for OpenEnv tasks")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        help="OpenAI model name (default: MODEL_NAME env var)",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--max-steps", type=int, default=64, help="Max steps per episode")
    args = parser.parse_args()

    report = run_baseline(model=args.model, episodes=args.episodes, max_steps=args.max_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
