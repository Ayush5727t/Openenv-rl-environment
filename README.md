---
title: OpenEnv Customer Support Triage
emoji: ""
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# OpenEnv Customer Support Triage

A real-world OpenEnv environment where agents perform customer support operations: classify tickets, set SLA priority, assign teams, reorder queue, escalate critical issues, draft safe responses, and resolve tickets.

## Why this environment

Customer support triage is a daily operational workflow in real companies. This environment provides deterministic evaluation and dense reward shaping so model improvements are measurable over full trajectories, not only final outcomes.

## OpenEnv API

The environment implements the standard API:

- `reset(task_id: Optional[str], seed: Optional[int]) -> Observation`
- `step(action: Action) -> (Observation, Reward, done: bool, info: dict)`
- `state() -> EnvironmentState`

Primary runtime class: `openenv_support.env.CustomerSupportEnv`

## Typed models

Pydantic models are defined in `openenv_support.models`:

- `Observation`
- `Action`
- `Reward`
- `EnvironmentState`

## Action space

`Action.action_type` supports:

- `classify` with `value` in: `billing`, `technical`, `account`, `abuse`
- `set_priority` with `value` in: `low`, `medium`, `high`, `urgent`
- `assign_team` with `value` in: `billing_ops`, `tech_support`, `trust_safety`, `account_support`
- `reorder_queue` with `ticket_id` and `target_index`
- `escalate` with `ticket_id`
- `draft_response` with `ticket_id` and `response_text`
- `resolve` with `ticket_id`
- `noop`

## Observation space

Each observation includes:

- task metadata (`task_id`, `step_count`, `remaining_steps`)
- queue ordering (`queue_order`)
- per-ticket state (`subject`, `customer_message`, predicted fields, escalation, resolution)
- available actions and instruction

## Tasks and difficulty

Three deterministic tasks with graders in range `0.0` to `1.0`:

1. Easy: Single ticket triage and resolution
2. Medium: Multi-ticket SLA-aware queue ordering plus triage
3. Hard: Escalation-critical triage with response quality checks

Task fixtures: `openenv_support/tasks/definitions.py`

Grader: `openenv_support/graders/task_graders.py`

## Reward design

Reward is trajectory-shaped and not purely terminal:

- positive signal for partial progress via grader score deltas
- penalties for invalid actions
- penalties for repeated looping behavior
- timeout penalty near episode end

This provides meaningful incremental learning signals while preserving final objective quality.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick run

```bash
python -c "from openenv_support.env import CustomerSupportEnv; env=CustomerSupportEnv(); obs=env.reset('easy'); print(obs.model_dump())"
```

## Validate OpenEnv metadata

```bash
openenv validate openenv.yaml
```

If `openenv` CLI is not installed in your environment, install it first using your preferred package manager.

## Baseline inference (OpenAI)

The baseline script uses the OpenAI API client and reads credentials from environment variable `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=your_key_here  # Windows PowerShell: $env:OPENAI_API_KEY="your_key_here"
python -m baseline.run_baseline --model gpt-4.1-mini --episodes 3 --max-steps 64
```

The script reports average task scores across `easy`, `medium`, and `hard` with fixed decoding parameters (`temperature=0`, `top_p=1`) for reproducible behavior.

## Test

```bash
pytest -q
```

## Hugging Face Spaces deployment

This repository is container-ready for HF Spaces (`sdk: docker`):

1. Push repository to a new Space
2. Ensure `Dockerfile` is at repository root
3. Add `OPENAI_API_KEY` as a Space secret if using OpenAI baseline
4. Space app entrypoint runs `python -m spaces.app`

## Docker

Build and run locally:

```bash
docker build -t openenv-support .
docker run --rm -p 7860:7860 openenv-support
```

Then open `http://localhost:7860`.

## Project layout

- `openenv_support/env.py`: environment runtime (`reset`, `step`, `state`)
- `openenv_support/models.py`: typed Pydantic models
- `openenv_support/tasks/definitions.py`: deterministic task fixtures
- `openenv_support/graders/task_graders.py`: deterministic score graders
- `openenv_support/reward.py`: reward shaping
- `baseline/run_baseline.py`: OpenAI baseline evaluation
- `spaces/app.py`: HF Spaces app
- `openenv.yaml`: OpenEnv metadata
- `tests/`: API and determinism tests
