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

A deterministic OpenEnv-style environment for customer support triage.

Agents perform realistic operations such as classification, priority assignment, team routing, queue reordering, escalation, response drafting, and ticket resolution.

## What this repository includes

- Environment runtime with `reset`, `step`, and `state` APIs
- Deterministic task definitions (`easy`, `medium`, `hard`)
- Reward shaping and deterministic graders
- Baseline OpenAI-compatible inference runner
- Root submission entrypoint `inference.py`
- Gradio app for local and Hugging Face Spaces usage

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick environment check

```bash
python -c "from openenv_support.env import CustomerSupportEnv; env=CustomerSupportEnv(); obs=env.reset('easy'); print(obs.model_dump())"
```

## Run tests

```bash
pytest -q
```

## Validate metadata

```bash
openenv validate openenv.yaml
```

If `openenv` CLI is not installed in your environment, install it first using your preferred package manager.

## Baseline inference

The baseline uses the OpenAI client with OpenAI-compatible endpoints.

Environment variables:

- `HF_TOKEN` (optional API key)
- `API_BASE_URL` (optional, default: `https://router.huggingface.co/v1`)
- `MODEL_NAME` (optional, default: `Qwen/Qwen2.5-72B-Instruct`)

```bash
export HF_TOKEN=your_key_here  # Windows PowerShell: $env:HF_TOKEN="your_key_here"
export API_BASE_URL=https://router.huggingface.co/v1  # optional
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct  # optional
python -m baseline.run_baseline --episodes 3 --max-steps 64
```

## Submission inference

Run the root submission script:

```bash
python inference.py
```

It emits the required line-based logs in this order:

- `[START]`
- `[STEP]` (one per `env.step`)
- `[END]`

## Hugging Face Spaces deployment

This repository is container-ready for HF Spaces (`sdk: docker`):

1. Push repository to a new Space
2. Ensure `Dockerfile` is at repository root
3. Add `HF_TOKEN` as a Space secret if you use a remote model provider
4. Space app entrypoint runs `python -m spaces.app`

## Docker

Build and run locally:

```bash
docker build -t openenv-support .
docker run --rm -p 7860:7860 openenv-support
```

Then open `http://localhost:7860`.

## Project structure

- `openenv_support/env.py`: environment runtime (`reset`, `step`, `state`)
- `openenv_support/models.py`: typed Pydantic models
- `openenv_support/tasks/definitions.py`: deterministic task fixtures
- `openenv_support/graders/task_graders.py`: deterministic score graders
- `openenv_support/reward.py`: reward shaping
- `baseline/run_baseline.py`: OpenAI baseline evaluation
- `baseline/inference.py`: baseline action generation
- `inference.py`: submission entrypoint
- `spaces/app.py`: HF Spaces app
- `openenv.yaml`: OpenEnv metadata
- `tests/`: API and determinism tests
