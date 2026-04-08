from __future__ import annotations

import json
from typing import Dict

try:
    import gradio as gr
except ImportError:  # pragma: no cover - deployment image installs gradio
    gr = None

from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action


def _heuristic_action(env: CustomerSupportEnv):
    state = env.state()
    task = env.task

    for ticket in task.tickets:
        ts = state.tickets[ticket.ticket_id]
        if ts.predicted_category is None:
            return Action(action_type="classify", ticket_id=ticket.ticket_id, value=ticket.correct_category.value)
        if ts.predicted_priority is None:
            return Action(action_type="set_priority", ticket_id=ticket.ticket_id, value=ticket.correct_priority.value)
        if ts.predicted_team is None:
            return Action(action_type="assign_team", ticket_id=ticket.ticket_id, value=ticket.correct_team.value)
        if ticket.requires_escalation and not ts.escalated:
            return Action(action_type="escalate", ticket_id=ticket.ticket_id)
        if task.task_id == "hard" and not ts.drafted_response:
            text = "We started investigation and will provide a mitigation update."
            return Action(action_type="draft_response", ticket_id=ticket.ticket_id, response_text=text)
        if not ts.resolved:
            return Action(action_type="resolve", ticket_id=ticket.ticket_id)

    return Action(action_type="noop")


def run_task(task_id: str, max_steps: int) -> Dict[str, float]:
    env = CustomerSupportEnv(task_id=task_id)
    env.reset(task_id)

    final_score = 0.0
    cumulative_delta = 0.0

    for _ in range(max_steps):
        action = _heuristic_action(env)
        _, reward, done, info = env.step(action)
        final_score = info.get("task_score", final_score)
        cumulative_delta += reward.shaped_delta
        if done:
            break

    return {
        "task_id": task_id,
        "final_task_score": round(final_score, 4),
        "trajectory_reward": round(cumulative_delta, 4),
        "steps": env.state().step_count,
    }


def ui_run(task_id: str, max_steps: int) -> str:
    result = run_task(task_id=task_id, max_steps=max_steps)
    return json.dumps(result, indent=2)


if gr is not None:
    with gr.Blocks(title="OpenEnv Customer Support Triage") as demo:
        gr.Markdown("# OpenEnv: Customer Support Triage")
        gr.Markdown("Run a deterministic heuristic baseline across easy/medium/hard tasks.")

        task_id = gr.Dropdown(choices=["easy", "medium", "hard"], value="easy", label="Task")
        max_steps = gr.Slider(minimum=8, maximum=80, value=32, step=1, label="Max steps")
        run_btn = gr.Button("Run")
        output = gr.Code(label="Result", language="json")

        run_btn.click(fn=ui_run, inputs=[task_id, max_steps], outputs=output)
else:
    demo = None


if __name__ == "__main__":
    if demo is None:
        print("Gradio is not installed in this environment.")
    else:
        demo.launch(server_name="0.0.0.0", server_port=7860)
