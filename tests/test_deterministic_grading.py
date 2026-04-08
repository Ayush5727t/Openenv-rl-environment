from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action
from openenv_support.graders import grade_state


def test_grading_deterministic_for_identical_trajectory() -> None:
    env_a = CustomerSupportEnv(task_id="medium")
    env_b = CustomerSupportEnv(task_id="medium")

    env_a.reset("medium")
    env_b.reset("medium")

    actions = [
        Action(action_type="reorder_queue", ticket_id="T-200", target_index=0),
        Action(action_type="classify", ticket_id="T-200", value="account"),
        Action(action_type="set_priority", ticket_id="T-200", value="urgent"),
        Action(action_type="assign_team", ticket_id="T-200", value="account_support"),
    ]

    for action in actions:
        env_a.step(action)
        env_b.step(action)

    score_a = grade_state(env_a.state(), env_a.task)
    score_b = grade_state(env_b.state(), env_b.task)

    assert score_a == score_b
