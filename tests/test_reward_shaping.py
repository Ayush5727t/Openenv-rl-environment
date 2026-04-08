from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action


def test_reward_increases_with_partial_progress() -> None:
    env = CustomerSupportEnv(task_id="easy")
    env.reset("easy")

    _, reward_bad, _, _ = env.step(Action(action_type="noop"))
    _, reward_good, _, _ = env.step(
        Action(action_type="classify", ticket_id="T-100", value="billing")
    )

    assert reward_good.task_score >= reward_bad.task_score
