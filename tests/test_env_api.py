from openenv_support.env import CustomerSupportEnv
from openenv_support.models import Action


def test_reset_step_state_contract() -> None:
    env = CustomerSupportEnv(task_id="easy")
    obs = env.reset("easy")

    assert obs.task_id == "easy"
    assert obs.step_count == 0
    assert len(obs.tickets) == 1

    action = Action(action_type="classify", ticket_id="T-100", value="billing")
    next_obs, reward, done, info = env.step(action)

    assert next_obs.step_count == 1
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(done, bool)
    assert "task_score" in info

    state = env.state()
    assert state.step_count == 1
    assert state.task_id == "easy"
