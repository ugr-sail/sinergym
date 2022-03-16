import pytest

from sinergym.utils.controllers import RuleBasedController


@pytest.mark.parametrize(
    'env_name',
    [
        (
            'env_demo_discrete'
        ),
        (
            'env_demo_continuous'
        ),
    ]
)
def test_rule_based_controller(env_name, request):
    env = request.getfixturevalue(env_name)
    rule_based_agent = RuleBasedController(env)
    obs = env.reset()

    for i in range(3):
        action = rule_based_agent.act(obs)
        assert isinstance(action, tuple)
        for value in action:
            assert value is not None
        obs, reward, done, info = env.step(action)

        assert tuple(info['action_']) == action

    env.close()
