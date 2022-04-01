import pytest

from sinergym.utils.controllers import RBC5Zone, RBCDatacenter


@pytest.mark.parametrize(
    'env_name',
    [
        (
            'env_demo'
        ),
        (
            'env_demo_continuous'
        ),
    ]
)
def test_rule_based_controller_5Zone(env_name, request):
    env = request.getfixturevalue(env_name)
    rule_based_agent = RBC5Zone(env)
    obs = env.reset()

    for i in range(3):
        action = rule_based_agent.act(obs)
        assert isinstance(action, tuple)
        for value in action:
            assert value is not None
        obs, reward, done, info = env.step(action)

        assert tuple(info['action_']) == action

    env.close()

def test_rule_based_controller_Datacenter(env_name, request):
    env = request.getfixturevalue(env_name)
    rule_based_agent = RBCDatacenter(env)
    obs = env.reset()

    for i in range(3):
        action = rule_based_agent.act(obs)
        assert isinstance(action, tuple)
        for value in action:
            assert value is not None
        obs, reward, done, info = env.step(action)

        assert tuple(info['action_']) == action

    env.close()
