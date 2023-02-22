import pytest


def test_base_reward(base_reward):
    with pytest.raises(NotImplementedError):
        base_reward()


@pytest.mark.parametrize('reward_name',
                         [('linear_reward'),
                          ('exponential_reward'),
                          ('hourly_linear_reward'),
                          ])
def test_rewards(reward_name, request):
    reward = request.getfixturevalue(reward_name)
    env = reward.env
    env.reset()
    a = env.action_space.sample()
    env.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    R, terms = reward()
    assert R <= 0
    assert env.reward_fn.W_energy == 0.5
    assert isinstance(terms, dict)
    assert len(terms) > 0


def test_custom_reward(custom_reward):
    env = custom_reward.env
    env.reset()
    a = env.action_space.sample()
    env.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    R, terms = custom_reward()
    assert R == -1.0
    assert isinstance(terms, dict)
    assert len(terms) == 0
