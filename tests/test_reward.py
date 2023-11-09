import pytest


def test_base_reward(base_reward):
    with pytest.raises(NotImplementedError):
        base_reward(obs_dict={})


@pytest.mark.parametrize('reward_name',
                         [('linear_reward'),
                          ('exponential_reward'),
                          ('hourly_linear_reward'),
                          ])
def test_rewards(reward_name, env_5zone, request):
    reward = request.getfixturevalue(reward_name)
    env_5zone.reset()
    a = env_5zone.action_space.sample()
    obs, _, _, _, _ = env_5zone.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    obs_dict = dict(zip(env_5zone.observation_variables, obs))
    R, terms = reward(obs_dict)
    assert R <= 0
    assert env_5zone.reward_fn.W_energy == 0.5
    assert isinstance(terms, dict)
    assert len(terms) > 0


def test_custom_reward(custom_reward):
    R, terms = custom_reward()
    assert R == -1.0
    assert isinstance(terms, dict)
    assert len(terms) == 0
