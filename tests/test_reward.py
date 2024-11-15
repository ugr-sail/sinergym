import pytest


def test_base_reward(base_reward):
    with pytest.raises(NotImplementedError):
        base_reward(obs_dict={})


@pytest.mark.parametrize('reward_name,env_name',
                         [('linear_reward', 'env_demo'),
                          ('energy_cost_linear_reward', 'env_demo_energy_cost'),
                          ('exponential_reward', 'env_demo_summer'),
                          ('hourly_linear_reward', 'env_demo'),
                          ('normalized_linear_reward', 'env_demo_summer')
                          ])
def test_rewards(reward_name, env_name, request):
    reward = request.getfixturevalue(reward_name)
    env = request.getfixturevalue(env_name)
    env.reset()
    a = env.action_space.sample()
    obs, _, terminated, truncated, _ = env.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), obs))
    R, terms = reward(obs_dict)
    assert R <= 0
    if reward_name == "energy_cost_linear_reward":
        assert env.reward_fn.W_energy + env.reward_fn.W_temperature < 1
    else:
        assert env.reward_fn.W_energy == 0.5
    assert isinstance(terms, dict)
    assert len(terms) > 0

    # Do an entire episode to manage different hours and seassons
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(a)
        obs_dict = dict(
            zip(env.get_wrapper_attr('observation_variables'), obs))
        R, terms = reward(obs_dict)


@pytest.mark.parametrize('reward_name,env_name',
                         [('linear_reward', 'env_demo'),
                          ('energy_cost_linear_reward', 'env_demo_energy_cost'),
                          ('exponential_reward', 'env_demo'),
                          ('hourly_linear_reward', 'env_demo'),
                          ('normalized_linear_reward', 'env_demo')
                          ])
def test_rewards_temperature_exception(reward_name, env_name, request):
    reward = request.getfixturevalue(reward_name)
    env = request.getfixturevalue(env_name)
    env.reset()
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), obs))

    # Forcing unknown reward temp variables
    reward.temp_names.append('Unknown_temp_variable')
    with pytest.raises(AssertionError):
        reward(obs_dict)


@pytest.mark.parametrize('reward_name,env_name',
                         [('linear_reward', 'env_demo_summer'),
                          ('energy_cost_linear_reward', 'env_demo_summer_energy_cost'),
                          ('exponential_reward', 'env_demo_summer'),
                          ('hourly_linear_reward', 'env_demo_summer'),
                          ('normalized_linear_reward', 'env_demo_summer')
                          ])
def test_rewards_energy_exception(reward_name, env_name, request):
    reward = request.getfixturevalue(reward_name)
    env = request.getfixturevalue(env_name)
    env.reset()
    a = env.action_space.sample()
    obs, _, _, _, _ = env.step(a)
    # Such as env has been created separately, it is important to calculate
    # specifically in reward class.
    obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), obs))

    # Forcing unknown energy temp variables
    reward.energy_names.append('Unknown_energy_variable')
    with pytest.raises(AssertionError):
        reward(obs_dict)


def test_custom_reward(custom_reward):
    R, terms = custom_reward()
    assert R == -1.0
    assert isinstance(terms, dict)
    assert len(terms) == 0
