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

    # Do an entire episode to manage different hours and seasons
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
    with pytest.raises(KeyError):
        reward(obs_dict)


@pytest.mark.parametrize('reward_name,env_name',
                         [('linear_reward', 'env_demo_summer'),
                          ('energy_cost_linear_reward',
                           'env_demo_summer_energy_cost'),
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
    with pytest.raises(KeyError):
        reward(obs_dict)


def test_custom_reward(custom_reward):
    R, terms = custom_reward()
    assert R == -1.0
    assert isinstance(terms, dict)
    assert len(terms) == 0


def test_multizone_reward(multizone_reward):
    # Fictional observation variables (threshold +/-0.5)
    obs_dict = {'air_temperature1': 20.3,
                'air_temperature2': 21.6,
                'setpoint_temperature1': 20.0,
                'setpoint_temperature2': 21.0,
                'HVAC_electricity_demand_rate': 0}
    R, terms = multizone_reward(obs_dict)
    assert round(R, 2) == -0.05  # 0.5 * 0.1
    assert isinstance(terms, dict)
    # Diferrent setpoints (threshold +/-1.0)
    multizone_reward.comfort_threshold = 1.0
    obs_dict = {'air_temperature1': 21.2,
                'air_temperature2': 19.7,
                'setpoint_temperature1': 19.0,
                'setpoint_temperature2': 22.0,
                'HVAC_electricity_demand_rate': 0}
    R, terms = multizone_reward(obs_dict)
    assert round(R, 2) == -1.25  # 0.5 * (1.2 + 1.3)
    assert isinstance(terms, dict)

    # Tests exceptions
    multizone_reward.comfort_threshold = 0.5
    # Forcing unknown reward temp variables
    obs_dict = {'unknown': 20.3,
                'air_temperature2': 21.6,
                'setpoint_temperature1': 20.0,
                'setpoint_temperature2': 21.0,
                'HVAC_electricity_demand_rate': 0}
    with pytest.raises(KeyError):
        multizone_reward(obs_dict)
    # Forcing unknown reward setpoint variable
    obs_dict = {'air_temperature1': 20.3,
                'air_temperature2': 21.6,
                'unknown': 20.0,
                'setpoint_temperature2': 21.0,
                'HVAC_electricity_demand_rate': 0}
    with pytest.raises(KeyError):
        multizone_reward(obs_dict)
    # Forcing unknown reward energy variable
    obs_dict = {'air_temperature1': 20.3,
                'air_temperature2': 21.6,
                'setpoint_temperature1': 20.0,
                'setpoint_temperature2': 21.0,
                'unknown': 0}
    with pytest.raises(KeyError):
        multizone_reward(obs_dict)
