import os
from random import randint, sample

import gymnasium as gym
import pytest

from sinergym.utils.constants import *
from sinergym.utils.env_checker import check_env


@pytest.mark.parametrize('env_name',
                         [('env_demo'),
                          ('env_demo_continuous'),
                          ('env_demo_continuous_stochastic')
                          ])
def test_reset(env_name, request):
    env = request.getfixturevalue(env_name)
    obs, info = env.reset()
    # obs check
    assert len(obs) == len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + \
        4  # year, month, day and hour
    assert env.simulator._episode_existed
    # info check
    assert isinstance(info, dict)
    assert len(info) > 0
    # default_options check
    if 'stochastic' not in env_name:
        assert not env.default_options.get('weather_variability')
    else:
        assert isinstance(env.default_options['weather_variability'], tuple)


def test_reset_custom_options(env_demo_continuous_stochastic):
    assert env_demo_continuous_stochastic.default_options['weather_variability'] == (
        1.0, 0.0, 0.001)
    custom_options = {'weather_variability': (1.1, 0.1, 0.002)}
    obs, info = env_demo_continuous_stochastic.reset(options=custom_options)
    # Check if epw with new variation is overwriting default options
    assert os.path.isfile(
        info['eplus_working_dir'] +
        '/USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3_Random_1.1_0.1_0.002.epw')


@pytest.mark.parametrize('env_name',
                         [('env_demo'),
                          ('env_demo_continuous'),
                          ])
def test_step(env_name, request):
    env = request.getfixturevalue(env_name)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + \
        4  # year, month, day and hour
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 1
    assert info['time_elapsed'] == env.simulator._eplus_run_stepsize * \
        info['timestep']

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + \
        4  # year, month, day and hour
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 2
    assert info['time_elapsed'] == env.simulator._eplus_run_stepsize * \
        info['timestep']


def test_close(env_demo):
    env_demo.reset()
    assert env_demo.simulator._episode_existed
    env_demo.close()
    assert not env_demo.simulator._episode_existed
    assert env_demo.simulator._conn is None


def test_render(env_demo):
    env_demo.render()


def test_get_schedulers(env_demo):
    # Check if generate a dictionary
    assert isinstance(env_demo.get_schedulers(), dict)
    # Check if specifying a path, generate an excel
    env_demo.get_schedulers(path='./TESTschedulers.xlsx')
    assert os.path.isfile('./TESTschedulers.xlsx')


def test_get_zones(env_demo):
    zones = env_demo.get_zones()
    assert isinstance(zones, list)
    assert len(zones) > 0


def test_get_action(env_demo):
    # Here is checked special cases
    # int
    action = randint(0, 9)
    _action = env_demo._get_action(action)
    assert isinstance(_action, list)
    assert len(_action) == 2
    # [int]
    action = [randint(0, 9)]
    _action = env_demo._get_action(action)
    assert isinstance(_action, list)
    assert len(_action) == 2
    # custom discrete action (without mapping)
    action = (22.0, 20.0)
    _action = env_demo._get_action(action)
    assert isinstance(_action, list)
    assert _action == [22.0, 20.0]
    # np.ndarray
    action = np.array([randint(0, 9)])
    _action = env_demo._get_action(action)
    assert isinstance(_action, list)
    assert len(_action) == 2
    # Not supported action
    action = 'fbsufb'
    with pytest.raises(RuntimeError):
        env_demo._get_action(action)


def test_all_environments():

    envs_id = [env_id for env_id in gym.envs.registration.registry.keys(
    ) if env_id.startswith('Eplus')]
    # Select 10 environments randomly (test would be too large)
    samples_id = sample(envs_id, 5)
    for env_id in samples_id:
        # Create env with TEST name
        env = gym.make(env_id)

        check_env(env)

        # Rename directory with name TEST for future remove
        os.rename(env.simulator._env_working_dir_parent, 'Eplus-env-TEST' +
                  env.simulator._env_working_dir_parent.split('/')[-1])

        env.close()
