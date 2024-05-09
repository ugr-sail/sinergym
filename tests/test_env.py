import os
from random import randint, sample

import gymnasium as gym
import pytest

from sinergym.utils.constants import *
from sinergym.utils.env_checker import check_env


@pytest.mark.parametrize('env_name',
                         [('env_5zone'),
                          ('env_5zone_stochastic')
                          ])
def test_reset(env_name, request):
    env = request.getfixturevalue(env_name)
    # Check state before reset
    assert env.get_wrapper_attr('episode') == 0
    assert env.get_wrapper_attr(
        'energyplus_simulator').energyplus_state is None
    obs, info = env.reset()
    # Check after reset
    assert env.get_wrapper_attr('episode') == 1
    assert env.get_wrapper_attr(
        'energyplus_simulator').energyplus_state is not None
    assert len(obs) == len(env.get_wrapper_attr('time_variables')) + len(env.get_wrapper_attr(
        'variables')) + len(env.get_wrapper_attr('meters'))  # year, month, day and hour
    assert isinstance(info, dict)
    assert len(info) > 0
    # default_options check
    if 'stochastic' not in env_name:
        assert not env.get_wrapper_attr('default_options').get(
            'weather_variability', False)
    else:
        assert isinstance(env.get_wrapper_attr('default_options')[
                          'weather_variability'], tuple)


def test_reset_custom_options(env_5zone_stochastic):
    assert isinstance(env_5zone_stochastic.get_wrapper_attr(
        'default_options')['weather_variability'], tuple)
    assert len(env_5zone_stochastic.get_wrapper_attr(
        'default_options')['weather_variability']) == 3
    custom_options = {'weather_variability': (1.1, 0.1, 0.002)}
    env_5zone_stochastic.reset(options=custom_options)
    # Check if epw with new variation is overwriting default options
    weather_path = env_5zone_stochastic.model._weather_path
    weather_file = weather_path.split('/')[-1][:-4]
    assert os.path.isfile(
        env_5zone_stochastic.episode_path +
        '/' +
        weather_file +
        '_Random_1.1_0.1_0.002.epw')


def test_step(env_5zone):
    env = env_5zone
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == env.observation_space.shape[0]
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 2
    old_time_elapsed = info['time_elapsed(hours)']
    assert old_time_elapsed > 0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == env.observation_space.shape[0]
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 3
    assert info['time_elapsed(hours)'] > old_time_elapsed

    # Not supported action

    # action = 'fbsufb'
    # with pytest.raises(Exception):
    #     env.step(action)


def test_close(env_5zone):
    env_5zone.reset()
    assert env_5zone.is_running
    env_5zone.close()
    assert not env_5zone.is_running


def test_render(env_5zone):
    env_5zone.render()


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
        os.rename(env.get_wrapper_attr('workspace_path'), 'Eplus-env-TEST' +
                  env.get_wrapper_attr('workspace_path').split('/')[-1])

        env.close()
