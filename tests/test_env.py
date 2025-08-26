import os
from queue import Queue
from random import sample

import gymnasium as gym
import pytest
from gymnasium.spaces import Dict, Discrete

from sinergym.utils.constants import *
from sinergym.utils.constants import DEFAULT_5ZONE_DISCRETE_FUNCTION
from sinergym.utils.env_checker import check_env
from sinergym.utils.wrappers import DiscretizeEnv, NormalizeObservation


@pytest.mark.parametrize('env_name', [('env_5zone'), ('env_5zone_stochastic')])
def test_reset(env_name, request):
    env = request.getfixturevalue(env_name)
    # Check state before reset
    assert env.get_wrapper_attr('episode') == 0
    assert env.get_wrapper_attr('timestep') == 0
    assert env.get_wrapper_attr('energyplus_simulator').energyplus_state is None
    obs, info = env.reset()
    # Check after reset
    assert env.get_wrapper_attr('episode') == 1
    assert env.get_wrapper_attr('timestep') == 0
    assert env.get_wrapper_attr('energyplus_simulator').energyplus_state is not None
    assert len(obs) == len(env.get_wrapper_attr('time_variables')) + len(
        env.get_wrapper_attr(
            # year, month, day and hour
            'variables'
        )
    ) + len(env.get_wrapper_attr('meters'))
    assert isinstance(info, dict)
    assert len(info) > 0
    # default_options check
    if 'stochastic' not in env_name:
        assert not env.get_wrapper_attr('default_options').get(
            'weather_variability', False
        )
    else:
        assert isinstance(
            env.get_wrapper_attr('default_options')['weather_variability'], dict
        )


def test_reset_custom_options(env_5zone_stochastic):
    assert isinstance(
        env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'],
        dict,
    )
    assert (
        len(
            env_5zone_stochastic.get_wrapper_attr('default_options')[
                'weather_variability'
            ]
        )
        == 1
    )
    custom_options = {'weather_variability': {'Dry Bulb Temperature': (1.1, 0.1, 30.0)}}
    env_5zone_stochastic.reset(options=custom_options)
    # Check if epw with new variation is overwriting default options
    weather_path = env_5zone_stochastic.model._weather_path
    weather_file = weather_path.split('/')[-1][:-4]
    assert os.path.isfile(
        env_5zone_stochastic.episode_path + '/' + weather_file + '_OU_Noise.epw'
    )


def test_step(env_5zone):
    env = env_5zone
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == env.observation_space.shape[0]
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 1
    old_time_elapsed = info['time_elapsed(hours)']
    assert old_time_elapsed > 0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert len(obs) == env.observation_space.shape[0]
    assert not isinstance(reward, type(None))
    assert not terminated
    assert not truncated
    assert info['timestep'] == 2
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


def test_update_context(env_5zone):
    env_5zone.reset()
    a = env_5zone.action_space.sample()
    obs, _, _, _, _ = env_5zone.step(a)
    # Check if obs has the occupancy value of initial context
    obs_dict = dict(zip(env_5zone.get_wrapper_attr('observation_variables'), obs))
    # Context occupancy is a percentage of 20 people
    assert (
        obs_dict['people_occupant']
        == env_5zone.get_wrapper_attr('last_context')[-1] * 20
    )
    # Try to update context with a new value
    env_5zone.update_context([0.5])
    obs, _, _, _, _ = env_5zone.step(a)
    # Check if obs has the new occupancy value
    obs_dict = dict(zip(env_5zone.get_wrapper_attr('observation_variables'), obs))
    assert obs_dict['people_occupant'] == 10
    assert env_5zone.get_wrapper_attr('last_context')[-1] == 0.5


def test_reset_reproducibility():

    # Disable environment global seed
    env = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1', env_name='PYTESTGYM', seed=None
    )

    # Check that the environment is reproducible
    action1 = env.action_space.sample()
    action2 = env.action_space.sample()
    # Case 1 (seed 0)
    obs_0_reset, _ = env.reset(seed=0)
    obs_0_step1, _, _, _, _ = env.step(action1)
    obs_0_step2, _, _, _, _ = env.step(action2)
    # Case 2 (seed 0)
    obs_1_reset, _ = env.reset(seed=0)
    obs_1_step1, _, _, _, _ = env.step(action1)
    obs_1_step2, _, _, _, _ = env.step(action2)

    assert np.allclose(obs_0_reset, obs_1_reset, atol=1e-6)
    assert np.allclose(obs_0_step1, obs_1_step1, atol=1e-6)
    assert np.allclose(obs_0_step2, obs_1_step2, atol=1e-6)

    # Case 3 (seed 1)
    obs_2_reset, _ = env.reset(seed=1)
    obs_2_step1, _, _, _, _ = env.step(action1)
    obs_2_step2, _, _, _, _ = env.step(action2)

    assert not np.allclose(obs_0_reset, obs_2_reset, atol=1e-6)
    assert not np.allclose(obs_0_step1, obs_2_step1, atol=1e-6)
    assert not np.allclose(obs_0_step2, obs_2_step2, atol=1e-6)


def test_global_reproducibility():

    def _check_reset_reproducibility_with_seed(
        env: gym.Env,
    ) -> Union[List[float], bool]:
        # Check seed is available
        if env.get_wrapper_attr('seed') is not None:
            # Store environment interaction info
            action1 = env.action_space.sample()
            action2 = env.action_space.sample()

            # Set the same seed in reset to check that global seed disable
            # reset seed
            obs_0_reset_0, _ = env.reset(seed=0)
            obs_0_step1_0, _, _, _, _ = env.step(action1)
            obs_0_step2_0, _, _, _, _ = env.step(action2)

            obs_1_reset_0, _ = env.reset(seed=0)
            obs_1_step1_0, _, _, _, _ = env.step(action1)
            obs_1_step2_0, _, _, _, _ = env.step(action2)

            assert not np.allclose(obs_0_reset_0, obs_1_reset_0, atol=1e-6)
            assert not np.allclose(obs_0_step1_0, obs_1_step1_0, atol=1e-6)
            assert not np.allclose(obs_0_step2_0, obs_1_step2_0, atol=1e-6)

            return [
                obs_0_reset_0,
                obs_0_step1_0,
                obs_0_step2_0,
                obs_1_reset_0,
                obs_1_step1_0,
                obs_1_step2_0,
            ]
        else:
            return False

    # With seed 1234
    env1 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1', env_name='PYTESTGYM', seed=1234
    )
    env1 = NormalizeObservation(env1)
    _check_reset_reproducibility_with_seed(env1)
    env2 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1', env_name='PYTESTGYM', seed=1234
    )
    env2 = NormalizeObservation(env2)
    _check_reset_reproducibility_with_seed(env2)

    # Check first and second execution have the same results
    # if isinstance(values1, list) and isinstance(values2, list):
    #     for i in range(6):
    #         assert np.allclose(values1[i], values2[i], atol=1e-4)


def test_all_environments():

    envs_id = [
        env_id
        for env_id in gym.envs.registration.registry.keys()  # type: ignore
        if env_id.startswith('Eplus')
    ]
    # Select 10 environments randomly (test would be too large)
    samples_id = sample(envs_id, 5)
    for env_id in samples_id:
        # Create env with TEST name
        env = gym.make(env_id, env_name='PYTEST' + env_id)

        check_env(env)

        # Rename directory with name TEST for future remove
        os.rename(
            env.get_wrapper_attr('workspace_path'),
            'PYTEST' + env.get_wrapper_attr('workspace_path').split('/')[-1],
        )

        env.close()


# -------------------------- Exceptions or rare test cases ------------------------- #


@pytest.mark.parametrize(
    'action',
    [
        (np.array([17.5], dtype=np.float32)),
        (np.array([5.5, 22.5], dtype=np.float32)),
        (np.array([5.5, 22.5, 22.5], dtype=np.float32)),
    ],
)
def test_wrong_action_space(env_5zone, action):
    env_5zone.reset()
    # Forcing wrong action for current action space
    with pytest.raises(ValueError):
        env_5zone.step(action)


def test_energyplus_thread_error(env_5zone):
    # Initialize EnergyPlus thread
    env_5zone.reset()
    # Forcing error in EnergyPlus thread
    env_5zone.energyplus_simulator.sim_results['exit_code'] = 1
    with pytest.raises(RuntimeError):
        env_5zone.step(env_5zone.action_space.sample())


def test_step_in_completed_episode(env_demo):

    env_demo.reset()

    # Running episode until completion
    truncated = terminated = False
    while not terminated and not truncated:
        obs, _, terminated, truncated, info = env_demo.step(
            env_demo.action_space.sample()
        )
    # Save last values
    last_obs = obs
    last_info = info

    # Terminated should be false, and truncated true
    assert not terminated
    assert truncated

    # Trying to step in a completed episode
    for _ in range(2):

        obs, _, terminated, truncated, info = env_demo.step(
            env_demo.action_space.sample()
        )
        # It does not raise exception, but it should return a truncated True again
        # and observation and info should be the same as last step
        assert not terminated
        assert truncated
        assert all(obs == last_obs)
        assert info == last_info


def test_observation_contradiction(env_demo):
    # Forcing observation variables and observation space error
    env_demo.observation_variables.append('unknown_variable')
    with pytest.raises(ValueError):
        env_demo._check_eplus_env()


def test_action_contradiction(env_demo):
    # Forcing action variables and action space error
    env_demo.action_variables.append('unknown_variable')
    with pytest.raises(ValueError):
        env_demo._check_eplus_env()


def test_wrong_weather_variability_conf(env_5zone_stochastic):

    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0), (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    # It should accept ranges
    env_5zone_stochastic._check_eplus_env()

    # It should raise an exception if is not a tuple or with wrong length (3)
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0), (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()
    # It should raise an exception if is not a tuple or with wrong length (3)
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': 25.0,
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()
    # It should raise an exception if the param is not a tuple or float
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ('a', (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()
    # It should raise an exception if the range has not 2 values
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0, 3.0), (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()


def test_is_discrete_property(env_5zone):
    assert isinstance(env_5zone.action_space, gym.spaces.Box)
    assert env_5zone.is_discrete == False

    env_5zone = DiscretizeEnv(
        env=env_5zone,
        discrete_space=gym.spaces.Discrete(10),
        action_mapping=DEFAULT_5ZONE_DISCRETE_FUNCTION,
    )

    assert isinstance(env_5zone.action_space, gym.spaces.Discrete)
    assert env_5zone.is_discrete

    env_5zone.action_space = Dict({})
    assert isinstance(env_5zone.action_space, gym.spaces.Dict)
    assert env_5zone.is_discrete == False
