import os
from queue import Queue
from random import sample

import gymnasium as gym
import numpy as np
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
    obs_dict = env_5zone.get_obs_dict(obs)
    # Context occupancy is a percentage of 20 people
    assert (
        obs_dict['people_occupant']
        == env_5zone.get_wrapper_attr('last_context')[0] * 20
    )
    # Try to update context with a new value (full vector)
    env_5zone.update_context([0.5, 0.5])
    obs, _, _, _, _ = env_5zone.step(a)
    # Check if obs has the new occupancy value
    obs_dict = env_5zone.get_obs_dict(obs)
    assert obs_dict['people_occupant'] == 10
    assert env_5zone.get_wrapper_attr('last_context')[0] == 0.5


def test_update_context_partial_dict(env_5zone):
    """Partial context update by variable name (dict): only update Occupancy."""
    env_5zone.reset()
    a = env_5zone.action_space.sample()
    obs, _, _, _, _ = env_5zone.step(a)
    # Initial context is [1.0, 0.5] (Occupancy, Clothing)
    last = env_5zone.get_wrapper_attr('last_context')
    assert last[0] == 1.0 and last[1] == 0.5
    # Partial update: only Occupancy to 0.5; Clothing stays 0.5
    env_5zone.update_context({'Occupancy': 0.5})
    obs, _, _, _, _ = env_5zone.step(a)
    obs_dict = env_5zone.get_obs_dict(obs)
    assert obs_dict['people_occupant'] == 10  # 0.5 * 20
    last = env_5zone.get_wrapper_attr('last_context')
    assert last[0] == 0.5 and last[1] == 0.5
    # Update only Clothing; Occupancy (index 0) should remain 0.5
    env_5zone.update_context({'Clothing': 0.8})
    env_5zone.step(a)
    last = env_5zone.get_wrapper_attr('last_context')
    assert last[0] == 0.5 and last[1] == 0.8


def test_update_context_partial_dict_raises_when_empty(env_5zone):
    """Partial update with dict raises ValueError when last_context is empty."""
    env_5zone.reset()
    # Simulate empty context (e.g. no initial_context was ever applied)
    env_5zone.unwrapped.last_context = np.array([], dtype=np.float32)
    with pytest.raises(ValueError, match='Initial context is empty'):
        env_5zone.update_context({'Occupancy': 0.5})


def test_update_context_partial_dict_unknown_variable_ignored(env_5zone):
    """Partial update with unknown variable name ignores it and applies known keys."""
    env_5zone.reset()
    env_5zone.step(env_5zone.action_space.sample())
    env_5zone.update_context({'Occupancy': 0.25, 'UnknownVar': 99.0})
    last = env_5zone.get_wrapper_attr('last_context')
    assert last[0] == 0.25  # Occupancy updated
    assert last[1] == 0.5  # Clothing unchanged (initial); UnknownVar ignored


def test_reset_reproducibility():
    # Disable environment global seed
    env = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1', env_name='PYTESTGYM', seed=None
    )

    # Check that the environment is reproducible with same reset seed
    action1 = env.action_space.sample()
    action2 = env.action_space.sample()

    # Case 1: First episode with seed=0
    obs_0_reset, _ = env.reset(seed=0)
    obs_0_step1, _, _, _, _ = env.step(action1)
    obs_0_step2, _, _, _, _ = env.step(action2)

    # Case 2: Second episode with same seed=0 (should be identical)
    obs_1_reset, _ = env.reset(seed=0)
    obs_1_step1, _, _, _, _ = env.step(action1)
    obs_1_step2, _, _, _, _ = env.step(action2)

    # Verify reproducibility: same seed produces same results
    assert np.allclose(
        obs_0_reset, obs_1_reset, atol=1e-6
    ), "Same reset seed should produce same initial observation"
    assert np.allclose(
        obs_0_step1, obs_1_step1, atol=1e-6
    ), "Same reset seed should produce same step 1 observation"
    assert np.allclose(
        obs_0_step2, obs_1_step2, atol=1e-6
    ), "Same reset seed should produce same step 2 observation"

    # Case 3: Different seed (seed=1) should produce different results
    obs_2_reset, _ = env.reset(seed=1)
    obs_2_step1, _, _, _, _ = env.step(action1)
    obs_2_step2, _, _, _, _ = env.step(action2)

    # Verify different seeds produce different results
    assert not np.allclose(
        obs_0_reset, obs_2_reset, atol=1e-6
    ), "Different reset seeds should produce different initial observations"
    assert not np.allclose(
        obs_0_step1, obs_2_step1, atol=1e-6
    ), "Different reset seeds should produce different step 1 observations"
    assert not np.allclose(
        obs_0_step2, obs_2_step2, atol=1e-6
    ), "Different reset seeds should produce different step 2 observations"


def test_global_reproducibility():
    global_seed = 1234
    different_global_seed = 9999

    # Create two environments with the same global seed
    env1 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1',
        env_name='PYTESTGYM',
        seed=global_seed,
    )
    env1 = NormalizeObservation(env1)

    env2 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1',
        env_name='PYTESTGYM',
        seed=global_seed,
    )
    env2 = NormalizeObservation(env2)

    # Use same actions for both environments
    action1 = env1.action_space.sample()
    action2 = env1.action_space.sample()

    # Run both environments with same sequence
    obs1_reset, _ = env1.reset(seed=0)  # seed=0 should be ignored due to global seed
    obs1_step1, _, _, _, _ = env1.step(action1)
    obs1_step2, _, _, _, _ = env1.step(action2)

    obs2_reset, _ = env2.reset(seed=0)  # seed=0 should be ignored due to global seed
    obs2_step1, _, _, _, _ = env2.step(action1)
    obs2_step2, _, _, _, _ = env2.step(action2)

    # Verify: Same global seed produces same results
    assert np.allclose(
        obs1_reset, obs2_reset, atol=1e-6
    ), "Same global seed should produce same initial observation across environments"
    assert np.allclose(
        obs1_step1, obs2_step1, atol=1e-6
    ), "Same global seed should produce same step 1 observation across environments"
    assert np.allclose(
        obs1_step2, obs2_step2, atol=1e-6
    ), "Same global seed should produce same step 2 observation across environments"

    # Verify: Different global seeds produce different results
    env3 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1',
        env_name='PYTESTGYM_DIFF',
        seed=different_global_seed,
    )
    env3 = NormalizeObservation(env3)

    obs3_reset, _ = env3.reset(seed=0)
    obs3_step1, _, _, _, _ = env3.step(action1)
    obs3_step2, _, _, _, _ = env3.step(action2)

    assert not np.allclose(
        obs1_reset, obs3_reset, atol=1e-6
    ), "Different global seeds should produce different initial observations"
    assert not np.allclose(
        obs1_step1, obs3_step1, atol=1e-6
    ), "Different global seeds should produce different step 1 observations"
    assert not np.allclose(
        obs1_step2, obs3_step2, atol=1e-6
    ), "Different global seeds should produce different step 2 observations"


def test_global_seed_ignores_reset_seed():
    global_seed = 5678

    # Create two separate environments with same global seed
    env1 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1',
        env_name='PYTESTGYM_SEED1',
        seed=global_seed,
    )
    env1 = NormalizeObservation(env1)

    env2 = gym.make(
        'Eplus-5zone-hot-continuous-stochastic-v1',
        env_name='PYTESTGYM_SEED2',
        seed=global_seed,
    )
    env2 = NormalizeObservation(env2)

    # Use same actions for both environments
    action1 = env1.action_space.sample()
    action2 = env1.action_space.sample()

    # Same global seed, but DIFFERENT reset seeds
    # If reset seed worked, these would produce different results
    # But since global seed takes priority, they should produce SAME results
    obs1, _ = env1.reset(seed=0)
    obs1_s1, _, _, _, _ = env1.step(action1)
    obs1_s2, _, _, _, _ = env1.step(action2)

    obs2, _ = env2.reset(seed=999)  # Different reset seed, but should be ignored
    obs2_s1, _, _, _, _ = env2.step(action1)
    obs2_s2, _, _, _, _ = env2.step(action2)

    # Verify: Same results despite different reset seeds (proves reset seed is ignored)
    assert np.allclose(
        obs1, obs2, atol=1e-6
    ), "Reset seed should be ignored when global seed is set - same initial obs expected"
    assert np.allclose(
        obs1_s1, obs2_s1, atol=1e-6
    ), "Reset seed should be ignored when global seed is set - same step 1 obs expected"
    assert np.allclose(
        obs1_s2, obs2_s2, atol=1e-6
    ), "Reset seed should be ignored when global seed is set - same step 2 obs expected"


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


def test_weather_variability_with_var_range(env_5zone_stochastic):
    # Correct configuration: sigma/mu/tau as float or tuple, var_range present
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0), (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
        'Relative Humidity': (2.0, 0.0, 24.0, (0, 100)),
    }
    # Should not raise
    env_5zone_stochastic._check_eplus_env()

    # Invalid: not a tuple or wrong length
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0), (-0.5, 0.5)),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()

    # Invalid: single float instead of tuple
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': 25.0,
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()

    # Invalid: non-numeric parameter
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ('a', (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()

    # Invalid: tuple with wrong number of values
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Dry Bulb Temperature': ((1.0, 2.0, 3.0), (-0.5, 0.5), 24.0),
        'Wind Speed': (3.0, 0.0, (30.0, 35.0)),
    }
    with pytest.raises(ValueError):
        env_5zone_stochastic._check_eplus_env()

    # Valid: var_range is optional and can be omitted
    env_5zone_stochastic.get_wrapper_attr('default_options')['weather_variability'] = {
        'Relative Humidity': (2.0, 0.0, 24.0),
    }
    # Should not raise
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
