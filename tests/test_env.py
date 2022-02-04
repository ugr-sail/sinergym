import os
from random import randint

import gym
from stable_baselines3.common.env_checker import check_env


def test_reset(env_demo):
    obs = env_demo.reset()
    assert len(obs) == 19
    assert env_demo.simulator._episode_existed


def test_step(env_demo):
    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 19
    assert not isinstance(reward, type(None))
    assert not done
    assert list(
        info.keys()) == [
        'timestep',
        'time_elapsed',
        'day',
        'month',
        'hour',
        'total_power',
        'total_power_no_units',
        'comfort_penalty',
        'temperatures',
        'out_temperature',
        'action_']
    assert info['timestep'] == 1
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']

    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 19
    assert not isinstance(reward, type(None))
    assert not done
    assert list(
        info.keys()) == [
        'timestep',
        'time_elapsed',
        'day',
        'month',
        'hour',
        'total_power',
        'total_power_no_units',
        'comfort_penalty',
        'temperatures',
        'out_temperature',
        'action_']
    assert info['timestep'] == 2
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']


def test_close(env_demo):
    env_demo.close()
    assert not env_demo.simulator._episode_existed
    assert env_demo.simulator._conn is None


def test_all_environments():

    envs_id = [env_spec.id for env_spec in gym.envs.registry.all()
               if env_spec.id.startswith('Eplus')]
    envs_id = [env_id for env_id in envs_id if 'IWMullion' not in env_id]
    for env_id in envs_id:
        # Create env with TEST name
        env = gym.make(env_id)

        # stable_baselines 3 environment checker. Check if environment follows
        # Gym API.
        check_env(env)

        # Rename directory with name TEST for future remove
        os.rename(env.simulator._env_working_dir_parent, 'Eplus-env-TEST' +
                  env.simulator._env_working_dir_parent.split('/')[-1])
