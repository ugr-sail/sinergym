import pytest
from random import randint
import gym
import energym
import os
import csv


def test_reset(env_demo):
    obs = env_demo.reset()
    assert len(obs) == 19
    assert env_demo.simulator._episode_existed


def test_step(env_demo):
    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 19
    assert type(reward) != None
    assert not done
    assert list(info.keys()) == ['timestep', 'time_elapsed', 'day', 'month', 'hour', 'total_power',
                                 'total_power_no_units', 'comfort_penalty', 'temperatures', 'out_temperature']
    assert info['timestep'] == 1
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']

    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 19
    assert type(reward) != None
    assert not done
    assert list(info.keys()) == ['timestep', 'time_elapsed', 'day', 'month', 'hour', 'total_power',
                                 'total_power_no_units', 'comfort_penalty', 'temperatures', 'out_temperature']
    assert info['timestep'] == 2
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']


def test_close(env_demo):
    env_demo.close()
    assert not env_demo.simulator._episode_existed
    assert env_demo.simulator._conn == None


def test_loggers(env_demo):

    logger = env_demo.logger

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env_demo.simulator._env_working_dir_parent+'/progress.csv'
    assert logger.log_file == env_demo.simulator._eplus_working_dir+'/monitor.csv'

    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(logger.log_file)

    # Check headers
    with open(logger.log_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row)+'\n' == logger.monitor_header
            break
    with open(logger.log_progress_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row)+'\n' == logger.progress_header
            break


def test_all_environments():

    envs_id = [env_spec.id for env_spec in gym.envs.registry.all()
               if env_spec.id.startswith('Eplus')]
    for env_id in envs_id:
        # Create env with TEST name
        env = gym.make(env_id)

        initial_obs = env.reset()
        assert len(initial_obs) > 0

        a = env.action_space.sample()
        assert a is not None

        obs, reward, done, info = env.step(a)
        assert len(initial_obs) == len(obs)
        assert reward != 0
        assert done is not None
        assert type(info) == dict and len(info) > 0

        # Rename directory with name TEST for future remove
        os.rename(env.simulator._env_working_dir_parent, 'Eplus-env-TEST' +
                  env.simulator._env_working_dir_parent.split('/')[-1])

        # env.close()
        # assert not env.simulator._episode_existed
        # assert env.simulator._conn==None
