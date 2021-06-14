import pytest
from random import randint
import gym
import energym.utils.wrappers
import os
import csv
from stable_baselines3.common.env_checker import check_env
from datetime import datetime


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
                                 'total_power_no_units', 'comfort_penalty', 'temperatures', 'out_temperature', 'action_']
    assert info['timestep'] == 1
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']

    action = randint(0, 9)
    obs, reward, done, info = env_demo.step(action)

    assert len(obs) == 19
    assert type(reward) != None
    assert not done
    assert list(info.keys()) == ['timestep', 'time_elapsed', 'day', 'month', 'hour', 'total_power',
                                 'total_power_no_units', 'comfort_penalty', 'temperatures', 'out_temperature', 'action_']
    assert info['timestep'] == 2
    assert info['time_elapsed'] == env_demo.simulator._eplus_run_stepsize * \
        info['timestep']


def test_close(env_demo):
    env_demo.close()
    assert not env_demo.simulator._episode_existed
    assert env_demo.simulator._conn == None


@pytest.mark.parametrize('env_name', [('env_demo'), ('env_wrapper'), ])
def test_loggers(env_name, request):
    env = request.getfixturevalue(env_name)
    logger = env.logger

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env.simulator._env_working_dir_parent+'/progress.csv'
    assert logger.log_file == env.simulator._eplus_working_dir+'/monitor.csv'

    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(logger.log_file)

    # If env is wrapped with normalize obs...
    if(type(env) == energym.utils.wrappers.NormalizeObservation):
        assert os.path.isfile(logger.log_file[:-4]+'_normalized.csv')
    else:
        assert not os.path.isfile(logger.log_file[:-4]+'_normalized.csv')

    # Check headers
    with open(logger.log_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row) == logger.monitor_header
            break
    with open(logger.log_progress_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row)+'\n' == logger.progress_header
            break
    if(type(env) == energym.utils.wrappers.NormalizeObservation):
        with open(logger.log_file[:-4]+'_normalized.csv', mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                assert ','.join(row) == logger.monitor_header
                break


def test_all_environments():

    envs_id = [env_spec.id for env_spec in gym.envs.registry.all()
               if env_spec.id.startswith('Eplus')]
    for env_id in envs_id:
        # Create env with TEST name
        env = gym.make(env_id)

        # stable_baselines 3 environment checker. Check if environment follows Gym API.
        check_env(env)

        # Lets run one episode and check time (only some envs randomly)
        random_value = randint(0, 2)
        check_time = random_value == 0

        if check_time:
            begin_time = datetime.now()
            done = False
            env.reset()
            while not done:
                a = env.action_space.sample()
                obs, reward, done, info = env.step(a)
            end_time = datetime.now()

            execution_time = end_time - begin_time
            print('ERROR: ', env_id, ' executing too much time')
            # 3 month simulation per episode
            if env_id == 'Eplus-demo-v1':
                assert execution_time.total_seconds() < 15
            # 1 year simulation per episode
            else:
                assert execution_time.total_seconds() < 30

        # close env
        env.close()

        # Rename directory with name TEST for future remove
        os.rename(env.simulator._env_working_dir_parent, 'Eplus-env-TEST' +
                  env.simulator._env_working_dir_parent.split('/')[-1])
