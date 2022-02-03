import csv
import os

import numpy as np
import pytest
from stable_baselines3.common.env_util import is_wrapped

from sinergym.utils.wrappers import NormalizeObservation


@pytest.mark.parametrize('env_name',
                         [('env_wrapper_logger'), ('env_all_wrappers'), ])
def test_logger_wrapper(env_name, request):
    env = request.getfixturevalue(env_name)
    logger = env.logger
    env.reset()

    # Check CSV's have been created and linked in simulator correctly
    assert logger.log_progress_file == env.simulator._env_working_dir_parent + '/progress.csv'
    assert logger.log_file == env.simulator._eplus_working_dir + '/monitor.csv'

    tmp_log_file = logger.log_file

    # simulating short episode
    for _ in range(10):
        env.step(env.action_space.sample())
    env.reset()

    assert os.path.isfile(logger.log_progress_file)
    assert os.path.isfile(tmp_log_file)

    # If env is wrapped with normalize obs...
    if is_wrapped(env, NormalizeObservation):
        print(logger.log_file[:-4] + '_normalized.csv')
        assert os.path.isfile(tmp_log_file[:-4] + '_normalized.csv')
    else:
        assert not os.path.isfile(tmp_log_file[:-4] + '_normalized.csv')

    # Check headers
    with open(tmp_log_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row) == logger.monitor_header
            break
    with open(logger.log_progress_file, mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            assert ','.join(row) + '\n' == logger.progress_header
            break
    if is_wrapped(env, NormalizeObservation):
        with open(tmp_log_file[:-4] + '_normalized.csv', mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                assert ','.join(row) == logger.monitor_header
                break

    env.close()


def test_env_wrappers(env_all_wrappers):
    # env_wrapper history should be empty at the beginning
    assert len(env_all_wrappers.history) == 0
    for i in range(1):  # Only need 1 episode
        obs = env_all_wrappers.reset()
        # This obs should be normalize --> [-1,1]
        assert (obs >= 0).all() and (obs <= 1).all()

        done = False
        while not done:
            a = env_all_wrappers.action_space.sample()
            obs, reward, done, info = env_all_wrappers.step(a)

    # Let's check if history has been completed succesfully
    assert len(env_all_wrappers.history) == 5
    assert isinstance(env_all_wrappers.history[0], np.ndarray)
    env_all_wrappers.close()
