import argparse
import os
import shutil
from datetime import datetime
from glob import glob

import gymnasium as gym

import sinergym

envs_id = [env_spec for env_spec in gym.envs.registration.registry.keys()
           if env_spec.startswith('Eplus')]
envs_id = ['Eplus-5zone-mixed-continuous-stochastic-v1',
           'Eplus-datacenter-mixed-continuous-stochastic-v1',
           'Eplus-office-mixed-continuous-stochastic-v1',
           'Eplus-warehouse-mixed-continuous-stochastic-v1',
           'Eplus-officegrid-mixed-continuous-stochastic-v1',
           'Eplus-shop-mixed-continuous-stochastic-v1']

parser = argparse.ArgumentParser()
parser.add_argument('--environments', '-envs', default=envs_id, nargs='+')
parser.add_argument('--episodes', '-ep', type=int, default=1)
args = parser.parse_args()
results = {}

for env_id in args.environments:
    env = gym.make(env_id)
    # BEGIN EXECUTION TIME
    begin_time = datetime.now()
    truncated = terminated = False
    for _ in range(args.episodes):
        env.reset()
        while not (terminated or truncated):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
    end_time = datetime.now()
    env.close()
    # END EXECUTION TIME

    execution_time = end_time - begin_time
    results[env_id] = execution_time.total_seconds()

    # Rename directory with name TEST for future remove
    os.rename(env.get_wrapper_attr('workspace_path'), 'Eplus-env-TEST' +
              env.get_wrapper_attr('workspace_path').split('/')[-1])

print('====================================================')
print('TIMES RECORDED IN ENVIRONMENTS WITH ', args.episodes, ' EPISODE(S):')
print('====================================================')
for key, value in results.items():
    print('{:<50}: {} SECONDS'.format(key, str(value)))


# Deleting all temporal directories generated during tests
directories = glob('Eplus-env-TEST*/')
for directory in directories:
    shutil.rmtree(directory)

# Deleting new random weather files once it has been checked
files = glob('sinergym/data/weather/*Random*.epw')
for file in files:
    os.remove(file)
