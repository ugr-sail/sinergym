import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.constants import RANGES_5ZONE, RANGES_DATACENTER, RANGES_IW
from sinergym.utils.rewards import ExpReward, LinearReward
from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation

# ---------------------------------------------------------------------------- #
#                                  Parameters                                  #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument(
    '--environment',
    '-env',
    required=True,
    type=str,
    dest='environment',
    help='Environment name of simulation (see sinergym/__init__.py).')
parser.add_argument(
    '--model',
    '-mod',
    required=True,
    type=str,
    default=None,
    dest='model',
    help='Path where model is stored.')
parser.add_argument(
    '--episodes',
    '-ep',
    type=int,
    default=1,
    dest='episodes',
    help='Number of episodes for training.')
parser.add_argument(
    '--algorithm',
    '-alg',
    type=str,
    default='PPO',
    dest='algorithm',
    help='Algorithm used to train (possible values: PPO, A2C, DQN, DDPG, SAC, TD3).')
parser.add_argument(
    '--reward',
    '-rw',
    type=str,
    default='linear',
    dest='reward',
    help='Reward function used by model, by default is linear (possible values: linear, exponential).')
parser.add_argument(
    '--energy_weight',
    '-rew',
    type=float,
    dest='energy_weight',
    help='Reward energy weight with compatible rewards types.')
parser.add_argument(
    '--normalization',
    '-norm',
    action='store_true',
    dest='normalization',
    help='Apply normalization to observations if this flag is specified.')
parser.add_argument(
    '--logger',
    '-log',
    action='store_true',
    dest='logger',
    help='Apply Sinergym CSVLogger class if this flag is specified.')
parser.add_argument(
    '--seed',
    '-sd',
    type=int,
    default=None,
    dest='seed',
    help='Seed used to algorithm training.')
parser.add_argument(
    '--id',
    '-id',
    type=str,
    default=None,
    dest='id',
    help='Custom load evaluation identifier.')
parser.add_argument(
    '--remote_store',
    '-sto',
    action='store_true',
    dest='remote_store',
    help='Determine if sinergym output will be sent to a Google Cloud Storage Bucket.')
parser.add_argument(
    '--group_name',
    '-group',
    type=str,
    dest='group_name',
    help='This field indicate instance group name')
parser.add_argument(
    '--bucket_name',
    '-buc',
    type=str,
    default='experiments-storage',
    dest='bucket_name',
    help='This field indicates bucket name (not used currently in script)')
parser.add_argument(
    '--auto_delete',
    '-del',
    action='store_true',
    dest='auto_delete',
    help='If is a GCE instance and this flag is active, that instance will be removed from GCP.')
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                Evaluation name                               #
# ---------------------------------------------------------------------------- #
name = args.model.split('/')[-1] + '-EVAL-episodes-' + \
    str(args.episodes)
if args.id:
    name += '-id-' + args.id

# ---------------------------------------------------------------------------- #
#                            Environment definition                            #
# ---------------------------------------------------------------------------- #
if args.reward == 'linear':
    reward = LinearReward
elif args.reward == 'exponential':
    reward = ExpReward
else:
    raise RuntimeError('Reward function specified is not registered.')

env = gym.make(args.environment, reward=reward)
if hasattr(env.reward_fn, 'W_energy') and args.energy_weight:
    env.reward_fn.W_energy = args.energy_weight

# ---------------------------------------------------------------------------- #
#                                   Wrappers                                   #
# ---------------------------------------------------------------------------- #

if args.normalization:
    # We have to know what dictionary ranges to use
    norm_range = None
    env_type = args.environment.split('-')[1]
    if env_type == 'datacenter':
        norm_range = RANGES_DATACENTER
    elif env_type == '5Zone':
        norm_range = RANGES_5ZONE
    elif env_type == 'IWMullion':
        norm_range = RANGES_IW
    else:
        raise NameError('env_type is not valid, check environment name')
    env = NormalizeObservation(env, ranges=norm_range)
if args.logger:
    env = LoggerWrapper(env)

# ---------------------------------------------------------------------------- #
#                                  Load Agent                                  #
# ---------------------------------------------------------------------------- #
# If a model is from a bucket, download model
if 'gs://' in args.model:
    # Download from given bucket (gcloud configured with privileges)
    client = gcloud.init_storage_client()
    bucket_name = args.model.split('/')[2]
    model_path = args.model.split(bucket_name + '/')[-1]
    gcloud.read_from_bucket(client, bucket_name, model_path)
    model_path = './' + model_path
else:
    model_path = args.model


model = None
if args.algorithm == 'DQN':
    model = DQN.load(model_path)
elif args.algorithm == 'DDPG':
    model = DDPG.load(model_path)
elif args.algorithm == 'A2C':
    model = A2C.load(model_path)
elif args.algorithm == 'PPO':
    model = PPO.load(model_path)
elif args.algorithm == 'SAC':
    model = SAC.load(model_path)
elif args.algorithm == 'TD3':
    model = TD3.load(model_path)
else:
    raise RuntimeError('Algorithm specified is not registered.')

# ---------------------------------------------------------------------------- #
#                             Execute loaded agent                             #
# ---------------------------------------------------------------------------- #
for i in range(args.episodes):
    obs, info = env.reset()
    rewards = []
    terminated = False
    current_month = 0
    while not terminated:
        a, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:
            current_month = info['month']
            print(info['month'], sum(rewards))
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()

# ---------------------------------------------------------------------------- #
#                                 Store results                                #
# ---------------------------------------------------------------------------- #
if args.remote_store:
    # Initiate Google Cloud client
    client = gcloud.init_storage_client()
    # Code for send output and tensorboard to common resource here.
    gcloud.upload_to_bucket(
        client,
        src_path=env.simulator._env_working_dir_parent,
        dest_bucket_name='experiments-storage',
        dest_path=name)

# ---------------------------------------------------------------------------- #
#                          Auto-delete remote container                        #
# ---------------------------------------------------------------------------- #
if args.group_name and args.auto_delete:
    token = gcloud.get_service_account_token()
    gcloud.delete_instance_MIG_from_container(args.group_name, token)
