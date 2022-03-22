import gym
import sinergym
import argparse

import numpy as np

from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation
from sinergym.utils.wrappers import NormalizeObservation, LoggerWrapper
from sinergym.utils.rewards import LinearReward, ExpReward

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# -------------------------------- Parameters -------------------------------- #

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
args = parser.parse_args()

# -------------------------- Environment definition -------------------------- #
if args.reward == 'linear':
    reward = LinearReward()
elif args.reward == 'exponential':
    reward = ExpReward()
else:
    raise RuntimeError('Reward function specified is not registered.')

env = gym.make(args.environment, reward=reward)

if args.normalization:
    env = NormalizeObservation(env)
if args.logger:
    env = LoggerWrapper(env)

# ------------------- Load Model dependending on algorithm ------------------- #
model = None
if args.algorithm == 'DQN':
    model = DQN.load(args.model)
elif args.algorithm == 'DDPG':
    model = DDPG.load(args.model)
elif args.algorithm == 'A2C':
    model = A2C.load(args.model)
elif args.algorithm == 'PPO':
    model = PPO.load(args.model)
elif args.algorithm == 'SAC':
    model = SAC.load(args.model)
elif args.algorithm == 'TD3':
    model = TD3.load(args.model)
else:
    raise RuntimeError('Algorithm specified is not registered.')


for i in range(args.episodes):
    obs = env.reset()
    rewards = []
    done = False
    current_month = 0
    while not done:
        a, _ = model.predict(obs)
        obs, reward, done, info = env.step(a)
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
