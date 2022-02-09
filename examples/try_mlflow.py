"""
Example of mlflow use to automatize experimentation
Requires: mlflow, stable-baselines

$ mlflow ui

Execution in localhost:5000 by default
"""

import argparse

import gym
import mlflow
import numpy as np
from stable_baselines3 import PPO2
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.policies import MlpPolicy

import sinergym

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', '-g', type=float, default=.99)
parser.add_argument('--n_steps', '-n', type=int, default=128)
parser.add_argument('--ent_coef', '-e', type=float, default=.01)
parser.add_argument('--learning_rate', '-r', type=float, default=.00025)
parser.add_argument('--vf_coef', '-v', type=float, default=.5)
parser.add_argument('--max_grad_norm', '-m', type=float, default=.5)
parser.add_argument('--lam', '-l', type=float, default=.95)
parser.add_argument('--nminibatches', '-b', type=int, default=4)
parser.add_argument('--noptepochs', '-o', type=int, default=4)
parser.add_argument('--timesteps', '-t', type=int, default=5000)
args = parser.parse_args()

with mlflow.start_run(run_name='PPO2_test'):

    mlflow.log_param('gamma', args.gamma)
    mlflow.log_param('n_steps', args.n_steps)
    mlflow.log_param('ent_coef', args.ent_coef)
    mlflow.log_param('learning_rate', args.learning_rate)
    mlflow.log_param('vf_coef', args.vf_coef)
    mlflow.log_param('max_grad_norm', args.max_grad_norm)
    mlflow.log_param('lam', args.lam)
    mlflow.log_param('nminibatches', args.nminibatches)
    mlflow.log_param('noptepochs', args.noptepochs)

    env = gym.make('Eplus-demo-v1')

    # Possible params: policy, gamma, n_steps, ent_coef, learning_rate,
    # vf_coef, max_grad_norm, lam, nminibatches, noptepochs...
    # https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

    model = PPO2(
        MlpPolicy,
        env,
        verbose=1,
        gamma=args.gamma,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        lam=args.lam,
        nminibatches=args.nminibatches,
        noptepochs=args.noptepochs)
    model.learn(total_timesteps=args.timesteps)
    model.save('ppo2_eplus')

    for i in range(1):
        obs = env.reset()
        rewards = []
        done = False
        current_month = 0
    while not done:
        a, _ = model.predict(obs)
        obs, reward, done, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))

    mlflow.log_metric('mean_reward', np.mean(rewards))
    mlflow.log_metric('sum_reward', sum(rewards))

    mlflow.log_artifact('./ppo2_eplus.zip')

    mlflow.end_run()
