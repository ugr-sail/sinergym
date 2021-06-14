# export EPLUS_PATH=/home/antonio/master/TFM/Gym-Eplus/eplus_env/envs/EnergyPlus-8-6-0
# export BCVTB_PATH=/home/antonio/master/TFM/Gym-Eplus/eplus_env/envs/bcvtb

# pip install stable-baselines3[extra]
# pip install opyplus

import gym
import energym
import argparse
import uuid
import mlflow

import numpy as np

from energym.utils.callbacks import LoggerCallback, LoggerEvalCallback
from energym.utils.wrappers import NormalizeObservation


from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv


parser = argparse.ArgumentParser()
parser.add_argument('--environment', '-env', type=str, default=None)
parser.add_argument('--episodes', '-ep', type=int, default=1)
parser.add_argument('--learning_rate', '-lr', type=float, default=.0007)
parser.add_argument('--n_steps', '-n', type=int, default=5)
parser.add_argument('--gamma', '-g', type=float, default=.99)
parser.add_argument('--gae_lambda', '-gl', type=float, default=1.0)
parser.add_argument('--ent_coef', '-ec', type=float, default=0)
parser.add_argument('--vf_coef', '-v', type=float, default=.5)
parser.add_argument('--max_grad_norm', '-m', type=float, default=.5)
parser.add_argument('--rms_prop_eps', '-rms', type=float, default=1e-05)
args = parser.parse_args()

# experiment ID
environment = args.environment
n_episodes = args.episodes
name = 'A2C-' + environment + '-' + str(n_episodes) + '-episodes'

with mlflow.start_run(run_name=name):

    mlflow.log_param('env', environment)
    mlflow.log_param('episodes', n_episodes)

    mlflow.log_param('learning_rate', args.learning_rate)
    mlflow.log_param('n_steps', args.n_steps)
    mlflow.log_param('gamma', args.gamma)
    mlflow.log_param('gae_lambda', args.gae_lambda)
    mlflow.log_param('ent_coef', args.ent_coef)
    mlflow.log_param('vf_coef', args.vf_coef)
    mlflow.log_param('max_grad_norm', args.max_grad_norm)
    mlflow.log_param('rms_prop_eps', args.rms_prop_eps)

    env = gym.make(environment)
    env = NormalizeObservation(env)

    #### TRAINING ####

    # Build model
    model = DQN('MlpPolicy', env, verbose=1,
                learning_rate=.0001,
                buffer_size=1000000,
                learning_starts=50000,
                batch_size=32,
                tau=1.0,
                gamma=.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=.05,
                max_grad_norm=10,
                tensorboard_log='./tensorboard_log/')
    # The noise objects for DDPG
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(
    #     n_actions), sigma=0.1 * np.ones(n_actions))
    # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1,
    #              tensorboard_log='./tensorboard_log/')
    # model = A2C('MlpPolicy', env, verbose=1,
    #             learning_rate=args.learning_rate,
    #             n_steps=args.n_steps,
    #             gamma=args.gamma,
    #             gae_lambda=args.gae_lambda,
    #             ent_coef=args.ent_coef,
    #             vf_coef=args.vf_coef,
    #             max_grad_norm=args.max_grad_norm,
    #             rms_prop_eps=args.rms_prop_eps,
    #             tensorboard_log='./tensorboard_log/')
    # model = PPO('MlpPolicy', env, verbose=1,
    #             learning_rate=.0003,
    #             n_steps=2048,
    #             batch_size=64,
    #             n_epochs=10,
    #             gamma=.99,
    #             gae_lambda=.95,
    #             clip_range=.2,
    #             ent_coef=0,
    #             vf_coef=.5,
    #             max_grad_norm=.5,
    #             tensorboard_log='./tensorboard_log/')

    # model = SAC(policy='MlpPolicy', env=env,
    #             tensorboard_log='./tensorboard_log/')

    n_timesteps_episode = env.simulator._eplus_one_epi_len / \
        env.simulator._eplus_run_stepsize
    timesteps = n_episodes * n_timesteps_episode

    env = DummyVecEnv([lambda: env])

    # Callbacks
    freq = 2  # evaluate every N episodes
    eval_callback = LoggerEvalCallback(env, best_model_save_path='./best_models/' + name + '/',
                                       log_path='./best_models/' + name + '/', eval_freq=n_timesteps_episode * freq,
                                       deterministic=True, render=False, n_eval_episodes=1)
    log_callback = LoggerCallback(energym_logger=True)
    callback = CallbackList([log_callback])

    # Training
    model.learn(total_timesteps=timesteps, callback=callback, log_interval=1)
    # model.save(name)

    #### LOAD MODEL ####

    # model = A2C.load(name)

    # for i in range(n_episodes - 1):
    #     obs = env.reset()
    #     rewards = []
    #     done = False
    #     current_month = 0
    #     while not done:
    #         a, _ = model.predict(obs)
    #         obs, reward, done, info = env.step(a)
    #         rewards.append(reward)
    #         if info['month'] != current_month:
    #             current_month = info['month']
    #             print(info['timestep'], sum(rewards))
    #     print('Episode ', i, 'Mean reward: ', np.mean(
    #         rewards), 'Cumulative reward: ', sum(rewards))
    # env.close()

    # mlflow.log_metric('mean_reward', np.mean(rewards))
    # mlflow.log_metric('cumulative_reward', sum(rewards))

    mlflow.end_run()
