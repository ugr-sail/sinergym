import gym
import sinergym
import argparse
import uuid
import mlflow
import os
from datetime import datetime

import numpy as np

from sinergym.utils.callbacks import LoggerCallback, LoggerEvalCallback
from sinergym.utils.wrappers import MultiObsWrapper, NormalizeObservation, LoggerWrapper
from sinergym.utils.rewards import *
import sinergym.utils.gcloud as gcloud


from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

#--------------------------------BATTERY ARGUMENTS DEFINITION---------------------------------#
parser = argparse.ArgumentParser()
# commons arguments for battery
parser.add_argument(
    '--environment',
    '-env',
    required=True,
    type=str,
    dest='environment',
    help='Environment name of simulation (see sinergym/__init__.py).')
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
    help='Algorithm used to train (possible values: PPO, A2C, DQN, DDPG, SAC).')
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
    '--multiobs',
    '-mobs',
    action='store_true',
    dest='multiobs',
    help='Apply Multi observations if this flag is specified.')
parser.add_argument(
    '--logger',
    '-log',
    action='store_true',
    dest='logger',
    help='Apply Sinergym CSVLogger class if this flag is specified.')
parser.add_argument(
    '--tensorboard',
    '-tens',
    type=str,
    default=None,
    dest='tensorboard',
    help='Tensorboard path for logging (if not specified, tensorboard log will not be stored).')
parser.add_argument(
    '--evaluation',
    '-eval',
    action='store_true',
    dest='evaluation',
    help='Evaluation is processed during training with this flag (save best model online).')
parser.add_argument(
    '--eval_freq',
    '-evalf',
    type=int,
    default=2,
    dest='eval_freq',
    help='Episodes executed before applying evaluation (if evaluation flag is not specified, this value is useless).')
parser.add_argument(
    '--eval_length',
    '-evall',
    type=int,
    default=2,
    dest='eval_length',
    help='Episodes executed during evaluation (if evaluation flag is not specified, this value is useless).')
parser.add_argument(
    '--log_interval',
    '-inter',
    type=int,
    default=1,
    dest='log_interval',
    help='model training log_interval parameter. See documentation since this value is different in every algorithm.')
parser.add_argument(
    '--seed',
    '-sd',
    type=int,
    default=None,
    dest='seed',
    help='Seed used to algorithm training.')
parser.add_argument(
    '--remote_store',
    '-sto',
    action='store_true',
    dest='remote_store',
    help='Determine if sinergym output will be sent to a common resource')

parser.add_argument('--learning_rate', '-lr', type=float, default=.0007)
parser.add_argument('--gamma', '-g', type=float, default=.99)
parser.add_argument('--n_steps', '-n', type=int, default=5)
parser.add_argument('--gae_lambda', '-gl', type=float, default=1.0)
parser.add_argument('--ent_coef', '-ec', type=float, default=0)
parser.add_argument('--vf_coef', '-v', type=float, default=.5)
parser.add_argument('--max_grad_norm', '-m', type=float, default=.5)
parser.add_argument('--rms_prop_eps', '-rms', type=float, default=1e-05)
parser.add_argument('--buffer_size', '-bfs', type=int, default=1000000)
parser.add_argument('--learning_starts', '-ls', type=int, default=100)
parser.add_argument('--tau', '-tu', type=float, default=0.005)
# for DDPG noise only
parser.add_argument('--sigma', '-sig', type=float, default=0.1)

args = parser.parse_args()
#---------------------------------------------------------------------------------------------#

# Environment construction (with reward specified)
if args.reward == 'linear':
    env = gym.make(args.environment, reward=LinearReward())
elif args.reward == 'exponential':
    env = gym.make(args.environment, reward=ExpReward())
else:
    raise RuntimeError('Reward function specified is not registered.')

# env wrappers (optionals)
if args.normalization:
    env = NormalizeObservation(env)
if args.logger:
    env = LoggerWrapper(env)
if args.multiobs:
    env = MultiObsWrapper(env)


######################## TRAINING ########################

experiment_date = datetime.today().strftime('%Y-%m-%d %H:%M')

# Defining model(algorithm)
model = None
name = args.algorithm + '-' + args.environment + \
    '-episodes_' + str(args.episodes)
if args.seed:
    name += '-seed_' + str(args.seed)
name += '(' + experiment_date + ')'
#--------------------------DQN---------------------------#
if args.algorithm == 'DQN':
    model = DQN('MlpPolicy', env, verbose=1,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=50000,
                batch_size=32,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=.05,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                tensorboard_log=args.tensorboard)
#--------------------------------------------------------#

#--------------------------DDPG--------------------------#
# The noise objects for DDPG
elif args.algorithm == 'DDPG':
    if args.sigma:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy",
                 env,
                 action_noise=action_noise,
                 verbose=1,
                 seed=args.seed,
                 tensorboard_log=args.tensorboard)
#--------------------------------------------------------#

#--------------------------A2C---------------------------#
elif args.algorithm == 'A2C':
    model = A2C('MlpPolicy', env, verbose=1,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                rms_prop_eps=args.rms_prop_eps,
                seed=args.seed,
                tensorboard_log=args.tensorboard)
#--------------------------------------------------------#

#--------------------------PPO---------------------------#
elif args.algorithm == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=64,
                n_epochs=10,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=.2,
                ent_coef=0,
                vf_coef=.5,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                tensorboard_log=args.tensorboard)
#--------------------------------------------------------#

#--------------------------SAC---------------------------#
elif args.algorithm == 'SAC':
    model = SAC(policy='MlpPolicy',
                env=env,
                seed=args.seed,
                tensorboard_log=args.tensorboard)
#--------------------------------------------------------#

#-------------------------ERROR?-------------------------#
else:
    raise RuntimeError('Algorithm specified is not registered.')
#--------------------------------------------------------#

# Calculating n_timesteps_episode for training
n_timesteps_episode = env.simulator._eplus_one_epi_len / \
    env.simulator._eplus_run_stepsize
timesteps = args.episodes * n_timesteps_episode

# For callbacks processing
env_vec = DummyVecEnv([lambda: env])

# Using Callbacks for training
callbacks = []

if args.evaluation:
    eval_callback = LoggerEvalCallback(
        env_vec,
        best_model_save_path=env.simulator._env_working_dir_parent +
        '/best_model/' +
        name +
        '/',
        log_path=env.simulator._env_working_dir_parent +
        '/best_model/' +
        name +
        '/',
        eval_freq=n_timesteps_episode *
        args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=args.eval_length)
    callbacks.append(eval_callback)

if args.tensorboard:
    log_callback = LoggerCallback(sinergym_logger=bool(args.logger))
    callbacks.append(log_callback)
    # lets change default dir for TensorboardFormatLogger only
    tb_path = args.tensorboard + '/' + name
    new_logger = configure(tb_path, ["tensorboard"])
    model.set_logger(new_logger)

callback = CallbackList(callbacks)

# Training
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=args.log_interval)
if not args.evaluation:
    model.save(env.simulator._env_working_dir_parent + '/' + name)

if args.remote_store:
    # Initiate Google Cloud client
    client = gcloud.init_storage_client()
    # Code for send output and tensorboard to common resource here.
    gcloud.upload_to_bucket(
        client,
        src_path=env.simulator._env_working_dir_parent,
        dest_bucket_name='experiments-storage',
        dest_path=name)
    if args.tensorboard:
        gcloud.upload_to_bucket(
            client,
            src_path=args.tensorboard,
            dest_bucket_name='experiments-storage',
            dest_path=name + '/tensorboard_log')

    # We can programming a command to shut down remote machine or stop it too
    os.system('sudo shutdown -h now')
