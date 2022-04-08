import argparse
import os
from datetime import datetime

import gym
import mlflow
import numpy as np
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import LoggerCallback, LoggerEvalCallback
from sinergym.utils.common import RANGES_5ZONE, RANGES_DATACENTER, RANGES_IW
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import (LoggerWrapper, MultiObsWrapper,
                                     NormalizeObservation)

# ---------------------------------------------------------------------------- #
#                             Parameters definition                            #
# ---------------------------------------------------------------------------- #
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
    '--id',
    '-id',
    type=str,
    default=None,
    dest='id',
    help='Custom experiment identifier.')
parser.add_argument(
    '--remote_store',
    '-sto',
    action='store_true',
    dest='remote_store',
    help='Determine if sinergym output will be sent to a Google Cloud Storage Bucket.')
parser.add_argument(
    '--mlflow_store',
    '-mlflow',
    action='store_true',
    dest='mlflow_store',
    help='Determine if sinergym output will be sent to a mlflow artifact storage')
parser.add_argument(
    '--group_name',
    '-group',
    type=str,
    dest='group_name',
    help='This field indicate instance group name')
parser.add_argument(
    '--auto_delete',
    '-del',
    action='store_true',
    dest='auto_delete',
    help='If is a GCE instance and this flag is active, that instance will be removed from GCP.')

parser.add_argument('--learning_rate', '-lr', type=float, default=.0003)
parser.add_argument('--n_steps', '-n', type=int, default=2048)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--n_epochs', '-ne', type=int, default=10)
parser.add_argument('--gamma', '-g', type=float, default=.99)
parser.add_argument('--gae_lambda', '-gl', type=float, default=.95)
parser.add_argument('--ent_coef', '-ec', type=float, default=0)
parser.add_argument('--vf_coef', '-v', type=float, default=.5)
parser.add_argument('--max_grad_norm', '-m', type=float, default=.5)
parser.add_argument('--buffer_size', '-bfs', type=int, default=1000000)
parser.add_argument('--learning_starts', '-ls', type=int, default=100)
parser.add_argument('--tau', '-tu', type=float, default=0.005)
parser.add_argument('--gradient_steps', '-gs', type=int, default=1)
parser.add_argument('--clip_range', '-cr', type=float, default=.2)
parser.add_argument('--sigma', '-sig', type=float, default=0.1)

args = parser.parse_args()
#------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------- #
#                               Register run name                              #
# ---------------------------------------------------------------------------- #
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
name = args.algorithm + '-' + args.environment + \
    '-episodes' + str(args.episodes)
if args.seed:
    name += '-seed' + str(args.seed)
if args.id:
    name += '-id' + str(args.id)
name += '_' + experiment_date
# ---------------------------------------------------------------------------- #
#                    Check if MLFLOW_TRACKING_URI is defined                   #
# ---------------------------------------------------------------------------- #
mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
if mlflow_tracking_uri is not None:
    # Check ping to server
    mlflow_ip = mlflow_tracking_uri.split('/')[-1].split(':')[0]
    # If server is not valid, setting default local path to mlflow
    response = os.system("ping -c 1 " + mlflow_ip)
    if response != 0:
        mlflow.set_tracking_uri('file://' + os.getcwd() + '/mlruns')
# MLflow track
with mlflow.start_run(run_name=name):
    # Log experiment params
    mlflow.log_param('sinergym-version', sinergym.__version__)

    mlflow.log_param('env', args.environment)
    mlflow.log_param('episodes', args.episodes)
    mlflow.log_param('algorithm', args.algorithm)
    mlflow.log_param('reward', args.reward)
    mlflow.log_param('normalization', bool(args.normalization))
    mlflow.log_param('multi-observations', bool(args.multiobs))
    mlflow.log_param('logger', bool(args.logger))
    mlflow.log_param('tensorboard', args.tensorboard)
    mlflow.log_param('evaluation', bool(args.evaluation))
    mlflow.log_param('evaluation-frequency', args.eval_freq)
    mlflow.log_param('evaluation-length', args.eval_length)
    mlflow.log_param('log-interval', args.log_interval)
    mlflow.log_param('seed', args.seed)
    mlflow.log_param('remote-store', bool(args.remote_store))

    mlflow.log_param('learning-rate', args.learning_rate)
    mlflow.log_param('n-steps', args.n_steps)
    mlflow.log_param('batch-size', args.batch_size)
    mlflow.log_param('n-epochs', args.n_epochs)
    mlflow.log_param('gamma', args.gamma)
    mlflow.log_param('gae-lambda', args.gae_lambda)
    mlflow.log_param('ent-coef', args.ent_coef)
    mlflow.log_param('vf-coef', args.vf_coef)
    mlflow.log_param('max-grad-norm', args.max_grad_norm)
    mlflow.log_param('buffer-size', args.buffer_size)
    mlflow.log_param('learning-starts', args.learning_starts)
    mlflow.log_param('tau', args.tau)
    mlflow.log_param('gradient-steps', args.gradient_steps)
    mlflow.log_param('clip-range', args.clip_range)
    mlflow.log_param('sigma', args.sigma)
    mlflow.log_param('id', args.id)

    # ---------------------------------------------------------------------------- #
    #               Environment construction (with reward specified)               #
    # ---------------------------------------------------------------------------- #
    if args.reward == 'linear':
        reward = LinearReward
    elif args.reward == 'exponential':
        reward = ExpReward
    else:
        raise RuntimeError('Reward function specified is not registered.')

    env = gym.make(args.environment, reward=reward)
    # env for evaluation if is enabled
    eval_env = None
    if args.evaluation:
        eval_env = gym.make(args.environment, reward=reward)

    # ---------------------------------------------------------------------------- #
    #                                   Wrappers                                   #
    # ---------------------------------------------------------------------------- #
    if args.normalization:
        # dictionary ranges to use
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
        if eval_env is not None:
            eval_env = NormalizeObservation(eval_env, ranges=norm_range)
    if args.logger:
        env = LoggerWrapper(env)
        if eval_env is not None:
            eval_env = LoggerWrapper(eval_env)
    if args.multiobs:
        env = MultiObsWrapper(env)
        if eval_env is not None:
            eval_env = MultiObsWrapper(eval_env)
    # ---------------------------------------------------------------------------- #
    #                           Defining model(algorithm)                          #
    # ---------------------------------------------------------------------------- #
    model = None
    #--------------------------DQN---------------------------#
    if args.algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1,
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    train_freq=4,
                    gradient_steps=args.gradient_steps,
                    target_update_interval=10000,
                    exploration_fraction=.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=.05,
                    max_grad_norm=args.max_grad_norm,
                    seed=args.seed,
                    tensorboard_log=args.tensorboard)
    #--------------------------------------------------------#
    #                           DDPG                         #
    #--------------------------------------------------------#
    # noise objects for DDPG
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
    #                           A2C                          #
    #--------------------------------------------------------#
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
    #                           PPO                          #
    #--------------------------------------------------------#
    elif args.algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1,
                    learning_rate=args.learning_rate,
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    seed=args.seed,
                    tensorboard_log=args.tensorboard)
    #--------------------------------------------------------#
    #                           SAC                          #
    #--------------------------------------------------------#
    elif args.algorithm == 'SAC':
        model = SAC(policy='MlpPolicy',
                    env=env,
                    seed=args.seed,
                    tensorboard_log=args.tensorboard)
    #--------------------------------------------------------#
    #                           TD3                          #
    #--------------------------------------------------------#
    elif args.algorithm == 'TD3':
        model = TD3(policy='MlpPolicy',
                    env=env, seed=args.seed,
                    tensorboard_log=args.tensorboard,
                    learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    train_freq=(1, 'episode'),
                    gradient_steps=-1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    create_eval_env=False,
                    policy_kwargs=None,
                    verbose=0,
                    device='auto',
                    _init_setup_model=True)
    #--------------------------------------------------------#
    #                           Error                        #
    #--------------------------------------------------------#
    else:
        raise RuntimeError('Algorithm specified is not registered.')
    #--------------------------------------------------------#

    # ---------------------------------------------------------------------------- #
    #       Calculating total training timesteps based on number of episodes       #
    # ---------------------------------------------------------------------------- #
    n_timesteps_episode = env.simulator._eplus_one_epi_len / \
        env.simulator._eplus_run_stepsize
    timesteps = args.episodes * n_timesteps_episode - 1

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #
    callbacks = []

    # Set up Evaluation and saving best model
    if args.evaluation:
        eval_callback = LoggerEvalCallback(
            eval_env,
            best_model_save_path='best_model/' + name + '/',
            log_path='best_model/' + name + '/',
            eval_freq=n_timesteps_episode *
            args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=args.eval_length)
        callbacks.append(eval_callback)

    # Set up tensorboard logger
    if args.tensorboard:
        log_callback = LoggerCallback(sinergym_logger=bool(args.logger))
        callbacks.append(log_callback)
        # lets change default dir for TensorboardFormatLogger only
        tb_path = args.tensorboard + '/' + name
        new_logger = configure(tb_path, ["tensorboard"])
        model.set_logger(new_logger)

    callback = CallbackList(callbacks)

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=args.log_interval)
    model.save(env.simulator._env_working_dir_parent + '/' + name)

    # If Algorithm doesn't reset or close environment, this script will do in
    # order to log correctly all simulation data (Energyplus + Sinergym logs)
    if env.simulator._episode_existed:
        env.close()

    # ---------------------------------------------------------------------------- #
    #                           Mlflow artifacts storege                           #
    # ---------------------------------------------------------------------------- #
    if args.mlflow_store:
        # Code for send output and tensorboard to mlflow artifacts.
        mlflow.log_artifacts(
            local_dir=env.simulator._env_working_dir_parent,
            artifact_path=name + '/')
        if args.evaluation:
            mlflow.log_artifacts(
                local_dir='best_model/' + name + '/',
                artifact_path='best_model/' + name + '/')
        # If tensorboard is active (in local) we should send to mlflow
        if args.tensorboard and 'gs://experiments-storage' not in args.tensorboard:
            mlflow.log_artifacts(
                local_dir=args.tensorboard + '/' + name + '/',
                artifact_path=os.path.abspath(args.tensorboard).split('/')[-1] + '/' + name + '/')

    # ---------------------------------------------------------------------------- #
    #                          Google Cloud Bucket Storage                         #
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
        if args.evaluation:
            gcloud.upload_to_bucket(
                client,
                src_path='best_model/' + name + '/',
                dest_bucket_name='experiments-storage',
                dest_path='best_model/' + name + '/')
        # If tensorboard is active (in local) we should send to bucket
        if args.tensorboard and 'gs://experiments-storage' not in args.tensorboard:
            gcloud.upload_to_bucket(
                client,
                src_path=args.tensorboard + '/' + name + '/',
                dest_bucket_name='experiments-storage',
                dest_path=os.path.abspath(args.tensorboard).split('/')[-1] + '/' + name + '/')
        # gcloud.upload_to_bucket(
        #     client,
        #     src_path='mlruns/',
        #     dest_bucket_name='experiments-storage',
        #     dest_path='mlruns/')

    # End mlflow run
    mlflow.end_run()

    # ---------------------------------------------------------------------------- #
    #                   Autodelete option if is a cloud resource                   #
    # ---------------------------------------------------------------------------- #
    if args.group_name and args.auto_delete:
        token = gcloud.get_service_account_token()
        gcloud.delete_instance_MIG_from_container(args.group_name, token)
