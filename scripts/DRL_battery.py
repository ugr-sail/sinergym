import argparse
import sys
import os
import json
from datetime import datetime

import gymnasium as gym
import mlflow
import numpy as np
import tensorboard
from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import LoggerCallback, LoggerEvalCallback
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
from sinergym.utils.logger import *

# ---------------------------------------------------------------------------- #
#                             Parameters definition                            #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--configuration',
    '-conf',
    required=True,
    type=str,
    dest='configuration',
    help='Path to experiment configuration (JSON file)'
)
args = parser.parse_args()
# ------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------- #
#                             Read json parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration) as json_conf:
    conf = json.load(json_conf)

# ---------------------------------------------------------------------------- #
#                               Register run name                              #
# ---------------------------------------------------------------------------- #
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
name = conf['algorithm']['name'] + '-' + conf['environment'] + \
    '-episodes-' + str(conf['episodes'])
if conf.get('seed'):
    name += '-seed-' + str(conf['seed'])
if conf.get('id'):
    name += '-id-' + str(conf['id'])
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
    # sinergym and python versions
    mlflow.log_param('sinergym-version', sinergym.__version__)
    mlflow.log_param('python-version', sys.version)
    # Main
    mlflow.log_param('environment', conf['environment'])
    mlflow.log_param('episodes', conf['episodes'])
    mlflow.log_param('algorithm', conf['algorithm']['name'])
    mlflow.log_param('reward', conf['reward']['class'])
    # Optional
    mlflow.log_param('tensorboard', conf.get('tensorboard', False))
    mlflow.log_param(
        'log-interval',
        conf['algorithm'].get(
            'log_interval',
            False))
    mlflow.log_param('seed', conf.get('seed', False))
    if conf.get('cloud', False):
        mlflow.log_param(
            'remote-store',
            conf['cloud'].get(
                'remote_store',
                False))
    if conf.get('wrappers'):
        for key in conf['wrappers']:
            mlflow.log_param(key, True)
    mlflow.log_param('evaluation', bool(conf.get('evaluation', False)))
    if conf.get('evaluation'):
        mlflow.log_param(
            'evaluation-frequency',
            conf['evaluation'].get('eval_freq'))
        mlflow.log_param(
            'evaluation-length',
            conf['evaluation'].get('eval_length'))

    # algorithm params
    mlflow.log_params(conf['algorithm'].get('parameters'))
    # reward params
    mlflow.log_params(conf['reward'].get('parameters'))

    # ---------------------------------------------------------------------------- #
    #               Environment construction (with reward specified)               #
    # ---------------------------------------------------------------------------- #
    reward = eval(conf['reward']['class'])
    reward_kwargs = conf['reward']['parameters']

    env = gym.make(
        conf['environment'],
        reward=reward,
        reward_kwargs=reward_kwargs)

    # env for evaluation if is enabled
    eval_env = None
    if conf.get('evaluation'):
        eval_env = gym.make(
            conf['environment'],
            reward=reward,
            reward_kwargs=reward_kwargs)

    # ---------------------------------------------------------------------------- #
    #                                   Wrappers                                   #
    # ---------------------------------------------------------------------------- #
    if conf.get('wrappers'):
        for key, parameters in conf['wrappers'].items():
            wrapper_class = eval(key)
            for name, value in parameters.items():
                # parse str parameters to sinergym Callable or Objects if it is
                # required
                if isinstance(value, str):
                    if 'sinergym.' in value:
                        parameters[name] = eval(value)
            env = wrapper_class(env=env, ** parameters)
            if eval_env is not None:
                eval_env = wrapper_class(env=eval_env, ** parameters)

    # ---------------------------------------------------------------------------- #
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    algorithm_name = conf['algorithm']['name']
    algorithm_parameters = conf['algorithm']['parameters']
    if conf.get('model') is None:

        # --------------------------------------------------------#
        #                           DQN                          #
        # --------------------------------------------------------#
        if algorithm_name == 'SB3-DQN':

            model = DQN(env=env,
                        seed=conf.get('seed', None),
                        tensorboard_log=conf.get('tensorboard', None),
                        ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           DDPG                         #
        # --------------------------------------------------------#
        elif algorithm_name == 'SB3-DDPG':
            model = DDPG(env=env,
                         seed=conf.get('seed', None),
                         tensorboard_log=conf.get('tensorboard', None),
                         ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           A2C                          #
        # --------------------------------------------------------#
        elif algorithm_name == 'SB3-A2C':
            model = A2C(env=env,
                        seed=conf.get('seed', None),
                        tensorboard_log=conf.get('tensorboard', None),
                        ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           PPO                          #
        # --------------------------------------------------------#
        elif algorithm_name == 'SB3-PPO':
            model = PPO(env=env,
                        seed=conf.get('seed', None),
                        tensorboard_log=conf.get('tensorboard', None),
                        ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           SAC                          #
        # --------------------------------------------------------#
        elif algorithm_name == 'SB3-SAC':
            model = SAC(env=env,
                        seed=conf.get('seed', None),
                        tensorboard_log=conf.get('tensorboard', None),
                        ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           TD3                          #
        # --------------------------------------------------------#
        elif algorithm_name == 'SB3-TD3':
            model = TD3(env=env,
                        seed=conf.get('seed', None),
                        tensorboard_log=conf.get('tensorboard', None),
                        ** algorithm_parameters)
        # --------------------------------------------------------#
        #                           Error                        #
        # --------------------------------------------------------#
        else:
            raise RuntimeError(
                F'Algorithm specified [{algorithm_name} ] is not registered.')

    else:
        model_path = ''
        if 'gs://' in conf['model']:
            # Download from given bucket (gcloud configured with privileges)
            client = gcloud.init_storage_client()
            bucket_name = conf['model'].split('/')[2]
            model_path = conf['model'].split(bucket_name + '/')[-1]
            gcloud.read_from_bucket(client, bucket_name, model_path)
            model_path = './' + model_path
        else:
            model_path = conf['model']

        model = None
        if algorithm_name == 'SB3-DQN':
            model = DQN.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        elif algorithm_name == 'SB3-DDPG':
            model = DDPG.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        elif algorithm_name == 'SB3-A2C':
            model = A2C.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        elif algorithm_name == 'SB3-PPO':
            model = PPO.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        elif algorithm_name == 'SB3-SAC':
            model = SAC.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        elif algorithm_name == 'SB3-TD3':
            model = TD3.load(
                model_path, tensorboard_log=conf.get(
                    'tensorboard', None))
        else:
            raise RuntimeError('Algorithm specified is not registered.')

        model.set_env(env)

    # ---------------------------------------------------------------------------- #
    #       Calculating total training timesteps based on number of episodes       #
    # ---------------------------------------------------------------------------- #
    n_timesteps_episode = env.simulator._eplus_one_epi_len / \
        env.simulator._eplus_run_stepsize
    timesteps = conf['episodes'] * n_timesteps_episode - 1

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #
    callbacks = []

    # Set up Evaluation and saving best model
    if conf.get('evaluation'):
        eval_callback = LoggerEvalCallback(
            eval_env,
            best_model_save_path='best_model/' + name,
            log_path='best_model/' + name + '/',
            eval_freq=n_timesteps_episode *
            conf['evaluation']['eval_freq'],
            deterministic=True,
            render=False,
            n_eval_episodes=conf['evaluation']['eval_length'])
        callbacks.append(eval_callback)

    # Set up tensorboard logger
    if conf.get('tensorboard'):
        log_callback = LoggerCallback()
        callbacks.append(log_callback)
        # lets change default dir for TensorboardFormatLogger only
        tb_path = conf['tensorboard'] + '/' + name
        new_logger = configure(tb_path, ["tensorboard"])
        model.set_logger(new_logger)

    callback = CallbackList(callbacks)

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=conf['algorithm']['log_interval'])
    model.save(env.simulator._env_working_dir_parent + '/' + name)

    # If the algorithm doesn't reset or close the environment, this script will do it in
    # order to correctly log all the simulation data (Energyplus + Sinergym
    # logs)
    if env.simulator._episode_existed:
        env.close()

    # ---------------------------------------------------------------------------- #
    #          Mlflow artifacts storege and Google Cloud Bucket Storage            #
    # ---------------------------------------------------------------------------- #
    if conf.get('cloud'):
        if conf['cloud'].get('remote_store'):
            # Initiate Google Cloud client
            client = gcloud.init_storage_client()
            # Code for send output and tensorboard to common resource here.
            gcloud.upload_to_bucket(
                client,
                src_path=env.simulator._env_working_dir_parent,
                dest_bucket_name=conf['cloud']['remote_store'],
                dest_path=name)
            # Code for send output and tensorboard to mlflow artifacts.
            mlflow.log_artifacts(
                local_dir=env.simulator._env_working_dir_parent,
                artifact_path=name)
            if conf.get('evaluation'):
                gcloud.upload_to_bucket(
                    client,
                    src_path='best_model/' + name + '/',
                    dest_bucket_name=conf['cloud']['remote_store'],
                    dest_path='best_model/' + name + '/')
                mlflow.log_artifacts(
                    local_dir='best_model/' + name,
                    artifact_path='best_model/' + name)
            # If tensorboard is active (in local) we should send to mlflow
            if conf.get('tensorboard') and 'gs://' + \
                    conf['cloud']['remote_store'] not in conf.get('tensorboard'):
                gcloud.upload_to_bucket(
                    client,
                    src_path=conf['tensorboard'] + '/' + name + '/',
                    dest_bucket_name=conf['cloud']['remote_store'],
                    dest_path=os.path.abspath(conf['tensorboard']).split('/')[-1] + '/' + name + '/')
                mlflow.log_artifacts(
                    local_dir=conf['tensorboard'] + '/' + name,
                    artifact_path=os.path.abspath(conf['tensorboard']).split('/')[-1] + '/' + name)

            # gcloud.upload_to_bucket(
            #     client,
            #     src_path='mlruns/',
            #     dest_bucket_name=conf['cloud']['remote_store'],
            #     dest_path='mlruns/')

    # End mlflow run
    mlflow.end_run()

    # ---------------------------------------------------------------------------- #
    #                   Autodelete option if is a cloud resource                   #
    # ---------------------------------------------------------------------------- #
    if conf.get('cloud'):
        if conf['cloud'].get(
                'remote_store') and conf['cloud'].get('auto_delete'):
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['group_name'], token)
