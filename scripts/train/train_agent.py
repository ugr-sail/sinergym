import argparse
import json
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb
from gymnasium.wrappers.normalize import NormalizeReward
from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import *
from sinergym.utils.common import is_wrapped
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# ---------------------------------------------------------------------------- #
#                       Function to process configuration                      #
# ---------------------------------------------------------------------------- #


def process_environment_parameters(env_params: dict) -> dict:

    # Transform required str's into Callables or list in tuples
    if env_params.get('action_space'):
        env_params['action_space'] = eval(
            env_params['action_space'])

    if env_params.get('variables'):
        for variable_name, components in env_params['variables'].items():
            env_params['variables'][variable_name] = tuple(components)

    if env_params.get('actuators'):
        for actuator_name, components in env_params['actuators'].items():
            env_params['actuators'][actuator_name] = tuple(components)

    if env_params.get('weather_variability'):
        env_params['weather_variability'] = tuple(
            env_params['weather_variability'])

    if env_params.get('reward'):
        env_params['reward'] = eval(env_params['reward'])

    if env_params.get('reward_kwargs'):
        for reward_param_name, reward_value in env_params.items():
            if reward_param_name in [
                'range_comfort_winter',
                'range_comfort_summer',
                'summer_start',
                    'summer_final']:
                env_params['reward_kwargs'][reward_param_name] = tuple(
                    reward_value)

    if env_params.get('config_params'):
        if env_params['config_params'].get('runperiod'):
            env_params['config_params']['runperiod'] = tuple(
                env_params['config_params']['runperiod'])

    # Add more keys if it is needed

    return env_params


def process_algorithm_parameters(alg_params: dict):

    # Transform required str's into Callables or list in tuples
    if alg_params.get('train_freq') and isinstance(
            alg_params.get('train_freq'), list):
        alg_params['train_freq'] = tuple(alg_params['train_freq'])

    if alg_params.get('action_noise'):
        alg_params['action_noise'] = eval(alg_params['action_noise'])
    # Add more keys if it is needed

    return alg_params


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

try:
    # ---------------------------------------------------------------------------- #
    #                               Register run name                              #
    # ---------------------------------------------------------------------------- #
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = conf['algorithm']['name'] + '-' + conf['environment'] + \
        '-episodes-' + str(conf['episodes'])
    if conf.get('seed'):
        experiment_name += '-seed-' + str(conf['seed'])
    if conf.get('id'):
        experiment_name += '-id-' + str(conf['id'])
    experiment_name += '_' + experiment_date

    # ---------------------------------------------------------------------------- #
    #                              WandB registration                              #
    # ---------------------------------------------------------------------------- #

    if conf.get('wandb'):
        # Create wandb.config object in order to log all experiment params
        experiment_params = {
            'sinergym-version': sinergym.__version__,
            'python-version': sys.version
        }
        experiment_params.update(conf)

        # Get wandb init params
        wandb_params = conf['wandb']['init_params']
        # Init wandb entry
        run = wandb.init(
            name=experiment_name + '_' + wandb.util.generate_id(),
            config=experiment_params,
            ** wandb_params
        )

    # --------------------- Overwrite environment parameters --------------------- #
    env_params = conf.get('env_params', {})
    env_params = process_environment_parameters(env_params)

    # ---------------------------------------------------------------------------- #
    #                           Environment construction                           #
    # ---------------------------------------------------------------------------- #
    # For this script, the execution name will be updated
    env_params.update({'env_name': experiment_name})
    env = gym.make(
        conf['environment'],
        ** env_params)

    # env for evaluation if is enabled
    eval_env = None
    if conf.get('evaluation'):
        eval_name = conf['evaluation'].get(
            'name', env.get_wrapper_attr('name') + '-EVAL')
        env_params.update({'env_name': eval_name})
        eval_env = gym.make(
            conf['environment'],
            ** env_params)

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
                    if '.' in value:
                        parameters[name] = eval(value)
            env = wrapper_class(env=env, ** parameters)
            if eval_env is not None:
                eval_env = wrapper_class(env=eval_env, ** parameters)

    # ---------------------------------------------------------------------------- #
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    alg_name = conf['algorithm']['name']
    alg_params = conf['algorithm'].get(
        'parameters', {'policy': 'MlpPolicy'})
    alg_params = process_algorithm_parameters(alg_params)

    if conf.get('model') is None:

        # --------------------------------------------------------#
        #                           DQN                          #
        # --------------------------------------------------------#
        if alg_name == 'SB3-DQN':

            model = DQN(env=env,
                        ** alg_params)
        # --------------------------------------------------------#
        #                           DDPG                         #
        # --------------------------------------------------------#
        elif alg_name == 'SB3-DDPG':
            model = DDPG(env=env,

                         ** alg_params)
        # --------------------------------------------------------#
        #                           A2C                          #
        # --------------------------------------------------------#
        elif alg_name == 'SB3-A2C':
            model = A2C(env=env,
                        ** alg_params)
        # --------------------------------------------------------#
        #                           PPO                          #
        # --------------------------------------------------------#
        elif alg_name == 'SB3-PPO':
            model = PPO(env=env,
                        ** alg_params)
        # --------------------------------------------------------#
        #                           SAC                          #
        # --------------------------------------------------------#
        elif alg_name == 'SB3-SAC':
            model = SAC(env=env,
                        ** alg_params)
        # --------------------------------------------------------#
        #                           TD3                          #
        # --------------------------------------------------------#
        elif alg_name == 'SB3-TD3':
            model = TD3(env=env,
                        ** alg_params)
        # --------------------------------------------------------#
        #                           Error                        #
        # --------------------------------------------------------#
        else:
            raise RuntimeError(
                F'Algorithm specified [{alg_name} ] is not registered.')

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
        if alg_name == 'SB3-DQN':
            model = DQN.load(
                model_path)
        elif alg_name == 'SB3-DDPG':
            model = DDPG.load(
                model_path)
        elif alg_name == 'SB3-A2C':
            model = A2C.load(
                model_path)
        elif alg_name == 'SB3-PPO':
            model = PPO.load(
                model_path)
        elif alg_name == 'SB3-SAC':
            model = SAC.load(
                model_path)
        elif alg_name == 'SB3-TD3':
            model = TD3.load(
                model_path)
        else:
            raise RuntimeError('Algorithm specified is not registered.')

        model.set_env(env)

    # ---------------------------------------------------------------------------- #
    #       Calculating total training timesteps based on number of episodes       #
    # ---------------------------------------------------------------------------- #
    timesteps = conf['episodes'] * \
        (env.get_wrapper_attr('timestep_per_episode') - 1)

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #
    callbacks = []

    # Set up Evaluation and saving best model
    if conf.get('evaluation'):
        eval_callback = LoggerEvalCallback(
            eval_env=eval_env,
            train_env=env,
            best_model_save_path=eval_env.get_wrapper_attr('workspace_path') +
            '/best_model/',
            log_path=eval_env.get_wrapper_attr('workspace_path') +
            '/best_model/',
            eval_freq=eval_env.get_wrapper_attr('timestep_per_episode') *
            conf['evaluation']['eval_freq'],
            deterministic=True,
            render=False,
            n_eval_episodes=conf['evaluation']['eval_length'])
        callbacks.append(eval_callback)

    # Set up wandb logger
    if conf.get('wandb'):
        # wandb logger and setting in SB3
        logger = SB3Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(
                    sys.stdout,
                    max_length=120),
                WandBOutputFormat()])
        model.set_logger(logger)
        # Append callback
        dump_frequency = conf['wandb'].get('dump_frequency', 100)
        log_callback = LoggerCallback(dump_frequency=dump_frequency)
        callbacks.append(log_callback)

    callback = CallbackList(callbacks)

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=conf['algorithm']['log_interval'])
    model.save(env.get_wrapper_attr('workspace_path') + '/model')

    # If the algorithm doesn't reset or close the environment, this script will do it in
    # order to correctly log all the simulation data (Energyplus + Sinergym
    # logs)
    if env.get_wrapper_attr('is_running'):
        env.close()

    # ---------------------------------------------------------------------------- #
    #                              Wandb artifact log                              #
    # ---------------------------------------------------------------------------- #

    if conf.get('wandb'):
        artifact = wandb.Artifact(
            name=conf['wandb']['artifact_name'],
            type=conf['wandb']['artifact_type'])
        artifact.add_dir(
            env.get_wrapper_attr('workspace_path'),
            name='training_output/')
        if conf.get('evaluation'):
            artifact.add_dir(
                eval_env.get_wrapper_attr('workspace_path'),
                name='evaluation_output/')
        run.log_artifact(artifact)

        # wandb has finished
        run.finish()

    # ---------------------------------------------------------------------------- #
    #                      Google Cloud Bucket Storage                             #
    # ---------------------------------------------------------------------------- #
    if conf.get('cloud'):
        if conf['cloud'].get('remote_store'):
            # Initiate Google Cloud client
            client = gcloud.init_storage_client()
            # Code for send output to common Google Cloud resource here.
            gcloud.upload_to_bucket(
                client,
                src_path=env.get_wrapper_attr('workspace_path'),
                dest_bucket_name=conf['cloud']['remote_store'],
                dest_path=experiment_name)
            if conf.get('evaluation'):
                gcloud.upload_to_bucket(
                    client,
                    src_path='best_model/' + experiment_name + '/',
                    dest_bucket_name=conf['cloud']['remote_store'],
                    dest_path='best_model/' + experiment_name + '/')
        # ---------------------------------------------------------------------------- #
        #                   Autodelete option if is a cloud resource                   #
        # ---------------------------------------------------------------------------- #
        if conf['cloud'].get('auto_delete'):
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['group_name'], token)

# If there is some error in the code, delete remote container if exists
except Exception as err:
    print("Error in process detected")

    # Current model state save
    model.save(env.get_wrapper_attr('workspace_path') + '/model')

    # Save current wandb artifacts state
    if conf.get('wandb'):
        artifact = wandb.Artifact(
            name=conf['wandb']['artifact_name'],
            type=conf['wandb']['artifact_type'])
        artifact.add_dir(
            env.get_wrapper_attr('workspace_path'),
            name='training_output/')
        if conf.get('evaluation'):
            artifact.add_dir(
                eval_env.get_wrapper_attr('workspace_path'),
                name='evaluation_output/')
        run.log_artifact(artifact)

        # wandb has finished
        run.finish()

    # Auto delete
    if conf.get('cloud'):
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['group_name'], token)
    raise err
