import sys
import traceback
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import *
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import *
from sinergym.utils.common import (
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
)
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *


def get_evaluation(env_params: Dict, train_env: gym.Env):
    """
    Creates an evaluation environment and evaluation callback for the environment.

    Args:
        env_params (dict): Parameters for creating the evaluation environment.
        train_env (gym.Env): Original training env in order to use to syncronize with evaluation if it was required.

    Returns:
        tuple: A tuple containing the evaluation environment and evaluation callback.
    """

    # Make evaluation environment
    params = env_params.copy()
    environment = wandb.config['environment']
    params.update({'env_name': environment + '-EVAL'})
    eval_env = gym.make(environment, **params)

    # Wrapper for evaluation environment
    if wandb.config.get('wrappers'):
        for wrapper in wandb.config['wrappers']:
            for key, parameters in wrapper.items():
                # In evaluation, WandBLogger is not required
                if key != 'WandBLogger':
                    wrapper_class = eval(key)
                    for name, value in parameters.items():
                        if isinstance(value, str):
                            # A item that must be evaluated if '.' is present
                            if '.' in value:
                                parameters[name] = eval(value)
                    eval_env = wrapper_class(
                        env=eval_env, ** parameters)

    # Make evaluation callback for environment
    eval_length = wandb.config['evaluation']['eval_length']
    eval_freq = wandb.config['evaluation']['eval_freq']
    eval_callback = LoggerEvalCallback(
        eval_env=eval_env,
        train_env=train_env,
        n_eval_episodes=eval_length,
        eval_freq_episodes=eval_freq,
        deterministic=True,
        excluded_metrics=[
            'episode_num',
            'length (timesteps)',
            'time_elapsed (hours)'],
        verbose=1)

    return eval_env, eval_callback


def train():
    try:
        # ---------------------------------------------------------------------------- #
        #                                  WandB init                                  #
        # ---------------------------------------------------------------------------- #
        # Default configuration (it could be override by sweeps totally or
        # partially)
        default_config = {
            'sinergym-version': sinergym.__version__,
            'python-version': sys.version,
            'stable-baselines3-version': sb3_version
        }
        run = wandb.init(config=default_config)

        # ------------------------------ Experiment name ----------------------------- #
        experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        experiment_name = 'SB3_' \
            + wandb.config['algorithm'] + '_' \
            + experiment_date + '_' \
            + wandb.config['environment']
        run.name = experiment_name

        # ---------------------------------------------------------------------------- #
        #                             Environment creation                             #
        # ---------------------------------------------------------------------------- #
        environment = wandb.config['environment']
        if wandb.config.get('environment_parameters'):
            env_params = wandb.config['environment_parameters']
            # Process types from yaml
            env_params = process_environment_parameters(env_params)
        env = gym.make(environment, **env_params)

        # ---------------------------------------------------------------------------- #
        #                           Application of wrapper(s)                          #
        # ---------------------------------------------------------------------------- #

        if wandb.config.get('wrappers'):
            for wrapper in wandb.config['wrappers']:
                for key, parameters in wrapper.items():
                    wrapper_class = eval(key)
                    for name, value in parameters.items():
                        if isinstance(value, str):
                            # A item that must be evaluated if '.' is present
                            if '.' in value:
                                parameters[name] = eval(value)
                    env = wrapper_class(env=env, ** parameters)

        assert is_wrapped(
            env, WandBLogger), 'Environments with sweeps must be wrapped with WandBLogger.'

        # ---------------------------------------------------------------------------- #
        #                           DRL model initialization                           #
        # ---------------------------------------------------------------------------- #
        algorithm_parameters = process_algorithm_parameters(
            wandb.config.get('algorithm_parameters', {'policy': 'MlpPolicy'}))
        algorithm_class = eval(wandb.config['algorithm'])

        # ------------------------ Training from scratch case ------------------------ #
        if wandb.config.get('model', None) is None:

            model = algorithm_class(env=env, **algorithm_parameters)

        # ------------------------ Training from a given model ----------------------- #
        else:
            # --------------------------- Local model path case -------------------------- #
            if wandb.config['model'].get('local_path'):
                model_path = wandb.config['model']['local_path']

            # --------------------- Weights and Bias model path case --------------------- #
            if wandb.config['model'].get('entity'):
                # Get model path
                artifact_tag = wandb.config['model'].get(
                    'artifact_tag', 'latest')
                wandb_path = wandb.config['model']['entity'] + '/' + wandb.config['model']['project'] + \
                    '/' + wandb.config['model']['artifact_name'] + ':' + artifact_tag

                # Download artifact
                artifact = run.use_artifact(wandb_path)
                artifact.download(
                    path_prefix=wandb.config['model']['artifact_path'],
                    root='./')

                # Set model path to local wandb downloaded file
                model_path = './' + wandb.config['model']['model_path']

            # ----------------------- Google cloud model path case ----------------------- #
            if wandb.config['model'].get('bucket_path'):
                # Download from given bucket (gcloud configured with
                # privileges)
                client = gcloud.init_storage_client()
                bucket_name = wandb.config['model']['bucket_path'].split(
                    '/')[2]
                model_path = wandb.config['model']['bucket_path'].split(
                    bucket_name + '/')[-1]
                gcloud.read_from_bucket(client, bucket_name, model_path)
                model_path = './' + model_path

            # ---------- Load calibration of normalization for model if required --------- #
            if wandb.config['model'].get('normalization') and is_wrapped(
                    env, NormalizeObservation):
                # Update calibrations
                env.get_wrapper_attr('set_mean')(
                    wandb.config['model']['normalization']['mean'])
                env.get_wrapper_attr('set_var')(
                    wandb.config['model']['normalization']['var'])

            # ------------------------ Load model from model_path ------------------------ #
            model = algorithm_class.load(model_path)
            model.set_env(env)

        # ---------------------------------------------------------------------------- #
        #                          Application of callback(s)                          #
        # ---------------------------------------------------------------------------- #
        callbacks = []

        # Set WandB session in SB3 model native logger
        logger = SB3Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(
                    sys.stdout,
                    max_length=120),
                WandBOutputFormat()])
        model.set_logger(logger)

        # Evaluation Callback if evaluations are specified
        evaluation = wandb.config.get('evaluation', False)
        if evaluation:
            eval_env, evaluation_callback = get_evaluation(env_params, env)
            callbacks.append(evaluation_callback)

        callback = CallbackList(callbacks)

        # ---------------------------------------------------------------------------- #
        #                                 DRL training                                 #
        # ---------------------------------------------------------------------------- #
        timesteps = wandb.config['episodes'] * \
            (env.get_wrapper_attr('timestep_per_episode') - 1)
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=wandb.config['log_interval'])

        # ---------------------------------------------------------------------------- #
        #                                Saving results                                #
        # ---------------------------------------------------------------------------- #
        model.save(env.get_wrapper_attr('workspace_path') + '/model')

        # ---------------------------------------------------------------------------- #
        #                            Google Cloud (optional)                           #
        # ---------------------------------------------------------------------------- #
        if wandb.config.get('cloud'):
            # ---------------------------------------------------------------------------- #
            #                      Google Cloud Bucket Storage                             #
            # ---------------------------------------------------------------------------- #
            if wandb.config['cloud'].get('remote_store'):
                # Initiate Google Cloud client
                client = gcloud.init_storage_client()
                # Send output to common Google Cloud resource
                gcloud.upload_to_bucket(
                    client,
                    src_path=env.get_wrapper_attr('workspace_path'),
                    dest_bucket_name=wandb.config['cloud']['remote_store'],
                    dest_path=experiment_name)

        # ---------------------------------------------------------------------------- #
        #                               Close environment                              #
        # ---------------------------------------------------------------------------- #
        if env.get_wrapper_attr('is_running'):
            env.close()

    # ---------------------------------------------------------------------------- #
    #                             Exception management                             #
    # ---------------------------------------------------------------------------- #
    except (Exception, KeyboardInterrupt) as err:
        print("Error or interruption in process detected")
        print(traceback.print_exc(), file=sys.stderr)

        # Current model state save
        model.save(env.get_wrapper_attr('workspace_path') + '/model')

        env.close()
        raise err
