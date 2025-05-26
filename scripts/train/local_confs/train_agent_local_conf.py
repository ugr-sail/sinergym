import argparse
import sys
import traceback
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb
import yaml
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from sinergym.utils.wrappers import WandBLogger

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import *
from sinergym.utils.common import (
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
    create_environment,
    import_from_path,
    deep_update
)
from sinergym.utils.logger import WandBOutputFormat

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
    help='Path to experiment configuration (YAML file)'
)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                             Read yaml parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration, 'r') as yaml_conf:
    conf = yaml.safe_load(yaml_conf)


# ---------------------------------------------------------------------------- #
#                            Train script execution                            #
# ---------------------------------------------------------------------------- #

try:
    # ---------------------------------------------------------------------------- #
    #                               Register run name                              #
    # ---------------------------------------------------------------------------- #
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
    if conf.get('experiment_name'):
        experiment_name = f'{conf['experiment_name']}_{experiment_date}'
    else:
        alg_name = conf['algorithm']['name'].split(':')[-1]
        experiment_name = f'{alg_name}_{experiment_date}'

    # ---------------------------------------------------------------------------- #
    #                Load model as starting point if defined                       #
    # ---------------------------------------------------------------------------- #
    model_path = None
    if conf.get('model'):
        # ---------------------------- Local model path ----------------------------- #
        if conf['model'].get('local_path'):
            model_path = conf['model']['local_path']

        # ------------------------ Weights and Bias model path ----------------------- #
        if conf['model'].get('entity'):

            api = wandb.Api()
            # Get model path
            artifact_tag = conf['model'].get(
                'artifact_tag', 'latest')
            wandb_path = f'{
                conf['model']['entity']}/{
                conf['model']['project']}/{
                conf['model']['artifact_name']}:{artifact_tag}'

            # Download artifact
            artifact = api.artifact(wandb_path)
            artifact.download(
                path_prefix=conf['model']['artifact_path'],
                root='./')

            # Set model path to local wandb downloaded file
            model_path = f'./{conf['model']['model_path']}'

        # -------------------------- Google cloud model path ------------------------- #
        if conf['model'].get('bucket_path'):
            # Download from given bucket (gcloud configured with privileges)
            client = gcloud.init_storage_client()
            bucket_name = conf['model']['bucket_path'].split('/')[2]
            model_path = f'{
                conf['model']['bucket_path'].split(
                    bucket_name + '/')[
                    -1]}'
            gcloud.read_from_bucket(client, bucket_name, model_path)
            model_path = f'./{model_path}'

    # ---------------------------------------------------------------------------- #
    #                           Environment parameters                              #
    # ---------------------------------------------------------------------------- #
    env_params = {}

    # ------- Update env params configuration with env yaml file if exists ------- #
    if conf.get('env_yaml_config'):
        with open(conf['env_yaml_config'], 'r') as env_yaml_conf:
            env_params.update(yaml.load(env_yaml_conf, Loader=yaml.FullLoader))

    # -- Update env params configuration with specified env parameters if exists -- #
    if conf.get('env_params'):
        env_params = deep_update(
            env_params, process_environment_parameters(
                conf['env_params']))

    # ------------ For this script, the execution name will be updated ----------- #
    env_params.update({'env_name': experiment_name})

    # ---------------------------------------------------------------------------- #
    #                            Wrappers definition                               #
    # ---------------------------------------------------------------------------- #
    wrappers = {}

    # ------------------ Read wrappers from yaml file if exists ------------------ #
    if conf.get('wrappers_yaml_config'):
        with open(conf['wrappers_yaml_config'], 'r') as f:
            wrappers = yaml.load(f, Loader=yaml.FullLoader)

    # ------ Read wrappers from yaml file and overwrite yaml file if exists ------ #
    if conf.get('wrappers'):
        # Update wrappers with the ones defined in the yaml file
        for wrapper in conf['wrappers']:
            for wrapper_name, wrapper_arguments in wrapper.items():
                for name, value in wrapper_arguments.items():
                    # parse str parameters to sinergym Callable or Objects if
                    # required
                    if isinstance(value, str):
                        if ':' in value:
                            wrapper_arguments[name] = import_from_path(value)
            wrappers = deep_update(wrappers, {wrapper_name: wrapper_arguments})

    # ---------------------------------------------------------------------------- #
    #                Create environment with parameters and wrappers               #
    # ---------------------------------------------------------------------------- #

    env = create_environment(
        env_id=conf['environment'],
        env_params=env_params,
        wrappers=wrappers)

    # ---------------------------------------------------------------------------- #
    #                 Register hyperparameters in wandb if enabled                 #
    # ---------------------------------------------------------------------------- #
    if is_wrapped(env, WandBLogger):
        experiment_params = {
            'sinergym-version': sinergym.__version__,
            'python-version': sys.version,
            'stable-baselines3-version': sb3_version
        }

        wandb.run.config.update(experiment_params)
        wandb.run.config.update(conf)
        # Overwrite env_params with the full environment parameters
        wandb.run.config.update(
            {'env_params': env.get_wrapper_attr('to_dict')()}, allow_val_change=True)

    # ---------------------------------------------------------------------------- #
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    alg_name = conf['algorithm']['name']
    alg_cls = import_from_path(alg_name)
    alg_params = conf['algorithm'].get(
        'parameters', {'policy': 'MlpPolicy'})
    alg_params = process_algorithm_parameters(alg_params)

    # --------------------------- Training from scratch -------------------------- #
    if model_path is None:
        try:
            model = alg_cls(env=env, ** alg_params)
        except NameError:
            raise NameError(
                'Algorithm {} does not exists. It must be a valid SB3 algorithm.'.format(alg_name))

    # --------------------- Traning from a pre-trained model --------------------- #
    else:
        model = None
        try:
            model = alg_cls.load(
                model_path)
        except NameError:
            raise NameError(
                'Algorithm {} does not exists. It must be a valid SB3 algorithm.'.format(alg_name))

        model.set_env(env)

    # ---------------------------------------------------------------------------- #
    #                              SET UP WANDB LOGGER                             #
    # ---------------------------------------------------------------------------- #
    if is_wrapped(env, WandBLogger):
        # wandb and SB3 logger
        logger = SB3Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(
                    sys.stdout,
                    max_length=120),
                WandBOutputFormat()])
        model.set_logger(logger)

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #
    callbacks = []

    # ---------------------------- EVALUATION CALLBACK --------------------------- #
    if conf.get('evaluation'):

        # ------------ Preparing the evaluation environment configuration ------------ #
        env_params['env_name'] = experiment_name + '_EVALUATION'

        # By default, the evaluation environment does not use WandBLogger
        if wrappers:
            key_to_remove = [
                key for key in wrappers if 'WandBLogger' in key][0]
            del wrappers[key_to_remove]

        # ----------------------- Create evaluation environment ---------------------- #
        eval_env = create_environment(
            env_id=conf['environment'],
            env_params=env_params,
            wrappers=wrappers)

        # ------------------------ Create evaluation callback ------------------------ #
        eval_callback = LoggerEvalCallback(
            eval_env=eval_env,
            train_env=env,
            n_eval_episodes=conf['evaluation']['eval_length'],
            eval_freq_episodes=conf['evaluation']['eval_freq'],
            deterministic=True,
            excluded_metrics=[
                'episode_num',
                'length (timesteps)',
                'time_elapsed (hours)'],
            verbose=1)

        callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    # ---------------------------------------------------------------------------- #
    #                                   TRAINING                                   #
    # ---------------------------------------------------------------------------- #
    timesteps = conf['episodes'] * \
        (env.get_wrapper_attr('timestep_per_episode'))

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=conf['algorithm']['log_interval'])

    model.save(env.get_wrapper_attr('workspace_path') + '/model')

    # If the environment is not closed, this script will do it in
    # order to correctly log all the simulation data (Energyplus + Sinergym
    # logs)
    if env.get_wrapper_attr('is_running'):
        env.close()

    # ---------------------------------------------------------------------------- #
    #                      Google Cloud Bucket Storage                             #
    # ---------------------------------------------------------------------------- #
    if conf.get('cloud'):
        if conf['cloud'].get('remote_store'):
            # Initiate Google Cloud client
            client = gcloud.init_storage_client()
            # Send output to common Google Cloud resource
            gcloud.upload_to_bucket(
                client,
                src_path=env.get_wrapper_attr('workspace_path'),
                dest_bucket_name=conf['cloud']['remote_store'],
                dest_path=experiment_name)
        # ---------------------------------------------------------------------------- #
        #                   Autodelete option if is a cloud resource                   #
        # ---------------------------------------------------------------------------- #
        if conf['cloud'].get('auto_delete'):
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['auto_delete']['group_name'], token)

# If there is some error in the code, delete remote container if exists
# include KeyboardInterrupt

except (Exception, KeyboardInterrupt) as err:
    print("Error or interruption in process detected")
    print(traceback.print_exc(), file=sys.stderr)

    # Current model state save
    model.save(env.get_wrapper_attr('workspace_path') + '/model')

    env.close()

    # Auto delete
    if conf.get('cloud'):
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['auto_delete']['group_name'], token)
    raise err
