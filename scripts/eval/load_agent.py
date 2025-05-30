import argparse
import logging
import sys
import traceback

import gymnasium as gym
import numpy as np
import wandb
import yaml
from stable_baselines3.common.monitor import Monitor

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.common import (
    create_environment,
    deep_update,
    import_from_path,
    process_environment_parameters,
)
from sinergym.utils.constants import *
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# ---------------------------------------------------------------------------- #
#                                  Parameters                                  #
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

# Optional: Terminal log in the same format as Sinergym.
# Logger info can be replaced by print.
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='EVALUATION',
    level=logging.INFO
)

# ---------------------------------------------------------------------------- #
#                             Read yaml parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration, 'r') as yaml_conf:
    conf = yaml.safe_load(yaml_conf)

try:
    # ---------------------------------------------------------------------------- #
    #                                Evaluation name                               #
    # ---------------------------------------------------------------------------- #
    evaluation_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    if conf.get('experiment_name'):
        evaluation_name = f'{conf['experiment_name']}_{evaluation_date}'
    else:
        alg_name = conf['algorithm']['name'].split(':')[-1]
        evaluation_name = f'{alg_name}_{evaluation_date}_evaluation'

    # ---------------------------------------------------------------------------- #
    #                                Get model path                                #
    # ---------------------------------------------------------------------------- #
    model_path = None

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
    #                           Environment parameters                             #
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
    env_params.update({'env_name': evaluation_name})

    # ---------------------------------------------------------------------------- #
    #                            Wrappers definition                               #
    # ---------------------------------------------------------------------------- #
    wrappers = {}

    # ------------------ Read wrappers from yaml file if exists ------------------ #
    if conf.get('wrappers_yaml_config'):
        with open(conf['wrappers_yaml_config'], 'r') as f:
            wrappers = yaml.load(f, Loader=yaml.FullLoader)

    # ----------------------- Delete WandBLogger by default ---------------------- #
    if wrappers:
        key_to_remove = [
            key for key in wrappers if 'WandBLogger' in key][0]
        del wrappers[key_to_remove]

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
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    alg_name = conf['algorithm']['name']
    alg_cls = import_from_path(alg_name)
    model = alg_cls.load(model_path)

    # ---------------------------------------------------------------------------- #
    #                             Execute loaded agent                             #
    # ---------------------------------------------------------------------------- #
    for i in range(conf['episodes']):
        # Reset the environment to start a new episode
        obs, info = env.reset()
        truncated = terminated = False
        while not (terminated or truncated):
            # Use the agent to predict the next action
            a, _ = model.predict(obs, deterministic=True)
            # Read observation and reward
            obs, reward, terminated, truncated, info = env.step(a)

    env.close()

    if conf.get('cloud'):
        # ---------------------------------------------------------------------------- #
        #                                 Store results                                #
        # ---------------------------------------------------------------------------- #
        if conf['cloud'].get('remote_store'):
            # Initiate Google Cloud client
            client = gcloud.init_storage_client()
            # Send output to common Google Cloud resource
            gcloud.upload_to_bucket(
                client,
                src_path=env.get_wrapper_attr('workspace_path'),
                dest_bucket_name=conf['cloud']['remote_store'],
                dest_path=evaluation_name)

        # ---------------------------------------------------------------------------- #
        #                          Auto-delete remote container                        #
        # ---------------------------------------------------------------------------- #
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['auto_delete']['group_name'], token)

except (Exception, KeyboardInterrupt) as err:
    print("Error or interruption in process detected")
    print(traceback.print_exc(), file=sys.stderr)

    env.close()

    # Auto delete
    if conf.get('cloud'):
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['auto_delete']['group_name'], token)
    raise err
