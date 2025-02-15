import argparse
import logging
import sys

import gymnasium as gym
import numpy as np
import wandb
import yaml
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.common import (
    is_wrapped,
    process_algorithm_parameters,
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
    evaluation_name = conf['algorithm']['name'] + '-' + conf['environment'] + \
        '-episodes-' + str(conf['episodes'])
    if conf.get('id'):
        evaluation_name += '-id-' + str(conf['id'])
    evaluation_name += '_' + evaluation_date

    # --------------------- Overwrite environment parameters --------------------- #
    env_params = {}
    # Transform required str's into Callables
    if conf.get('env_params'):
        env_params = process_environment_parameters(conf['env_params'])

    # ---------------------------------------------------------------------------- #
    #                            Environment definition                            #
    # ---------------------------------------------------------------------------- #
    env_params.update({'env_name': evaluation_name})
    env = gym.make(
        conf['environment'],
        ** env_params)

    # ---------------------------------------------------------------------------- #
    #                                   Wrappers                                   #
    # ---------------------------------------------------------------------------- #
    if conf.get('wrappers'):
        for key, parameters in conf['wrappers'].items():
            wrapper_class = eval(key)
            for name, value in parameters.items():
                # parse str parameters to sinergym Callable or Objects if
                # required
                if isinstance(value, str):
                    if '.' in value and '.txt' not in value:
                        parameters[name] = eval(value)
            env = wrapper_class(env=env, ** parameters)

    # ---------------------------------------------------------------------------- #
    #                                  Load Agent                                  #
    # ---------------------------------------------------------------------------- #

    # ------------------------ Weights and Bias model path ----------------------- #
    if conf['model'].get('entity'):
        # Get wandb run or generate a new one
        if is_wrapped(env, WandBLogger):
            wandb_run = env.get_wrapper_attr('wandb_run')
        else:
            wandb_run = wandb.init()

        # Get model path
        artifact_tag = conf['model'].get(
            'artifact_tag', 'latest')
        wandb_path = conf['model']['entity'] + '/' + conf['model']['project'] + \
            '/' + conf['model']['artifact_name'] + ':' + artifact_tag

        # Download artifact
        artifact = wandb_run.use_artifact(wandb_path)
        artifact.download(
            path_prefix=conf['model']['artifact_path'],
            root='./')

        # Set model path to local wandb downloaded file
        model_path = './' + conf['model']['model_path']

    # -------------------------- Google cloud model path ------------------------- #
    if conf['model'].get('bucket_path'):
        # Download from given bucket (gcloud configured with privileges)
        client = gcloud.init_storage_client()
        bucket_name = conf['model']['bucket_path'].split('/')[2]
        model_path = conf['model']['bucket_path'].split(bucket_name + '/')[-1]
        gcloud.read_from_bucket(client, bucket_name, model_path)
        model_path = './' + model_path

    # ----------------------------- Local model path ----------------------------- #
    if conf['model'].get('local_path'):
        model_path = conf['model']['local_path']

    # ---------- Load calibration of normalization for model if required --------- #
    if conf['model'].get('normalization') and is_wrapped(
            env, NormalizeObservation):
        # Update calibrations
        env.get_wrapper_attr('set_mean')(
            conf['model']['normalization']['mean'])
        env.get_wrapper_attr('set_var')(
            conf['model']['normalization']['var'])

    model = None
    alg_name = conf['algorithm']['name']
    try:
        model = eval(alg_name).load(model_path)
    except NameError:
        raise NameError(
            'Algorithm {} does not exists. It must be a valid SB3 algorithm.'.format(alg_name))

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

    env.close()

    # Auto delete
    if conf.get('cloud'):
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['auto_delete']['group_name'], token)
    raise err
