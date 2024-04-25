import argparse
import json
import sys

import gymnasium as gym
import numpy as np
import wandb
from gymnasium.wrappers.normalize import NormalizeReward
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.common import is_wrapped
from sinergym.utils.constants import *
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
    help='Path to experiment configuration (JSON file)'
)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                             Read json parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration) as json_conf:
    conf = json.load(json_conf)

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
            name=evaluation_name + '_' + wandb.util.generate_id(),
            config=experiment_params,
            ** wandb_params
        )

    # --------------------- Overwrite environment parameters --------------------- #
    env_params = {}
    # Transform required str's into Callables
    if conf.get('env_params'):
        if conf['env_params'].get('reward'):
            conf['env_params']['reward'] = eval(conf['env_params']['reward'])
        if conf['env_params'].get('action_space'):
            conf['env_params']['action_space'] = eval(
                conf['env_params']['action_space'])

        env_params = conf['env_params']

    # ---------------------------------------------------------------------------- #
    #                            Environment definition                            #
    # ---------------------------------------------------------------------------- #
    env_params.update({'env_name': evaluation_name})
    env = gym.make(
        conf['environment'],
        ** env_params)
    env = Monitor(env)

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

    # ---------------------------------------------------------------------------- #
    #                                  Load Agent                                  #
    # ---------------------------------------------------------------------------- #
    # ------------------------ Weights and Bias model path ----------------------- #
    if conf.get('wandb'):
        if conf['wandb'].get('load_model'):
            # get model path
            artifact_tag = conf['wandb']['load_model'].get(
                'artifact_tag', 'latest')
            wandb_path = conf['wandb']['load_model']['entity'] + '/' + conf['wandb']['load_model']['project'] + \
                '/' + conf['wandb']['load_model']['artifact_name'] + ':' + artifact_tag
            # Download artifact
            artifact = run.use_artifact(wandb_path)
            artifact.get_path(conf['wandb']['load_model']
                              ['artifact_path']).download('.')
            # Set model path to local wandb file downloaded
            model_path = './' + conf['wandb']['load_model']['artifact_path']

    # -------------------------- Google cloud model path ------------------------- #
    elif 'gs://' in conf['model']:
        # Download from given bucket (gcloud configured with privileges)
        client = gcloud.init_storage_client()
        bucket_name = conf['model'].split('/')[2]
        model_path = conf['model'].split(bucket_name + '/')[-1]
        gcloud.read_from_bucket(client, bucket_name, model_path)
        model_path = './' + model_path
    # ----------------------------- Local model path ----------------------------- #
    else:
        model_path = conf['model']

    model = None
    algorithm_name = conf['algorithm']['name']
    if algorithm_name == 'SB3-DQN':
        model = DQN.load(model_path)
    elif algorithm_name == 'SB3-DDPG':
        model = DDPG.load(model_path)
    elif algorithm_name == 'SB3-A2C':
        model = A2C.load(model_path)
    elif algorithm_name == 'SB3-PPO':
        model = PPO.load(model_path)
    elif algorithm_name == 'SB3-SAC':
        model = SAC.load(model_path)
    elif algorithm_name == 'SB3-TD3':
        model = TD3.load(model_path)
    else:
        raise RuntimeError('Algorithm specified is not registered.')

    # ---------------------------------------------------------------------------- #
    #                             Execute loaded agent                             #
    # ---------------------------------------------------------------------------- #
    for i in range(conf['episodes']):
        obs, info = env.reset()
        rewards = []
        truncated = terminated = False
        current_month = 0
        while not (terminated or truncated):
            a, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(a)
            rewards.append(reward)
            if info['month'] != current_month:
                current_month = info['month']
                print(info['month'], sum(rewards))
        print(
            'Episode ',
            i,
            'Mean reward: ',
            np.mean(rewards),
            'Cumulative reward: ',
            sum(rewards))
    env.close()

    # ---------------------------------------------------------------------------- #
    #                                Wandb Artifacts                               #
    # ---------------------------------------------------------------------------- #

    if conf.get('wandb'):
        if conf['wandb'].get('evaluation_registry'):
            artifact = wandb.Artifact(
                name=conf['wandb']['evaluation_registry']['artifact_name'],
                type=conf['wandb']['evaluation_registry']['artifact_type'])
            artifact.add_dir(
                env.get_wrapper_attr('workspace_path'),
                name='evaluation_output/')

            run.log_artifact(artifact)

        # wandb has finished
        run.finish()

    # ---------------------------------------------------------------------------- #
    #                                 Store results                                #
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
                dest_path=evaluation_name)

        # ---------------------------------------------------------------------------- #
        #                          Auto-delete remote container                        #
        # ---------------------------------------------------------------------------- #
        if conf['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                conf['cloud']['group_name'], token)

except Exception as err:
    print("Error in process detected")

    # Save current wandb artifacts state
    if conf.get('wandb'):
        if conf['wandb'].get('evaluation_registry'):
            artifact = wandb.Artifact(
                name=conf['wandb']['evaluation_registry']['artifact_name'],
                type=conf['wandb']['evaluation_registry']['artifact_type'])
            artifact.add_dir(
                env.get_wrapper_attr('workspace_path'),
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
