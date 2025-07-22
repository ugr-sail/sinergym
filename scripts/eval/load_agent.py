import argparse
import logging
from datetime import datetime
import sys
import traceback

import wandb
import yaml

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.common import (
    create_environment,
    deep_update,
    import_from_path,
    process_environment_parameters,
)

from sinergym.utils.logger import TerminalLogger

# ---------------------------------------------------------------------------- #
#                                Terminal Logger                               #
# ---------------------------------------------------------------------------- #
# Optional: Terminal log in the same format as Sinergym.
# Logger info can be replaced by print.
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='EVALUATION',
    level=logging.INFO
)

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

# ---------------------------------------------------------------------------- #
#                             Read yaml parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration, 'r') as yaml_conf:
    config = yaml.safe_load(yaml_conf)

try:
    # ---------------------------------------------------------------------------- #
    #                                Evaluation name                               #
    # ---------------------------------------------------------------------------- #
    evaluation_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    if config.get('experiment_name'):
        evaluation_name = f'{config['experiment_name']}_{evaluation_date}'
    else:
        alg_name = config['algorithm']['name'].split(':')[-1]
        evaluation_name = f'{alg_name}_{evaluation_date}_evaluation'

    # ---------------------------------------------------------------------------- #
    #                                Get model path                                #
    # ---------------------------------------------------------------------------- #
    model_path = None

    # ---------------------------- Local model path ----------------------------- #
    if config['model'].get('local_path'):
        model_path = config['model']['local_path']

    # ------------------------ Weights and Bias model path ----------------------- #
    if config['model'].get('entity'):

        api = wandb.Api()
        # Get model path
        artifact_tag = config['model'].get(
            'artifact_tag', 'latest')
        wandb_path = f'{
            config['model']['entity']}/{
            config['model']['project']}/{
            config['model']['artifact_name']}:{artifact_tag}'

        # Download artifact
        artifact = api.artifact(wandb_path)
        artifact.download(
            path_prefix=config['model']['artifact_path'],
            root='./')

        # Set model path to local wandb downloaded file
        model_path = f'./{config['model']['model_path']}'

    # -------------------------- Google cloud model path ------------------------- #
    if config['model'].get('bucket_path'):
        # Download from given bucket (gcloud configured with privileges)
        client = gcloud.init_storage_client()
        bucket_name = config['model']['bucket_path'].split('/')[2]
        model_path = f'{
            config['model']['bucket_path'].split(
                bucket_name + '/')[
                -1]}'
        gcloud.read_from_bucket(client, bucket_name, model_path)
        model_path = f'./{model_path}'

    logger.info(f'Model path set to {model_path}')

    # ---------------------------------------------------------------------------- #
    #                           Environment parameters                             #
    # ---------------------------------------------------------------------------- #
    env_params = {}

    # ------- Update env params configuration with env yaml file if exists ------- #
    if config.get('env_yaml_config'):
        logger.info(
            f'Reading environment parameters from {
                config['env_yaml_config']}')
        with open(config['env_yaml_config'], 'r') as env_yaml_conf:
            env_params.update(yaml.load(env_yaml_conf, Loader=yaml.FullLoader))

    # -- Update env params configuration with specified env parameters if exists -- #
    if config.get('env_params'):
        logger.info(
            f'Reading environment parameters from env_params config')
        if env_params:
            logger.info(
                f'Overwriting (deep_update) environment parameters from env_yaml_config with env_params config')
        env_params = deep_update(
            env_params, process_environment_parameters(
                config['env_params']))

    # ------------ For this script, the execution name will be updated ----------- #
    env_params.update({'env_name': evaluation_name})

    # ---------------------------------------------------------------------------- #
    #                            Wrappers definition                               #
    # ---------------------------------------------------------------------------- #
    wrappers = {}

    # ------------------ Read wrappers from yaml file if exists ------------------ #
    if config.get('wrappers_yaml_config'):
        logger.info(
            f'Reading wrappers from {
                config['wrappers_yaml_config']}')
        with open(config['wrappers_yaml_config'], 'r') as f:
            wrappers = yaml.load(f, Loader=yaml.FullLoader)

    # ----------------------- Delete WandBLogger by default ---------------------- #
    if wrappers:
        key_to_remove = [
            key for key in wrappers if 'WandBLogger' in key][0]
        del wrappers[key_to_remove]
        logger.info(f'WandBLogger removed from wrappers if exists')

    # ------ Read wrappers from yaml file and overwrite yaml file if exists ------ #
    if config.get('wrappers'):
        logger.info(f'Reading wrappers from wrappers config')
        # Update wrappers with the ones defined in the yaml file
        for wrapper in config['wrappers']:
            for wrapper_name, wrapper_arguments in wrapper.items():
                for name, value in wrapper_arguments.items():
                    # parse str parameters to sinergym Callable or Objects if
                    # required
                    if isinstance(value, str):
                        if ':' in value:
                            wrapper_arguments[name] = import_from_path(value)
            wrappers = deep_update(wrappers, {wrapper_name: wrapper_arguments})
        logger.info(f'Wrappers updated with wrappers config')

    # ---------------------------------------------------------------------------- #
    #                Create environment with parameters and wrappers               #
    # ---------------------------------------------------------------------------- #
    env = create_environment(
        env_id=config['environment'],
        env_params=env_params,
        wrappers=wrappers)
    logger.info(
        f'Environment created with ultimate environment parameters and wrappers')

    # ---------------------------------------------------------------------------- #
    #                           Defining model (algorithm)                         #
    # ---------------------------------------------------------------------------- #
    alg_name = config['algorithm']['name']
    alg_cls = import_from_path(alg_name)
    model = alg_cls.load(model_path)
    logger.info(f'Model loaded from {model_path}')

    # ---------------------------------------------------------------------------- #
    #                             Execute loaded agent                             #
    # ---------------------------------------------------------------------------- #
    logger.info(f'Starting evaluation with {config['episodes']} episodes')
    for i in range(config['episodes']):
        # Reset the environment to start a new episode
        obs, info = env.reset()
        truncated = terminated = False
        while not (terminated or truncated):
            # Use the agent to predict the next action
            a, _ = model.predict(obs, deterministic=True)
            # Read observation and reward
            obs, reward, terminated, truncated, info = env.step(a)

    env.close()

    if config.get('cloud'):
        # ---------------------------------------------------------------------------- #
        #                                 Store results                                #
        # ---------------------------------------------------------------------------- #
        if config['cloud'].get('remote_store'):
            # Initiate Google Cloud client
            client = gcloud.init_storage_client()
            # Send output to common Google Cloud resource
            gcloud.upload_to_bucket(
                client,
                src_path=env.get_wrapper_attr('workspace_path'),
                dest_bucket_name=config['cloud']['remote_store'],
                dest_path=evaluation_name)

        # ---------------------------------------------------------------------------- #
        #                          Auto-delete remote container                        #
        # ---------------------------------------------------------------------------- #
        if config['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                config['cloud']['auto_delete']['group_name'], token)

except (Exception, KeyboardInterrupt) as err:
    print("Error or interruption in process detected")
    print(traceback.print_exc(), file=sys.stderr)

    env.close()

    # Auto delete
    if config.get('cloud'):
        if config['cloud'].get('auto_delete'):
            print('Deleting remote container')
            token = gcloud.get_service_account_token()
            gcloud.delete_instance_MIG_from_container(
                config['cloud']['auto_delete']['group_name'], token)
    raise err
