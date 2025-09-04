import logging
import sys
import traceback
from datetime import datetime

import yaml
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.noise import NormalActionNoise

import sinergym
import sinergym.utils.gcloud as gcloud
import wandb
from sinergym.utils.callbacks import *
from sinergym.utils.common import (
    create_environment,
    deep_update,
    import_from_path,
    is_wrapped,
    process_algorithm_parameters,
    process_environment_parameters,
)
from sinergym.utils.constants import *
from sinergym.utils.logger import WandBOutputFormat
from wandb.util import generate_id


def train():

    # Optional: Terminal log in the same format as Sinergym.
    # Logger info can be replaced by print.
    terminal_logger = TerminalLogger()
    logger = terminal_logger.getLogger(name='TRAINING', level=logging.INFO)

    try:
        # ---------------------------------------------------------------------------- #
        #                                  WandB init                                  #
        # ---------------------------------------------------------------------------- #
        # Default configuration (it could be override by sweeps totally or
        # partially)
        default_config = {
            'sinergym-version': sinergym.__version__,
            'python-version': sys.version,
            'stable-baselines3-version': sb3_version,
        }
        run = wandb.init(config=default_config)

        # ---------------------------------------------------------------------------- #
        #                               Register run name                              #
        # ---------------------------------------------------------------------------- #
        experiment_date = datetime.today().strftime('%Y-%m-%d_%H-%M')
        if wandb.config.get('experiment_name'):
            experiment_name = f'{
                wandb.config['experiment_name']}_{experiment_date}_{
                generate_id()}'
        else:
            alg_name = wandb.config['algorithm'].split(':')[-1]
            experiment_name = f'{alg_name}_{experiment_date}_{
                generate_id()}'

        run.name = experiment_name

        # ---------------------------------------------------------------------------- #
        #                Load model as starting point if defined                       #
        # ---------------------------------------------------------------------------- #
        model_path = None
        if wandb.config.get('model'):
            logger.info(f'Training from pre-trained model defined in config')
            # ---------------------------- Local model path ----------------------------- #
            if wandb.config['model'].get('local_path'):
                model_path = wandb.config['model']['local_path']

            # ------------------------ Weights and Bias model path ----------------------- #
            if wandb.config['model'].get('entity'):

                api = wandb.Api()
                # Get model path
                artifact_tag = wandb.config['model'].get('artifact_tag', 'latest')
                wandb_path = f'{
                    wandb.config['model']['entity']}/{
                    wandb.config['model']['project']}/{
                    wandb.config['model']['artifact_name']}:{artifact_tag}'

                # Download artifact
                artifact = api.artifact(wandb_path)
                artifact.download(
                    path_prefix=wandb.config['model']['artifact_path'], root='./'
                )

                # Set model path to local wandb downloaded file
                model_path = f'./{wandb.config['model']['model_path']}'

            # -------------------------- Google cloud model path ------------------------- #
            if wandb.config['model'].get('bucket_path'):
                # Download from given bucket (gcloud configured with
                # privileges)
                client = gcloud.init_storage_client()
                bucket_name = wandb.config['model']['bucket_path'].split('/')[2]
                model_path = f'{
                    wandb.config['model']['bucket_path'].split(
                        bucket_name + '/')[
                        -1]}'
                gcloud.read_from_bucket(client, bucket_name, model_path)
                model_path = f'./{model_path}'

            logger.info(f'Model path: {model_path}')

        # ---------------------------------------------------------------------------- #
        #                           Environment parameters                              #
        # ---------------------------------------------------------------------------- #
        env_params = {}

        # ------- Update env params configuration with env yaml file if exists ------- #
        if wandb.config.get('env_yaml_config'):
            logger.info(
                f'Reading environment parameters from {
                    wandb.config['env_yaml_config']}'
            )
            with open(wandb.config['env_yaml_config'], 'r') as env_yaml_conf:
                env_params.update(yaml.load(env_yaml_conf, Loader=yaml.FullLoader))

        # -- Update env params configuration with specified env parameters if exists -- #
        if wandb.config.get('env_params'):
            logger.info(f'Reading environment parameters from env_params config')
            if env_params:
                logger.info(
                    f'Overwriting (deep_update) environment parameters from env_yaml_config with env_params config'
                )
            env_params = deep_update(
                env_params, process_environment_parameters(wandb.config['env_params'])
            )

        env_params['env_name'] = experiment_name

        # ---------------------------------------------------------------------------- #
        #                            Wrappers definition                               #
        # ---------------------------------------------------------------------------- #
        wrappers = {}

        # ------------------ Read wrappers from yaml file if exists ------------------ #
        if wandb.config.get('wrappers_yaml_config'):
            logger.info(
                f'Reading wrappers from {
                    wandb.config['wrappers_yaml_config']}'
            )
            with open(wandb.config['wrappers_yaml_config'], 'r') as f:
                wrappers = yaml.load(f, Loader=yaml.FullLoader)

        # ------ Read wrappers from yaml file and overwrite yaml file if exists ------ #
        if wandb.config.get('wrappers'):
            logger.info(f'Reading wrappers from wrappers config')
            # Update wrappers with the ones defined in the yaml file
            for wrapper in wandb.config['wrappers']:
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
            env_id=wandb.config['environment'],
            env_params=env_params,
            wrappers=wrappers,
            env_deep_update=wandb.config.get('env_deep_update', True),
        )
        logger.info(
            f'Environment created with ultimate environment parameters and wrappers'
        )

        # --------------- With sweeps, WandBLogger wrapper is required --------------- #
        assert is_wrapped(
            env, WandBLogger
        ), 'Environments with sweeps must be wrapped with WandBLogger.'

        # ---------------------------------------------------------------------------- #
        #         Update the environment parameters with the ones defined here         #
        # ---------------------------------------------------------------------------- #
        # `delete lock on sweep parameters
        run.config.__dict__["_locked"] = {}
        run.config.update(
            {'env_params': env.get_wrapper_attr('to_dict')()}, allow_val_change=True
        )
        logger.info(f'Environment parameters registered in wandb')

        # ---------------------------------------------------------------------------- #
        #                           Defining model (algorithm)                         #
        # ---------------------------------------------------------------------------- #
        alg_name = wandb.config['algorithm']
        alg_cls = import_from_path(alg_name)
        alg_params = wandb.config.get('algorithm_parameters', {'policy': 'MlpPolicy'})
        alg_params = process_algorithm_parameters(alg_params)

        # --------------------------- Training from scratch -------------------------- #
        if model_path is None:
            try:
                model = alg_cls(env=env, **alg_params)
                logger.info(f'Model created from scratch')
            except NameError:
                raise NameError(
                    'Algorithm {} does not exists. It must be a valid SB3 algorithm.'.format(
                        alg_name
                    )
                )

        # --------------------- Training from a pre-trained model --------------------- #
        else:
            model = None
            try:
                model = alg_cls.load(model_path)
                logger.info(f'Model loaded from {model_path}')
            except NameError:
                raise NameError(
                    'Algorithm {} does not exists. It must be a valid SB3 algorithm.'.format(
                        alg_name
                    )
                )

            model.set_env(env)
            logger.info(f'Model set to environment')
        # ---------------------------------------------------------------------------- #
        #                              SET UP WANDB LOGGER                             #
        # ---------------------------------------------------------------------------- #
        sb3_logger = SB3Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(sys.stdout, max_length=120),
                WandBOutputFormat(),
            ],
        )
        model.set_logger(sb3_logger)
        logger.info(f'WandB logger format set to model')

        # ---------------------------------------------------------------------------- #
        #                          Application of callback(s)                          #
        # ---------------------------------------------------------------------------- #
        callbacks = []

        # ---------------------------- EVALUATION CALLBACK --------------------------- #
        if wandb.config.get('evaluation', False):

            # ------------ Preparing the evaluation environment configuration ------------ #
            env_params['env_name'] = experiment_name + '_EVALUATION'
            logger.info(
                f'Evaluation enabled with environment name: {
                    env_params["env_name"]}'
            )

            # By default, the evaluation environment does not use WandBLogger
            if wrappers:
                key_to_remove = [key for key in wrappers if 'WandBLogger' in key][0]
                del wrappers[key_to_remove]
                logger.info(f'Wrappers updated without WandBLogger for evaluations')
            # ----------------------- Create evaluation environment ---------------------- #
            eval_env = create_environment(
                env_id=wandb.config['environment'],
                env_params=env_params,
                wrappers=wrappers,
                env_deep_update=wandb.config.get('env_deep_update', True),
            )
            logger.info(
                f'Evaluation environment created with the same parameters and wrappers (except WandBLogger)'
            )

            # ------------------------ Create evaluation callback ------------------------ #
            eval_callback = LoggerEvalCallback(
                eval_env=eval_env,
                train_env=env,
                n_eval_episodes=wandb.config['evaluation']['eval_length'],
                eval_freq_episodes=wandb.config['evaluation']['eval_freq'],
                deterministic=True,
                excluded_metrics=[
                    'episode_num',
                    'length (timesteps)',
                    'time_elapsed (hours)',
                ],
                verbose=1,
            )

            callbacks.append(eval_callback)

        callback = CallbackList(callbacks)

        # ---------------------------------------------------------------------------- #
        #                                 DRL training                                 #
        # ---------------------------------------------------------------------------- #
        timesteps = wandb.config['episodes'] * (
            env.get_wrapper_attr('timestep_per_episode')
        )

        logger.info(f'Starting training with {timesteps} total timesteps')
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            log_interval=wandb.config['log_interval'],
        )
        logger.info(f'Training completed')
        logger.info(
            f'Model saved to {
                env.get_wrapper_attr("workspace_path")}/model'
        )

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
                    dest_path=experiment_name,
                )

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
        if model:
            model.save(env.get_wrapper_attr('workspace_path') + '/model')

        env.close()
        raise err
