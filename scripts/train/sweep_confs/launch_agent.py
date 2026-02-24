import argparse
import importlib.util
import multiprocessing
import os
import sys
import types

import yaml

from sinergym.utils import gcloud


def _load_train_module(train_script_path: str) -> types.ModuleType:
    """Dynamically import the training script as a module."""
    spec = importlib.util.spec_from_file_location('train', train_script_path)
    if spec and spec.loader:
        train = types.ModuleType(spec.name)
        sys.modules[spec.name] = train
        spec.loader.exec_module(train)
        return train
    raise ImportError(f"The script could not be imported from {train_script_path}")


def _run_wandb_agent(
    *,
    sweep_id: str,
    entity: str,
    project: str,
    count: int,
    train_script_path: str,
) -> None:
    """Run a wandb agent inside an isolated process.

    Important: `wandb` is imported inside the child process to avoid inheriting
    any W&B service/client state from the parent (a common source of issues when
    using `fork`).
    """
    import wandb  # local import on purpose

    train = _load_train_module(train_script_path)
    wandb.agent(
        sweep_id=sweep_id,
        entity=entity,
        project=project,
        count=count,
        function=train.train,
    )


if __name__ == '__main__':

    # Use spawn to avoid subtle W&B issues with forking.
    # (Also required because we run a top-level function as Process target.)
    multiprocessing.set_start_method('spawn', force=True)

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
        help='Path to launch agents configuration (YAML file)',
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------- #
    #                             Read yaml parameters                             #
    # ---------------------------------------------------------------------------- #

    with open(args.configuration, 'r') as yaml_conf:
        conf = yaml.safe_load(yaml_conf)

    # ---------------------------------------------------------------------------- #
    #                             Processing parameters                            #
    # ---------------------------------------------------------------------------- #
    sweep_id = conf['sweep_id']
    entity = conf['entity']
    project = conf['project']
    parallel_agents = conf['parallel_agents']
    sequential_experiments = conf['sequential_experiments']

    # --------------------------------- Optionals -------------------------------- #
    if conf.get('wandb_api_key'):
        os.environ.update({'WANDB_API_KEY': conf['wandb_api_key']})
    if conf.get('wandb_group'):
        os.environ.update({'WANDB_RUN_GROUP': conf['wandb_group']})
    if conf.get('wandb_tags'):
        os.environ.update({'WANDB_TAGS': conf['wandb_tags']})

    # ---------------------------------------------------------------------------- #
    #                   Import train methodology from script path                  #
    # ---------------------------------------------------------------------------- #
    train_script_path = conf['train_script_path']

    # ---------------------------------------------------------------------------- #
    #                                Launch agent(s)                               #
    # ---------------------------------------------------------------------------- #
    list_process = []

    print("Number of parallel processes: ", parallel_agents)
    print("Number of sequential experiments by agent: ", sequential_experiments)
    print("Total executions: ", parallel_agents * sequential_experiments)

    while parallel_agents > 0:
        process = multiprocessing.Process(
            target=_run_wandb_agent,
            kwargs={
                "sweep_id": sweep_id,
                "entity": entity,
                "project": project,
                "count": sequential_experiments,
                "train_script_path": train_script_path,
            },
        )
        process.start()
        list_process.append(process)
        parallel_agents -= 1

    for wait_process in list_process:
        wait_process.join()

    # ---------------------------------------------------------------------------- #
    #                   Autodelete option if is a cloud resource                   #
    # ---------------------------------------------------------------------------- #
    if conf.get('autodelete'):
        token = gcloud.get_service_account_token()
        gcloud.delete_instance_MIG_from_container(conf['group_name'], token)
