import argparse
import importlib
import importlib.util
import multiprocessing
import os
import sys
import types

import wandb
import yaml

if __name__ == '__main__':

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
        help='Path to launch agents configuration (YAML file)'
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

    spec = importlib.util.spec_from_file_location('train', train_script_path)
    if spec and spec.loader:
        # Crear un módulo vacío con el nombre correcto
        train = types.ModuleType(spec.name)
        sys.modules[spec.name] = train  # Registrar el módulo en sys.modules
        spec.loader.exec_module(train)  # Ejecutar el script dentro del módulo
    else:
        raise ImportError(
            f"The script could not be imported from {train_script_path}")

    # ---------------------------------------------------------------------------- #
    #                                Launch agent(s)                               #
    # ---------------------------------------------------------------------------- #
    list_process = []

    print("Number of parallel processes: ", parallel_agents)
    print("Number of sequential experiments by agent: ", sequential_experiments)
    print("Total executions: ", parallel_agents * sequential_experiments)

    while parallel_agents > 0:
        process = multiprocessing.Process(
            target=lambda: wandb.agent(
                sweep_id=sweep_id,
                entity=entity,
                project=project,
                count=sequential_experiments,
                function=train.train
            )
        )
        process.start()
        list_process.append(process)
        parallel_agents -= 1

    for wait_process in list_process:
        wait_process.join()
