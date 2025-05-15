"""Serialization utilities for Sinergym in YAML format.
This module provides functions to serialize and deserialize Sinergym
environments and their components using the YAML format.
"""
import yaml
import numpy as np
import gymnasium as gym
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.rewards import *

# ---------------------------------------------------------------------------- #
#                               Numpy Arrays                                   #
# ---------------------------------------------------------------------------- #


def array_representer(dumper, obj):
    mapping = {
        'object': obj.tolist(),
        'dtype': f'np.{str(obj.dtype)}'
    }
    return dumper.represent_mapping('!NumpyArray', mapping)


def array_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    values['dtype'] = eval(values['dtype'])
    return np.array(**values)

# ---------------------------------------------------------------------------- #
#                               Gymnasium Spaces                               #
# ---------------------------------------------------------------------------- #


def space_representer(dumper, obj):
    mapping = {
        'class': f'gym.spaces.{obj.__class__.__name__}',
        'arguments': {
            'low': obj.low,
            'high': obj.high,
            'shape': obj.shape,
            'dtype': f'np.{str(obj.dtype)}'
        }
    }
    return dumper.represent_mapping('!GymSpace', mapping)


def space_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    space_class = eval(values['class'])
    values['arguments']['dtype'] = eval(values['arguments']['dtype'])
    return space_class(**values['arguments'])

# ---------------------------------------------------------------------------- #
#                             Sinergym Environment                             #
# ---------------------------------------------------------------------------- #


def env_representer(dumper, obj):
    env = obj.unwrapped
    mapping = {
        'building_file': env.building_file,
        'weather_files': env.weather_files,
        'action_space': env.action_space,
        'time_variables': env.time_variables,
        'variables': env.variables,
        'meters': env.meters,
        'actuators': env.actuators,
        'context': env.context,
        'initial_context': env.default_options.get(
            'initial_context'),
        'weather_variability': env.default_options.get(
            'weather_variability'),
        'reward': str(env.reward_fn.__class__.__name__),
        'reward_kwargs': env.reward_kwargs,
        'max_ep_store': env.max_ep_store,
        'env_name': env.name,
        'building_config': env.building_config,
        'seed': env.seed
    }

    return dumper.represent_mapping('!EplusEnv', mapping)


def env_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    values['reward'] = eval(values['reward'])
    return EplusEnv.from_dict(values)

# ---------------------------------------------------------------------------- #
#                 Registration of representers and constructors                #
# ---------------------------------------------------------------------------- #


def create_sinergym_yaml_serializers():
    """
    Register custom YAML representers and constructors for Sinergym
    environments, gym spaces, numpy arrays and more.
    """

    # ------------------------------- Numpy arrays ------------------------------- #
    yaml.add_representer(np.ndarray, array_representer)
    yaml.add_constructor(
        '!NumpyArray',
        array_constructor,
        Loader=yaml.FullLoader)

    # ----------------------------- Gymnasium spaces ----------------------------- #
    gym_space_classes = [
        cls for cls in gym.spaces.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, gym.Space)
    ]
    for gym_space_class in gym_space_classes:
        yaml.add_representer(gym_space_class, space_representer)
    yaml.add_constructor(
        '!GymSpace',
        space_constructor,
        Loader=yaml.FullLoader)

    # --------------------------- Sinergym environments -------------------------- #
    yaml.add_representer(EplusEnv, env_representer)
    yaml.add_constructor('!EplusEnv', env_constructor, Loader=yaml.FullLoader)
