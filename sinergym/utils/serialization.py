"""Serialization utilities for Sinergym in YAML format.
This module provides functions to serialize and deserialize Sinergym
environments and their components using the YAML format.
"""

import importlib
import types

import gymnasium as gym
import numpy as np
import yaml

from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.common import import_from_path

# ---------------------------------------------------------------------------- #
#                           Python Class and Function                          #
# ---------------------------------------------------------------------------- #


def class_representer(dumper, obj):
    class_path = f'{obj.__module__}.{obj.__name__}'
    return dumper.represent_scalar('!Class', class_path)


def function_representer(dumper, obj):
    func_path = f'{obj.__module__}.{obj.__name__}'
    return dumper.represent_scalar('!Function', func_path)


def class_constructor(loader, node):
    class_path = loader.construct_scalar(node)
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def function_constructor(loader, node):
    func_path = loader.construct_scalar(node)
    module_path, func_name = func_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


# ---------------------------------------------------------------------------- #
#                                 Python Tuples                                #
# ---------------------------------------------------------------------------- #


def tuple_representer(dumper, obj):
    return dumper.represent_sequence('!Tuple', obj)


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


# ---------------------------------------------------------------------------- #
#                               Numpy Arrays                                   #
# ---------------------------------------------------------------------------- #


def array_representer(dumper, obj):
    mapping = {'object': obj.tolist(), 'dtype': f'numpy:{str(obj.dtype)}'}
    return dumper.represent_mapping('!NumpyArray', mapping)


def array_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    values['dtype'] = import_from_path(values['dtype'])
    return np.array(**values)


# ---------------------------------------------------------------------------- #
#                               Gymnasium Spaces                               #
# ---------------------------------------------------------------------------- #


def space_representer(dumper, obj):
    class_name = obj.__class__.__name__
    arguments = {}
    if class_name == 'Box':
        arguments = {
            'low': obj.low,
            'high': obj.high,
            'shape': obj.shape,
            'dtype': f'numpy:{str(obj.dtype)}',
            #   'seed': obj.seed
        }
    elif class_name == 'Discrete':
        arguments = {
            'n': int(obj.n),
            #   'seed': obj.seed,
            'start': int(obj.start),
        }
    elif class_name == 'MultiDiscrete':
        arguments = {
            'nvec': obj.nvec,
            'dtype': f'numpy:{str(obj.dtype)}',
            #   'seed': obj.seed,
            'start': obj.start,
        }
    elif class_name == 'MultiBinary':
        arguments = {
            'n': int(obj.n),
            #   'seed': obj.seed
        }

    mapping = {'class': obj.__class__, 'arguments': arguments}
    return dumper.represent_mapping('!GymSpace', mapping)


def space_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    cls = values['class']
    args = values['arguments']
    class_name = cls.__name__

    if class_name == 'Box':
        args['dtype'] = import_from_path(args['dtype'])
        return cls(
            low=args['low'], high=args['high'], shape=args['shape'], dtype=args['dtype']
        )
    elif class_name == 'Discrete':
        return cls(n=args['n'], start=args.get('start', 0))
    elif class_name == 'MultiDiscrete':
        args['dtype'] = import_from_path(args['dtype'])
        return cls(nvec=args['nvec'], dtype=args['dtype'], start=args.get('start', 0))
    elif class_name == 'MultiBinary':
        return cls(n=args['n'])
    else:
        # Fallback for unknown spaces
        return cls(**args)


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
        'initial_context': env.default_options.get('initial_context'),
        'weather_variability': env.default_options.get('weather_variability'),
        'reward': env.reward_fn.__class__,
        'reward_kwargs': env.reward_kwargs,
        'max_ep_store': env.max_ep_store,
        'env_name': env.name,
        'building_config': env.building_config,
        'seed': env.seed,
    }

    return dumper.represent_mapping('!EplusEnv', mapping)


def env_constructor(loader, node):
    values = loader.construct_mapping(node, deep=True)
    # return EplusEnv.from_dict(values)
    return values


# ---------------------------------------------------------------------------- #
#                 Registration of representer and constructors                 #
# ---------------------------------------------------------------------------- #


def create_sinergym_yaml_serializers():
    """
    Register custom YAML representer and constructors for Sinergym
    environments, gym spaces, numpy arrays and more.
    """

    # ----------------------------- Python classes ------------------------------ #
    yaml.add_multi_representer(type, class_representer)
    yaml.add_constructor('!Class', class_constructor, Loader=yaml.FullLoader)

    # ----------------------------- Python functions ---------------------------- #
    yaml.add_multi_representer(types.FunctionType, function_representer)
    yaml.add_constructor('!Function', function_constructor, Loader=yaml.FullLoader)

    # ------------------------------- Python tuples ----------------------------- #
    yaml.add_representer(tuple, tuple_representer)
    yaml.add_constructor('!Tuple', tuple_constructor, Loader=yaml.FullLoader)

    # ------------------------------- Numpy arrays ------------------------------- #
    yaml.add_representer(np.ndarray, array_representer)
    yaml.add_constructor('!NumpyArray', array_constructor, Loader=yaml.FullLoader)

    # ----------------------------- Gymnasium spaces ----------------------------- #
    gym_space_classes = [
        cls
        for cls in gym.spaces.__dict__.values()
        if isinstance(cls, type) and issubclass(cls, gym.Space)
    ]
    for gym_space_class in gym_space_classes:
        yaml.add_representer(gym_space_class, space_representer)
    yaml.add_constructor('!GymSpace', space_constructor, Loader=yaml.FullLoader)

    # --------------------------- Sinergym environments -------------------------- #
    yaml.add_representer(EplusEnv, env_representer)
    yaml.add_constructor('!EplusEnv', env_constructor, Loader=yaml.FullLoader)
