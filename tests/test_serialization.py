import types

import gymnasium as gym
import numpy as np
import yaml

from sinergym.config.modeling import ModelJSON
from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.common import get_delta_seconds
from sinergym.utils.serialization import create_sinergym_yaml_serializers


def test_class_and_function_representer_and_constructor():
    create_sinergym_yaml_serializers()
    import math

    # Class
    class_yaml = yaml.dump(ModelJSON)
    loaded_class = yaml.load(class_yaml, Loader=yaml.FullLoader)
    assert isinstance(loaded_class, type)
    assert loaded_class == ModelJSON

    # Function
    func_yaml = yaml.dump(get_delta_seconds)
    loaded_func = yaml.load(func_yaml, Loader=yaml.FullLoader)
    assert isinstance(loaded_func, types.FunctionType)
    assert isinstance(loaded_func, get_delta_seconds.__class__)


def test_tuple_representer_and_constructor():
    create_sinergym_yaml_serializers()
    data = (1, 2, 3)
    dumped = yaml.dump(data)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert loaded == data


def test_array_representer_and_constructor():
    create_sinergym_yaml_serializers()
    arr = np.array([1, 2, 3], dtype=np.float32)
    dumped = yaml.dump(arr)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert np.allclose(loaded, arr)
    assert loaded.dtype == arr.dtype


def test_gym_space_representer_and_constructor():
    create_sinergym_yaml_serializers()

    # Box
    box = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
    dumped = yaml.dump(box)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert isinstance(loaded, gym.spaces.Box)
    assert np.allclose(loaded.low, box.low)
    assert np.allclose(loaded.high, box.high)
    assert loaded.shape == box.shape

    # Discrete
    discrete = gym.spaces.Discrete(5, start=2)
    dumped = yaml.dump(discrete)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert isinstance(loaded, gym.spaces.Discrete)
    assert loaded.n == discrete.n
    assert loaded.start == discrete.start

    # MultiDiscrete
    multidiscrete = gym.spaces.MultiDiscrete([5, 2, 2], dtype=np.int64)
    dumped = yaml.dump(multidiscrete)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert isinstance(loaded, gym.spaces.MultiDiscrete)
    assert np.allclose(loaded.nvec, multidiscrete.nvec)
    assert loaded.dtype == multidiscrete.dtype

    # MultiBinary
    multibinary = gym.spaces.MultiBinary(4)
    dumped = yaml.dump(multibinary)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert isinstance(loaded, gym.spaces.MultiBinary)
    assert loaded.n == multibinary.n


def test_env_representer_and_constructor(env_5zone):
    create_sinergym_yaml_serializers()
    # Serializar el entorno
    dumped = yaml.dump(env_5zone)
    # Deserializar (devuelve un dict, no un EplusEnv real, por diseño actual)
    loaded = yaml.load(dumped, Loader=yaml.FullLoader)
    assert isinstance(loaded, dict)
    assert 'building_file' in loaded
    assert 'weather_files' in loaded
