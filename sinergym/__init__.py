# Suppress known warning before any imports
try:
    import warnings

    warnings.filterwarnings("ignore", message=".*epw.data submodule is not installed.*")
    # Silence known Pydantic v2 schema warnings from third-party libs
    warnings.filterwarnings(
        "ignore",
        message=r"The 'repr' attribute with value .* was provided to the `Field\(\)` function",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The 'frozen' attribute with value .* was provided to the `Field\(\)` function",
    )
except ImportError:
    pass


import logging
import os
from typing import Union

import gymnasium as gym
import numpy as np
import yaml
from gymnasium.envs.registration import WrapperSpec, register

from sinergym.utils.common import convert_conf_to_env_parameters, import_from_path
from sinergym.utils.serialization import create_sinergym_yaml_serializers

# --------------------- Serialization of Sinergym in YAML -------------------- #
create_sinergym_yaml_serializers()

# ------------------------- Set __version__ in module ------------------------ #

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("sinergym")
except Exception:
    import os
    import tomllib

    project_root = os.path.dirname(os.path.dirname(__file__))
    pyproject_file = os.path.join(project_root, "pyproject.toml")
    with open(pyproject_file, "rb") as f:
        __version__ = tomllib.load(f)["tool"]["poetry"]["version"]

# ---------------------------- 0) Demo environment --------------------------- #
register(
    id='Eplus-demo-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'building_file': '5ZoneAutoDXVAV.epJSON',
        'weather_files': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'action_space': gym.spaces.Box(
            low=np.array([15.0, 22.5], dtype=np.float32),
            high=np.array([22.5, 30.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        ),
        'time_variables': ['month', 'day_of_month', 'hour'],
        'variables': {
            'outdoor_temperature': (
                'Site Outdoor Air DryBulb Temperature',
                'Environment',
            ),
            'htg_setpoint': (
                'Zone Thermostat Heating Setpoint Temperature',
                'SPACE5-1',
            ),
            'clg_setpoint': (
                'Zone Thermostat Cooling Setpoint Temperature',
                'SPACE5-1',
            ),
            'air_temperature': ('Zone Air Temperature', 'SPACE5-1'),
            'air_humidity': ('Zone Air Relative Humidity', 'SPACE5-1'),
            'HVAC_electricity_demand_rate': (
                'Facility Total HVAC Electricity Demand Rate',
                'Whole Building',
            ),
        },
        'meters': {},
        'actuators': {
            'Heating_Setpoint_RL': (
                'Schedule:Compact',
                'Schedule Value',
                'HTG-SETP-SCH',
            ),
            'Cooling_Setpoint_RL': (
                'Schedule:Compact',
                'Schedule Value',
                'CLG-SETP-SCH',
            ),
        },
        'reward': import_from_path('sinergym.utils.rewards:LinearReward'),
        'reward_kwargs': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
        },
        'env_name': 'demo-v1',
        'building_config': {
            'runperiod': (1, 1, 1991, 1, 3, 1991),
            'timesteps_per_hour': 1,
        },
    },
)

# ------------------- Read environment configuration files ------------------- #


def register_envs_from_yaml(yaml_path: str):
    """
    Register environments from a YAML configuration file.

    :param yaml_path: Path to the YAML configuration file.
    """
    with open(yaml_path, 'r') as yaml_conf:
        conf = yaml.safe_load(yaml_conf)

    # configurations = Dict [key=environment_id, value=env_kwargs dict]
    configurations = convert_conf_to_env_parameters(conf)

    for env_id, env_kwargs in configurations.items():

        if not conf.get('only_discrete', False):

            register(
                id=env_id,
                entry_point='sinergym.envs:EplusEnv',
                # additional_wrappers=additional_wrappers,
                # order_enforce=False,
                # disable_env_checker=True,
                kwargs=env_kwargs,
            )

        # If discrete space is included, add the same environment with
        # discretization
        if conf.get('action_space_discrete'):
            # Copy the dictionary since is used by reference
            env_kwargs_discrete = env_kwargs.copy()

            # Action mapping must be included in constants.
            discrete_function = f'DEFAULT_{
                conf["id_base"].upper()}_DISCRETE_FUNCTION'
            action_mapping = import_from_path(
                f'sinergym.utils.constants:{discrete_function}'
            )

            discrete_wrapper_spec = WrapperSpec(
                name='DiscretizeEnv',
                entry_point='sinergym.utils.wrappers:DiscretizeEnv',
                kwargs={
                    'discrete_space': eval(conf['action_space_discrete']),
                    'action_mapping': action_mapping,
                },
            )
            additional_wrappers = (discrete_wrapper_spec,)

            env_kwargs_discrete['env_name'] = env_kwargs_discrete['env_name'].replace(
                'continuous', 'discrete'
            )

            register(
                id=env_id.replace('continuous', 'discrete'),
                entry_point='sinergym.envs:EplusEnv',
                additional_wrappers=additional_wrappers,
                # order_enforce=False,
                # disable_env_checker=True,
                kwargs=env_kwargs_discrete,
            )


# ------------------ Read default configuration files ------------------ #
configuration_path = os.path.join(
    os.path.dirname(__file__), 'data/default_configuration'
)
for root, dirs, files in os.walk(configuration_path):
    for file in files:
        # Obtain the whole path for each configuration file
        file_path = os.path.join(root, file)
        # For each conf file, set up environments
        register_envs_from_yaml(file_path)

# ---------------- Available Sinergym environment's ids getter --------------- #


def ids():
    return [
        env_id
        for env_id in gym.envs.registration.registry.keys()  # type: ignore
        if env_id.startswith('Eplus')
    ]


# ----------------------------- Log level system ----------------------------- #
def set_logger_level(name: str, level: Union[str, int]):
    if name.upper() == 'WRAPPER' or name.upper() == 'WRAPPERS':
        # Get all logger whose name contains 'WRAPPER'
        for logger_name in logging.root.manager.loggerDict.keys():
            if 'WRAPPER' in logger_name.upper():
                logger = logging.getLogger(logger_name)
                logger.setLevel(level)
    else:
        logger = logging.getLogger(name.upper())
        logger.setLevel(level)
        if name == 'ENVIRONMENT':
            logger = logging.getLogger('Printer')
            logger.setLevel(level)
