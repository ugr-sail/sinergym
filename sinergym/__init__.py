import json
import os

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec, register

from sinergym.utils.common import convert_conf_to_env_parameters
from sinergym.utils.constants import *
from sinergym.utils.rewards import *

# ------------------------- Set __version__ in module ------------------------ #
version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_file, 'r') as file_handler:
    __version__ = file_handler.read().strip()

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
            dtype=np.float32),
        'time_variables': ['month', 'day_of_month', 'hour'],
        'variables': {
            'outdoor_temperature': (
                'Site Outdoor Air DryBulb Temperature',
                'Environment'),
            'htg_setpoint': (
                'Zone Thermostat Heating Setpoint Temperature',
                'SPACE5-1'),
            'clg_setpoint': (
                'Zone Thermostat Cooling Setpoint Temperature',
                'SPACE5-1'),
            'air_temperature': (
                'Zone Air Temperature',
                'SPACE5-1'),
            'air_humidity': (
                'Zone Air Relative Humidity',
                'SPACE5-1'),
            'HVAC_electricity_demand_rate': (
                'Facility Total HVAC Electricity Demand Rate',
                'Whole Building')
        },
        'meters': {},
        'actuators': {
            'Heating_Setpoint_RL': (
                'Schedule:Compact',
                'Schedule Value',
                'HTG-SETP-SCH'),
            'Cooling_Setpoint_RL': (
                'Schedule:Compact',
                'Schedule Value',
                'CLG-SETP-SCH')
        },
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variables': ['air_temperature'],
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)},
        'env_name': 'demo-v1',
        'config_params': {
            'runperiod': (1, 1, 1991, 1, 3, 1991),
            'timesteps_per_hour': 1
        }})

# ------------------- Read environment configuration files ------------------- #
conf_files = []
configuration_path = os.path.join(
    os.path.dirname(__file__),
    'data/default_configuration')
for root, dirs, files in os.walk(configuration_path):
    for file in files:
        # Obtain the whole path for each configuration file
        file_path = os.path.join(root, file)
        conf_files.append(file_path)

# ---------------- For each conf file, setting up environments --------------- #
for conf_file in conf_files:
    with open(conf_file) as json_f:
        conf = json.load(json_f)

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
                kwargs=env_kwargs
            )

        # If discrete space is included, add the same environment with
        # discretization
        if conf.get('action_space_discrete'):
            # Copy the dictionary since is used by reference
            env_kwargs_discrete = env_kwargs.copy()

            # Action mapping must be included in constants.
            action_mapping = eval(
                "DEFAULT_" +
                conf["id_base"].upper() +
                "_DISCRETE_FUNCTION")

            discrete_wrapper_spec = WrapperSpec(
                name='DiscretizeEnv',
                entry_point='sinergym.utils.wrappers:DiscretizeEnv',
                kwargs={
                    'discrete_space': eval(conf['action_space_discrete']),
                    'action_mapping': action_mapping})
            additional_wrappers = (discrete_wrapper_spec,)

            env_kwargs_discrete['env_name'] = env_kwargs_discrete['env_name'].replace(
                'continuous', 'discrete')

            register(
                id=env_id.replace('continuous', 'discrete'),
                entry_point='sinergym.envs:EplusEnv',
                additional_wrappers=additional_wrappers,
                # order_enforce=False,
                # disable_env_checker=True,
                kwargs=env_kwargs_discrete
            )
