import os

import gym
from gym.envs.registration import register

from sinergym.utils.constants import *
from sinergym.utils.rewards import *

# Set __version__ in module
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()

# ---------------------------------------------------------------------------- #
#                          5ZoneAutoDXVAV Environments                         #
# ---------------------------------------------------------------------------- #
# 0) Demo environment
register(
    id='Eplus-demo-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': 'demo-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 1) 5-zone, hot weather, discrete actions
register(
    id='Eplus-5Zone-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 2) 5-zone, mixed weather, discrete actions
register(
    id='Eplus-5Zone-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-discrete-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 3) 5-zone, cool weather, discrete actions
register(
    id='Eplus-5Zone-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-discrete-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 4) 5-zone, hot weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-hot-discrete-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 5) 5-zone, mixed weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-discrete-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 6) 5-zone, cool weather, discrete actions and stochastic
register(
    id='Eplus-5Zone-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-discrete-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 7) 5-zone, hot weather, continuous actions
register(
    id='Eplus-5Zone-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-hot-continuous-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 8) 5-zone, mixed weather, continuous actions
register(
    id='Eplus-5Zone-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-mixed-continuous-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 9) 5-zone, cool weather, continuous actions
register(
    id='Eplus-5Zone-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-cool-continuous-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 10) 5-zone, hot weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (
            1.0,
            0.0,
            0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': '5Zone-hot-continuous-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 11) 5-zone, mixed weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
                'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
                'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
                'range_comfort_winter': (20.0, 23.5),
                'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-mixed-continuous-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# 12) 5-zone, cool weather, continuous actions and stochastic
register(
    id='Eplus-5Zone-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_5ZONE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_5ZONE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_5ZONE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_5ZONE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
                'temperature_variable': 'Zone Air Temperature(SPACE1-1)',
                'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
                'range_comfort_winter': (20.0, 23.5),
                'range_comfort_summer': (23.0, 26.0)
        },
        'env_name': '5Zone-cool-continuous-stochastic-v1',
        'action_definition': DEFAULT_5ZONE_ACTION_DEFINITION})

# ---------------------------------------------------------------------------- #
#                            Datacenter Environments                           #
# ---------------------------------------------------------------------------- #
# 13) DC, hot weather, discrete actions
register(
    id='Eplus-datacenter-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-discrete-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# 14) DC, hot weather, continuous actions
register(
    id='Eplus-datacenter-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-continuous-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# 15) DC, hot weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-discrete-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# 16) DC, hot weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-hot-continuous-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION
    }
)

# 17) DC, mixed weather, discrete actions
register(
    id='Eplus-datacenter-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-discrete-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 18) DC, mixed weather, continuous actions
register(
    id='Eplus-datacenter-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-continuous-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 19) DC, mixed weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-discrete-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 20) DC, mixed weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-mixed-continuous-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 21) DC, cool weather, discrete actions
register(
    id='Eplus-datacenter-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)},
        'env_name': 'datacenter-cool-discrete-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 22) DC, cool weather, continuous actions
register(
    id='Eplus-datacenter-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-continuous-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 23) DC, cool weather, discrete actions and stochastic
register(
    id='Eplus-datacenter-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-discrete-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# 24) DC, cool weather, continuous actions and stochastic
register(
    id='Eplus-datacenter-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': '2ZoneDataCenterHVAC_wEconomizer.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_DATACENTER_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_DATACENTER_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_DATACENTER_ACTION_VARIABLES,
        'action_mapping': DEFAULT_DATACENTER_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(West Zone)',
                'Zone Air Temperature(East Zone)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'datacenter-cool-continuous-stochastic-v1',
        'action_definition': DEFAULT_DATACENTER_ACTION_DEFINITION})

# ---------------------------------------------------------------------------- #
#                          Warehouse Environments                              #
# ---------------------------------------------------------------------------- #

# 25) WH, hot weather, discrete actions
register(
    id='Eplus-warehouse-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-hot-discrete-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 26) WH, hot weather, continuous actions
register(
    id='Eplus-warehouse-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-hot-continuous-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 27) WH, hot weather, discrete actions and stochastic
register(
    id='Eplus-warehouse-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-hot-discrete-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 28) WH, hot weather, continuous actions and stochastic
register(
    id='Eplus-warehouse-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-hot-continuous-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 29) WH, mixed weather, discrete actions
register(
    id='Eplus-warehouse-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-mixed-discrete-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 30) WH, mixed weather, continuous actions
register(
    id='Eplus-warehouse-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-mixed-continuous-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 31) WH, mixed weather, discrete actions and stochastic
register(
    id='Eplus-warehouse-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-mixed-discrete-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 32) WH, mixed weather, continuous actions and stochastic
register(
    id='Eplus-warehouse-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-mixed-continuous-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 33) WH, cool weather, discrete actions
register(
    id='Eplus-warehouse-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-cool-discrete-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 34) WH, cool weather, continuous actions
register(
    id='Eplus-warehouse-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-cool-continuous-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 35) WH, cool weather, discrete actions and stochastic
register(
    id='Eplus-warehouse-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-cool-discrete-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})

# 36) WH, cool weather, continuous actions and stochastic
register(
    id='Eplus-warehouse-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_Warehouse_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_WAREHOUSE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_WAREHOUSE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_WAREHOUSE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Zone1 Office)',
                'Zone Air Temperature(Zone2 Fine Storage)',
                'Zone Air Temperature(Zone3 Bulk Storage)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'warehouse-cool-continuous-stochastic-v1',
        'action_definition': DEFAULT_WAREHOUSE_ACTION_DEFINITION})


# ---------------------------------------------------------------------------- #
#                              Medium Office                                   #
# ---------------------------------------------------------------------------- #

# 37) MO, hot weather, discrete actions
register(
    id='Eplus-office-hot-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-hot-discrete-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 38) MO, hot weather, continuous actions
register(
    id='Eplus-office-hot-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-hot-continuous-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 39) MO, hot weather, discrete actions and stochastic
register(
    id='Eplus-office-hot-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-hot-discrete-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 40) MO, hot weather, continuous actions and stochastic
register(
    id='Eplus-office-hot-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-hot-continuous-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 41) MO, mixed weather, discrete actions
register(
    id='Eplus-office-mixed-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-mixed-discrete-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 42) MO, mixed weather, continuous actions
register(
    id='Eplus-office-mixed-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-mixed-continuous-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 43) MO, mixed weather, discrete actions and stochastic
register(
    id='Eplus-office-mixed-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-mixed-discrete-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 44) MO, mixed weather, continuous actions and stochastic
register(
    id='Eplus-office-mixed-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-mixed-continuous-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 45) MO, cool weather, discrete actions
register(
    id='Eplus-office-cool-discrete-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-cool-discrete-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 46) MO, cool weather, continuous actions
register(
    id='Eplus-office-cool-continuous-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-cool-continuous-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 47) MO, cool weather, discrete actions and stochastic
register(
    id='Eplus-office-cool-discrete-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_DISCRETE,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-cool-discrete-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})

# 48) MO, cool weather, continuous actions and stochastic
register(
    id='Eplus-office-cool-continuous-stochastic-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'idf_file': 'ASHRAE9012016_OfficeMedium_Denver.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw',
        'observation_space': DEFAULT_OFFICE_OBSERVATION_SPACE,
        'observation_variables': DEFAULT_OFFICE_OBSERVATION_VARIABLES,
        'action_space': DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS,
        'action_variables': DEFAULT_OFFICE_ACTION_VARIABLES,
        'action_mapping': DEFAULT_OFFICE_ACTION_MAPPING,
        'weather_variability': (1.0, 0.0, 0.001),
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variable': [
                'Zone Air Temperature(Core_bottom)',
                'Zone Air Temperature(TopFloor_Plenum)',
                'Zone Air Temperature(MidFloor_Plenum)',
                'Zone Air Temperature(FirstFloor_Plenum)',
                'Zone Air Temperature(Core_mid)',
                'Zone Air Temperature(Core_top)',
                'Zone Air Temperature(Perimeter_top_ZN_3)',
                'Zone Air Temperature(Perimeter_top_ZN_2)',
                'Zone Air Temperature(Perimeter_top_ZN_1)',
                'Zone Air Temperature(Perimeter_top_ZN_4)',
                'Zone Air Temperature(Perimeter_bot_ZN_3)',
                'Zone Air Temperature(Perimeter_bot_ZN_2)',
                'Zone Air Temperature(Perimeter_bot_ZN_1)',
                'Zone Air Temperature(Perimeter_bot_ZN_4)',
                'Zone Air Temperature(Perimeter_mid_ZN_3)',
                'Zone Air Temperature(Perimeter_mid_ZN_2)',
                'Zone Air Temperature(Perimeter_mid_ZN_1)',
                'Zone Air Temperature(Perimeter_mid_ZN_4)'
            ],
            'energy_variable': 'Facility Total HVAC Electricity Demand Rate(Whole Building)',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        },
        'env_name': 'office-cool-continuous-stochastic-v1',
        'action_definition': DEFAULT_OFFICE_ACTION_DEFINITION})
