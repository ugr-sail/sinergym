import os

import gymnasium as gym
from gymnasium.envs.registration import register

from sinergym.utils.constants import *
from sinergym.utils.rewards import *

# Set __version__ in module
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()


# 0) Demo environment
register(
    id='Eplus-demo-v1',
    entry_point='sinergym.envs:EplusEnv',
    kwargs={
        'building_file': '5ZoneAutoDXVAV.epJSON',
        'weather_files': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw',
        'action_space': DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS,
        'time_variables': DEFAULT_TIME_VARIABLES,
        'variables': DEFAULT_5ZONE_VARIABLES,
        'meters': DEFAULT_5ZONE_METERS,
        'actuators': DEFAULT_5ZONE_ACTUATORS,
        'reward': LinearReward,
        'reward_kwargs': {
            'temperature_variables': 'air_temperature',
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)},
        'env_name': 'demo-v1'})


id_bases = ['5zone', 'datacenter', 'warehouse', 'office', 'officegrid', 'shop']
id_specifics = [
    'hot-discrete',
    'mixed-discrete',
    'cool-discrete',
    'hot-continuous',
    'mixed-continuous',
    'cool-continuous',
    'hot-discrete-stochastic',
    'mixed-discrete-stochastic',
    'cool-discrete-stochastic',
    'hot-continuous-stochastic',
    'mixed-continuous-stochastic',
    'cool-continuous-stochastic']
variation = (1.0, 0.0, 0.001)
for building in id_bases:
    reg_kwargs = {'time_variables': DEFAULT_TIME_VARIABLES}

    if building == '5zone':
        reg_kwargs['building_file'] = '5ZoneAutoDXVAV.epJSON'
        action_space_continuous = DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_5ZONE_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_5ZONE_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_5ZONE_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_5ZONE_VARIABLES
        reg_kwargs['meters'] = DEFAULT_5ZONE_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': 'air_temperature',
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)}

    elif building == 'datacenter':
        reg_kwargs['building_file'] = '2ZoneDataCenterHVAC_wEconomizer.epJSON'
        action_space_continuous = DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_DATACENTER_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_DATACENTER_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_DATACENTER_VARIABLES
        reg_kwargs['meters'] = DEFAULT_DATACENTER_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': [
                'west_zone_temperature',
                'east_zone_temperature'],
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)}

    elif building == 'warehouse':
        reg_kwargs['building_file'] = 'ASHRAE901_Warehouse_STD2019_Denver.epJSON'
        action_space_continuous = DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_WAREHOUSE_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_WAREHOUSE_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_WAREHOUSE_VARIABLES
        reg_kwargs['meters'] = DEFAULT_WAREHOUSE_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': [
                'office_temperature',
                'fstorage_temperature',
                'bstorage_temperature'],
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                18,
                27),
            'range_comfort_summer': (
                18,
                27)}

    elif building == 'office':
        reg_kwargs['building_file'] = 'ASHRAE901_OfficeMedium_STD2019_Denver.epJSON'
        action_space_continuous = DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_OFFICE_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_OFFICE_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_OFFICE_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_OFFICE_VARIABLES
        reg_kwargs['meters'] = DEFAULT_OFFICE_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': [
                'zone1_temperature',
                'zone2_temperature',
                'zone3_temperature',
                'zone4_temperature',
                'zone5_temperature',
                'zone6_temperature',
                'zone7_temperature',
                'zone8_temperature',
                'zone9_temperature',
                'zone10_temperature',
                'zone11_temperature',
                'zone12_temperature',
                'zone13_temperature',
                'zone14_temperature',
                'zone15_temperature',
                'zone16_temperature',
                'zone17_temperature',
                'zone18_temperature'
            ],
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (18, 27),
            'range_comfort_summer': (18, 27)
        }

    elif building == 'officegrid':
        reg_kwargs['building_file'] = 'LrgOff_GridStorageScheduled.epJSON'
        action_space_continuous = DEFAULT_OFFICEGRID_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_OFFICEGRID_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_OFFICEGRID_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_OFFICEGRID_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_OFFICEGRID_VARIABLES
        reg_kwargs['meters'] = DEFAULT_OFFICEGRID_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': [
                'zone1_temperature',
                'zone2_temperature',
                'zone3_temperature',
                'zone4_temperature',
                'zone5_temperature',
                'zone6_temperature',
                'zone7_temperature',
                'zone8_temperature',
                'zone9_temperature',
                'zone10_temperature',
                'zone11_temperature',
                'zone12_temperature',
                'zone13_temperature',
                'zone14_temperature',
                'zone15_temperature',
                'zone16_temperature',
                'zone17_temperature',
                'zone18_temperature',
                'zone19_temperature'],
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)}

    elif building == 'shop':
        reg_kwargs['building_file'] = 'ShopWithPVandBattery.epJSON'
        action_space_continuous = DEFAULT_SHOP_ACTION_SPACE_CONTINUOUS
        action_space_discrete = DEFAULT_SHOP_ACTION_SPACE_DISCRETE
        reg_kwargs['action_mapping'] = DEFAULT_SHOP_ACTION_MAPPING
        reg_kwargs['actuators'] = DEFAULT_SHOP_ACTUATORS
        reg_kwargs['variables'] = DEFAULT_SHOP_VARIABLES
        reg_kwargs['meters'] = DEFAULT_SHOP_METERS
        reg_kwargs['reward'] = LinearReward
        reg_kwargs['reward_kwargs'] = {
            'temperature_variables': [
                'zone1_temperature',
                'zone2_temperature',
                'zone3_temperature',
                'zone4_temperature',
                'zone5_temperature'],
            'energy_variables': 'HVAC_electricity_demand_rate',
            'range_comfort_winter': (
                20.0,
                23.5),
            'range_comfort_summer': (
                23.0,
                26.0)}

    for id_specific in id_specifics:
        id = 'Eplus-' + building + '-' + id_specific + '-v1'
        reg_kwargs['env_name'] = building + '-' + id_specific + '-v1'

        register_conf = id_specific.split('-')
        if register_conf[0] == 'hot':
            reg_kwargs['weather_files'] = 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'
        elif register_conf[0] == 'mixed':
            reg_kwargs['weather_files'] = 'USA_NY_New.York-J.F.Kennedy.Intl.AP.744860_TMY3.epw'
        elif register_conf[0] == 'cool':
            reg_kwargs['weather_files'] = 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw'

        if register_conf[1] == 'discrete':
            reg_kwargs['action_space'] = action_space_discrete
        elif register_conf[1] == 'continuous':
            reg_kwargs['action_space'] = action_space_continuous

        reg_kwargs['weather_variability'] = None
        if len(register_conf) == 3:
            if register_conf[2] == 'stochastic':
                reg_kwargs['weather_variability'] = variation

        register(
            id=id,
            entry_point='sinergym.envs:EplusEnv',
            kwargs=reg_kwargs.copy()
        )
