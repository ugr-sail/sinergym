"""Constants used in whole project."""

import os
from typing import List

import gymnasium as gym
import numpy as np
import pkg_resources

# ---------------------------------------------------------------------------- #
#                               Generic constants                              #
# ---------------------------------------------------------------------------- #
# Sinergym Data path
PKG_DATA_PATH = pkg_resources.resource_filename(
    'sinergym', 'data/')
# Weekday encoding for simulations
WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
# Default start year (Non leap year please)
YEAR = 1991
# cwd
CWD = os.getcwd()

# Logger values (environment layer, simulator layer and modeling layer)
LOG_ENV_LEVEL = 'INFO'
LOG_SIM_LEVEL = 'INFO'
LOG_MODEL_LEVEL = 'INFO'
LOG_WRAPPERS_LEVEL = 'INFO'
LOG_REWARD_LEVEL = 'INFO'
# LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
LOG_FORMAT = "[%(name)s] (%(levelname)s) : %(message)s"


# ---------------------------------------------------------------------------- #
#              Default Eplus discrete environments action mappings             #
# ---------------------------------------------------------------------------- #

# -------------------------------------5ZONE---------------------------------- #


def DEFAULT_5ZONE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]


# ----------------------------------DATACENTER--------------------------------- #


DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)


def DEFAULT_DATACENTER_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]


DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5], dtype=np.float32),
    high=np.array([22.5, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32)

# ----------------------------------WAREHOUSE--------------------------------- #


def DEFAULT_WAREHOUSE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30, 15, 30, 15],
        1: [16, 29, 16, 29, 16],
        2: [17, 28, 17, 28, 17],
        3: [18, 27, 18, 27, 18],
        4: [19, 26, 19, 26, 19],
        5: [20, 25, 20, 25, 20],
        6: [21, 24, 21, 24, 21],
        7: [22, 23, 22, 23, 22.5],
        8: [22, 22.5, 22, 22.5, 22.5],
        9: [21, 22.5, 21, 22.5, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICE--------------------------------- #


def DEFAULT_OFFICE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICEGRID---------------------------- #


def DEFAULT_OFFICEGRID_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30, 0.0, 0.0],
        1: [16, 29, 0.0, 0.0],
        2: [17, 28, 0.0, 0.0],
        3: [18, 27, 0.0, 0.0],
        4: [19, 26, 0.0, 0.0],
        5: [20, 25, 0.0, 0.0],
        6: [21, 24, 0.0, 0.0],
        7: [22, 23, 0.0, 0.0],
        8: [22, 22.5, 0.0, 0.0],
        9: [21, 22.5, 0.0, 0.0]
    }

    return mapping[action]

# ----------------------------------SHOP--------------------- #


def DEFAULT_SHOP_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# -------------------------------- AUTOBALANCE ------------------------------- #

DEFAULT_AUTOBALANCE_VARIABLES = {
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),

    'htg_setpoint_livroom': ('Zone Thermostat Heating Setpoint Temperature', 'FirstFloor:LivingRoom'),
    'htg_setpoint_kitchen': ('Zone Thermostat Heating Setpoint Temperature', 'FirstFloor:Kitchen'),
    'htg_setpoint_bed1': ('Zone Thermostat Heating Setpoint Temperature', 'SecondFloor:Bedroom01'),
    'htg_setpoint_bed2': ('Zone Thermostat Heating Setpoint Temperature', 'SecondFloor:Bedroom02'),
    'htg_setpoint_bed3': ('Zone Thermostat Heating Setpoint Temperature', 'SecondFloor:Bedroom03'),

    'clg_setpoint_livroom': ('Zone Thermostat Cooling Setpoint Temperature', 'FirstFloor:LivingRoom'),
    'clg_setpoint_kitchen': ('Zone Thermostat Cooling Setpoint Temperature', 'FirstFloor:Kitchen'),
    'clg_setpoint_bed1': ('Zone Thermostat Cooling Setpoint Temperature', 'SecondFloor:Bedroom01'),
    'clg_setpoint_bed2': ('Zone Thermostat Cooling Setpoint Temperature', 'SecondFloor:Bedroom02'),
    'clg_setpoint_bed3': ('Zone Thermostat Cooling Setpoint Temperature', 'SecondFloor:Bedroom03'),

    'air_temperature_livroom': ('Zone Air Temperature', 'FirstFloor:LivingRoom'),
    'air_temperature_kitchen': ('Zone Air Temperature', 'FirstFloor:Kitchen'),
    'air_temperature_bed1': ('Zone Air Temperature', 'SecondFloor:Bedroom01'),
    'air_temperature_bed2': ('Zone Air Temperature', 'SecondFloor:Bedroom02'),
    'air_temperature_bed3': ('Zone Air Temperature', 'SecondFloor:Bedroom03'),

    'air_humidity_livroom': ('Zone Air Relative Humidity', 'FirstFloor:LivingRoom'),
    'air_humidity_kitchen': ('Zone Air Relative Humidity', 'FirstFloor:Kitchen'),
    'air_humidity_bed1': ('Zone Air Relative Humidity', 'SecondFloor:Bedroom01'),
    'air_humidity_bed2': ('Zone Air Relative Humidity', 'SecondFloor:Bedroom02'),
    'air_humidity_bed3': ('Zone Air Relative Humidity', 'SecondFloor:Bedroom03'),

    'thermal_comfort_mean_radiant_temperature_livroom': ('Zone Thermal Comfort Mean Radiant Temperature', 'People FirstFloor:LivingRoom'),
    'thermal_comfort_mean_radiant_temperature_kitchen': ('Zone Thermal Comfort Mean Radiant Temperature', 'People FirstFloor:Kitchen'),
    'thermal_comfort_mean_radiant_temperature_bed1': ('Zone Thermal Comfort Mean Radiant Temperature', 'People SecondFloor:Bedroom01'),
    'thermal_comfort_mean_radiant_temperature_bed2': ('Zone Thermal Comfort Mean Radiant Temperature', 'People SecondFloor:Bedroom02'),
    'thermal_comfort_mean_radiant_temperature_bed3': ('Zone Thermal Comfort Mean Radiant Temperature', 'People SecondFloor:Bedroom03'),

    'thermal_comfort_clothing_value_livroom': ('Zone Thermal Comfort Clothing Value', 'People FirstFloor:LivingRoom'),
    'thermal_comfort_clothing_value_kitchen': ('Zone Thermal Comfort Clothing Value', 'People FirstFloor:Kitchen'),
    'thermal_comfort_clothing_value_bed1': ('Zone Thermal Comfort Clothing Value', 'People SecondFloor:Bedroom01'),
    'thermal_comfort_clothing_value_bed2': ('Zone Thermal Comfort Clothing Value', 'People SecondFloor:Bedroom02'),
    'thermal_comfort_clothing_value_bed3': ('Zone Thermal Comfort Clothing Value', 'People SecondFloor:Bedroom03'),

    'thermal_comfort_fanger_model_ppd_livroom': ('Zone Thermal Comfort Fanger Model PPD', 'People FirstFloor:LivingRoom'),
    'thermal_comfort_fanger_model_ppd_kitchen': ('Zone Thermal Comfort Fanger Model PPD', 'People FirstFloor:Kitchen'),
    'thermal_comfort_fanger_model_ppd_bed1': ('Zone Thermal Comfort Fanger Model PPD', 'People SecondFloor:Bedroom01'),
    'thermal_comfort_fanger_model_ppd_bed2': ('Zone Thermal Comfort Fanger Model PPD', 'People SecondFloor:Bedroom02'),
    'thermal_comfort_fanger_model_ppd_bed3': ('Zone Thermal Comfort Fanger Model PPD', 'People SecondFloor:Bedroom03'),

    'people_occupant_livroom': ('Zone People Occupant Count', 'FirstFloor:LivingRoom'),
    'people_occupant_kitchen': ('Zone People Occupant Count', 'FirstFloor:Kitchen'),
    'people_occupant_bed1': ('Zone People Occupant Count', 'SecondFloor:Bedroom01'),
    'people_occupant_bed2': ('Zone People Occupant Count', 'SecondFloor:Bedroom02'),
    'people_occupant_bed3': ('Zone People Occupant Count', 'SecondFloor:Bedroom03'),

    'people_air_temperature_livroom': ('People Air Temperature', 'People FirstFloor:LivingRoom'),
    'people_air_temperature_kitchen': ('People Air Temperature', 'People FirstFloor:Kitchen'),
    'people_air_temperature_bed1': ('People Air Temperature', 'People SecondFloor:Bedroom01'),
    'people_air_temperature_bed2': ('People Air Temperature', 'People SecondFloor:Bedroom02'),
    'people_air_temperature_bed3': ('People Air Temperature', 'People SecondFloor:Bedroom03'),

    'radiant_hvac_outlet_temperature_livroom': ('Zone Radiant HVAC Outlet Temperature', 'FirstFloor:LivingRoom radiant surface'),
    'radiant_hvac_outlet_temperature_kitchen': ('Zone Radiant HVAC Outlet Temperature', 'FirstFloor:Kitchen radiant surface'),
    'radiant_hvac_outlet_temperature_bed1': ('Zone Radiant HVAC Outlet Temperature', 'SecondFloor:Bedroom01 radiant surface'),
    'radiant_hvac_outlet_temperature_bed2': ('Zone Radiant HVAC Outlet Temperature', 'SecondFloor:Bedroom02 radiant surface'),
    'radiant_hvac_outlet_temperature_bed3': ('Zone Radiant HVAC Outlet Temperature', 'SecondFloor:Bedroom03 radiant surface'),

    'radiant_hvac_inlet_temperature_livroom': ('Zone Radiant HVAC Inlet Temperature', 'FirstFloor:LivingRoom radiant surface'),
    'radiant_hvac_inlet_temperature_kitchen': ('Zone Radiant HVAC Inlet Temperature', 'FirstFloor:Kitchen radiant surface'),
    'radiant_hvac_inlet_temperature_bed1': ('Zone Radiant HVAC Inlet Temperature', 'SecondFloor:Bedroom01 radiant surface'),
    'radiant_hvac_inlet_temperature_bed2': ('Zone Radiant HVAC Inlet Temperature', 'SecondFloor:Bedroom02 radiant surface'),
    'radiant_hvac_inlet_temperature_bed3': ('Zone Radiant HVAC Inlet Temperature', 'SecondFloor:Bedroom03 radiant surface'),

    'surface_internal_source_location_temperature_livroom': ('Surface Internal Source Location Temperature', 'FirstFloor:LivingRoom_groundfloor_0_0_0'),
    'surface_internal_source_location_temperature_kitchen': ('Surface Internal Source Location Temperature', 'FirstFloor:Kitchen_groundfloor_0_0_0'),
    'surface_internal_source_location_temperature_bed1': ('Surface Internal Source Location Temperature', 'SecondFloor:Bedroom01_floor_0_0_0'),
    'surface_internal_source_location_temperature_bed2': ('Surface Internal Source Location Temperature', 'SecondFloor:Bedroom02_floor_0_0_0'),
    'surface_internal_source_location_temperature_bed3': ('Surface Internal Source Location Temperature', 'SecondFloor:Bedroom03_floor_0_0_0'),

    'surface_internal_user_specified_location_temperature_livroom': ('Surface Internal User Specified Location Temperature', 'FirstFloor:LivingRoom_groundfloor_0_0_0'),
    'surface_internal_user_specified_location_temperature_kitchen': ('Surface Internal User Specified Location Temperature', 'FirstFloor:Kitchen_groundfloor_0_0_0'),
    'surface_internal_user_specified_location_temperature_bed1': ('Surface Internal User Specified Location Temperature', 'SecondFloor:Bedroom01_floor_0_0_0'),
    'surface_internal_user_specified_location_temperature_bed2': ('Surface Internal User Specified Location Temperature', 'SecondFloor:Bedroom02_floor_0_0_0'),
    'surface_internal_user_specified_location_temperature_bed3': ('Surface Internal User Specified Location Temperature', 'SecondFloor:Bedroom03_floor_0_0_0'),

    'boiler_rate': ('Boiler NaturalGas Rate', 'Boiler'),
    'boiler_load_ratio': ('Boiler Part Load Ratio', 'Boiler'),

    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
}

DEFAULT_AUTOBALANCE_METERS = {}

DEFAULT_AUTOBALANCE_ACTUATORS = {
    # 'radiant_livroom': (
    #     'Hydronic Low Temp Radiant',
    #     'Water Mass Flow Rate',
    #     'FIRSTFLOOR:LIVINGROOM RADIANT SURFACE'),
    # 'radiant_kitchen': (
    #     'Hydronic Low Temp Radiant',
    #     'Water Mass Flow Rate',
    #     'FIRSTFLOOR:KITCHEN RADIANT SURFACE'),
    # 'radiant_bed1': (
    #     'Hydronic Low Temp Radiant',
    #     'Water Mass Flow Rate',
    #     'SECONDFLOOR:BEDROOM01 RADIANT SURFACE'),
    # 'radiant_bed2': (
    #     'Hydronic Low Temp Radiant',
    #     'Water Mass Flow Rate',
    #     'SECONDFLOOR:BEDROOM02 RADIANT SURFACE'),
    # 'radiant_bed3': (
    #     'Hydronic Low Temp Radiant',
    #     'Water Mass Flow Rate',
    #     'SECONDFLOOR:BEDROOM03 RADIANT SURFACE'),
    'radiant_livroom': (
        'Schedule:Compact',
        'Schedule Value',
        'LIVING RADIANT AVAILAVILITY'),
    'radiant_kitchen': (
        'Schedule:Compact',
        'Schedule Value',
        'KITCHEN RADIANT AVAILAVILITY'),
    'radiant_bed1': (
        'Schedule:Compact',
        'Schedule Value',
        'BED1 RADIANT AVAILAVILITY'),
    'radiant_bed2': (
        'Schedule:Compact',
        'Schedule Value',
        'BED2 RADIANT AVAILAVILITY'),
    'radiant_bed3': (
        'Schedule:Compact',
        'Schedule Value',
        'BED3 RADIANT AVAILAVILITY'),
    'water_temperature': (
        'Schedule:Compact',
        'Schedule Value',
        'HEATING HIGH WATER TEMPERATURE SCHEDULE: ALWAYS 65.00')}

DEFAULT_AUTOBALANCE_ACTION_SPACE = gym.spaces.MultiDiscrete(nvec=np.array(
    [2, 2, 2, 2, 2, 11]), start=np.array([0, 0, 0, 0, 0, 25]))


# ----------------------------------HOSPITAL--------------------------------- #
# DEFAULT_HOSPITAL_OBSERVATION_VARIABLES = [
#     'Zone Air Temperature(Basement)',
#     'Facility Total HVAC Electricity Demand Rate(Whole Building)',
#     'Site Outdoor Air Drybulb Temperature(Environment)'
# ]

# DEFAULT_HOSPITAL_ACTION_VARIABLES = [
#     'hospital-heating-rl',
#     'hospital-cooling-rl',
# ]

# DEFAULT_HOSPITAL_OBSERVATION_SPACE = gym.spaces.Box(
#     low=-5e6,
#     high=5e6,
#     shape=(len(DEFAULT_HOSPITAL_OBSERVATION_VARIABLES) + 4,),
#     dtype=np.float32)

# DEFAULT_HOSPITAL_ACTION_MAPPING = {
#     0: (15, 30),
#     1: (16, 29),
#     2: (17, 28),
#     3: (18, 27),
#     4: (19, 26),
#     5: (20, 25),
#     6: (21, 24),
#     7: (22, 23),
#     8: (22, 22),
#     9: (21, 21)
# }

# DEFAULT_HOSPITAL_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

# DEFAULT_HOSPITAL_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
#     low=np.array([15.0, 22.5], dtype=np.float32),
#     high=np.array([22.5, 30.0], dtype=np.float32),
#     shape=(2,),
#     dtype=np.float32)

# DEFAULT_HOSPITAL_ACTION_DEFINITION = {
#     '': {'name': '', 'initial_value': 21},
#     '': {'name': '', 'initial_value': 25}
# }
