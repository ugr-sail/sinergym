"""Constants used in whole project."""

import os

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
# LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
LOG_FORMAT = "[%(name)s] (%(levelname)s) : %(message)s"


# ---------------------------------------------------------------------------- #
#                       Default Eplus Environments values                      #
# ---------------------------------------------------------------------------- #

# ---------------------------------- GENERAL --------------------------------- #
DEFAULT_TIME_VARIABLES = ['month', 'day_of_month', 'hour']

# -------------------------------------5ZONE---------------------------------- #

DEFAULT_5ZONE_VARIABLES = {
    'outdoor_temperature': (
        'Site Outdoor Air DryBulb Temperature',
        'Environment'),
    'outdoor_humidity': (
        'Site Outdoor Air Relative Humidity',
        'Environment'),
    'wind_speed': (
        'Site Wind Speed',
        'Environment'),
    'wind_direction': (
        'Site Wind Direction',
        'Environment'),
    'diffuse_solar_radiation': (
        'Site Diffuse Solar Radiation Rate per Area',
        'Environment'),
    'direct_solar_radiation': (
        'Site Direct Solar Radiation Rate per Area',
        'Environment'),
    'htg_setpoint': (
        'Zone Thermostat Heating Setpoint Temperature',
        'SPACE1-1'),
    'clg_setpoint': (
        'Zone Thermostat Cooling Setpoint Temperature',
        'SPACE1-1'),
    'air_temperature': (
        'Zone Air Temperature',
        'SPACE1-1'),
    'air_humidity': (
        'Zone Air Relative Humidity',
        'SPACE1-1'),
    'people_occupant': (
        'Zone People Occupant Count',
        'SPACE1-1'),
    'co2_emission': (
        'Environmental Impact Total CO2 Emissions Carbon Equivalent Mass',
        'site'),
    'HVAC_electricity_demand_rate': (
        'Facility Total HVAC Electricity Demand Rate',
        'Whole Building')
}

DEFAULT_5ZONE_METERS = {'total_electricity_HVAC': 'Electricity:HVAC'}

DEFAULT_5ZONE_ACTUATORS = {
    'Heating_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'HTG-SETP-SCH'),
    'Cooling_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'CLG-SETP-SCH')}

DEFAULT_5ZONE_ACTION_MAPPING = {
    0: (15, 30),
    1: (16, 29),
    2: (17, 28),
    3: (18, 27),
    4: (19, 26),
    5: (20, 25),
    6: (21, 24),
    7: (22, 23),
    8: (22, 22),
    9: (21, 21)
}

DEFAULT_5ZONE_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_5ZONE_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5], dtype=np.float32),
    high=np.array([22.5, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32
)

# ----------------------------------DATACENTER--------------------------------- #
DEFAULT_DATACENTER_VARIABLES = {
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
    'west_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'West Zone'),
    'east_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'East Zone'),
    'west_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'West Zone'),
    'east_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'East Zone'),
    'west_zone_temperature': ('Zone Air Temperature', 'West Zone'),
    'east_zone_temperature': ('Zone Air Temperature', 'East Zone'),
    'west_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'West Zone PEOPLE'),
    'east_zone_thermal_comfort_mean_radiant_temperature': ('Zone Thermal Comfort Mean Radiant Temperature', 'East Zone PEOPLE'),
    'west_zone_air_humidity': ('Zone Air Relative Humidity', 'West Zone'),
    'east_zone_air_humidity': ('Zone Air Relative Humidity', 'East Zone'),
    'west_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'West Zone PEOPLE'),
    'east_zone_thermal_comfort_clothing_value': ('Zone Thermal Comfort Clothing Value', 'East Zone PEOPLE'),
    'west_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'West Zone PEOPLE'),
    'east_zone_thermal_comfort_fanger_model_ppd': ('Zone Thermal Comfort Fanger Model PPD', 'East Zone PEOPLE'),
    'west_zone_people_occupant': ('Zone People Occupant Count', 'West Zone'),
    'east_zone_people_occupant': ('Zone People Occupant Count', 'East Zone'),
    'west_people_air_temperature': ('People Air Temperature', 'West Zone PEOPLE'),
    'east_people_air_temperature': ('People Air Temperature', 'East Zone PEOPLE'),
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
}

DEFAULT_DATACENTER_METERS = {}

DEFAULT_DATACENTER_ACTUATORS = {
    'Heating_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Heating Setpoints'),
    'Cooling_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Cooling Setpoints')}

DEFAULT_DATACENTER_ACTION_MAPPING = {
    0: (15, 30),
    1: (16, 29),
    2: (17, 28),
    3: (18, 27),
    4: (19, 26),
    5: (20, 25),
    6: (21, 24),
    7: (22, 23),
    8: (22, 22),
    9: (21, 21)
}

DEFAULT_DATACENTER_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_DATACENTER_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5], dtype=np.float32),
    high=np.array([22.5, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32)

# ----------------------------------WAREHOUSE--------------------------------- #
DEFAULT_WAREHOUSE_VARIABLES = {
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
    'office_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'Zone1 Office'),
    'office_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'Zone1 Office'),
    'office_temperature': ('Zone Air Temperature', 'Zone1 Office'),
    'office_humidity': ('Zone Air Relative Humidity', 'Zone1 Office'),
    'office_people_count': ('Zone People Occupant Count', 'Zone1 Office'),
    'fstorage_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'Zone2 Fine Storage'),
    'fstorage_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'Zone2 Fine Storage'),
    'fstorage_temperature': ('Zone Air Temperature', 'Zone2 Fine Storage'),
    'fstorage_humidity': ('Zone Air Relative Humidity', 'Zone2 Fine Storage'),
    'bstorage_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'Zone3 Bulk Storage'),
    'bstorage_temperature': ('Zone Air Temperature', 'Zone3 Bulk Storage'),
    'bstorage_humidity': ('Zone Air Relative Humidity', 'Zone3 Bulk Storage'),
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
}

DEFAULT_WAREHOUSE_METERS = {}

DEFAULT_WAREHOUSE_ACTUATORS = {
    'Office_Heating_RL': (
        'Schedule:Year',
        'Schedule Value',
        'Office Heating Schedule'),
    'Office_Cooling_RL': (
        'Schedule:Year',
        'Schedule Value',
        'Office Cooling Schedule'),
    'FineStorage_Heating_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Fine Storage Heating Setpoint Schedule'),
    'FineStorage_Cooling_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Fine Storage Cooling Setpoint Schedule'),
    'BulkStorage_Heating_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Bulk Storage Heating Setpoint Schedule')}

DEFAULT_WAREHOUSE_ACTION_MAPPING = {
    0: (15, 30, 15, 30, 15),
    1: (16, 29, 16, 29, 16),
    2: (17, 28, 17, 28, 17),
    3: (18, 27, 18, 27, 18),
    4: (19, 26, 19, 26, 19),
    5: (20, 25, 20, 25, 20),
    6: (21, 24, 21, 24, 21),
    7: (22, 23, 22, 23, 22),
    8: (22, 22, 22, 22, 23),
    9: (21, 21, 21, 21, 24)
}

DEFAULT_WAREHOUSE_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_WAREHOUSE_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5, 15.0, 22.5, 15.0], dtype=np.float32),
    high=np.array([22.5, 30.0, 22.5, 30.0, 22.5], dtype=np.float32),
    shape=(5,),
    dtype=np.float32)

# ----------------------------------OFFICE--------------------------------- #

DEFAULT_OFFICE_VARIABLES = {
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'Core_bottom'),
    'clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'Core_bottom'),
    'zone1_temperature': ('Zone Air Temperature', 'Core_bottom'),
    'zone2_temperature': ('Zone Air Temperature', 'TopFloor_Plenum'),
    'zone3_temperature': ('Zone Air Temperature', 'MidFloor_Plenum'),
    'zone4_temperature': ('Zone Air Temperature', 'FirstFloor_Plenum'),
    'zone5_temperature': ('Zone Air Temperature', 'Core_mid'),
    'zone6_temperature': ('Zone Air Temperature', 'Core_top'),
    'zone7_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_3'),
    'zone8_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_2'),
    'zone9_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_1'),
    'zone10_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_4'),
    'zone11_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_3'),
    'zone12_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_2'),
    'zone13_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_1'),
    'zone14_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_4'),
    'zone15_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_3'),
    'zone16_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_2'),
    'zone17_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_1'),
    'zone18_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_4'),
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
}

DEFAULT_OFFICE_METERS = {}

DEFAULT_OFFICE_ACTUATORS = {
    'Office_Heating_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'HTGSETP_SCH_YES_OPTIMUM'),
    'Office_Cooling_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'CLGSETP_SCH_YES_OPTIMUM')}

DEFAULT_OFFICE_ACTION_MAPPING = {
    0: (15, 30),
    1: (16, 29),
    2: (17, 28),
    3: (18, 27),
    4: (19, 26),
    5: (20, 25),
    6: (21, 24),
    7: (22, 23),
    8: (22, 22),
    9: (21, 21)
}

DEFAULT_OFFICE_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_OFFICE_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5], dtype=np.float32),
    high=np.array([22.5, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32)

# ----------------------------------OFFICEGRID---------------------------- #

DEFAULT_OFFICEGRID_VARIABLES = {
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building'),
    'battery_charge_state': ('Electric Storage Simple Charge State', 'Battery'),
    'clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'Basement'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
    'htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'Basement'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'zone10_humidity': ('Zone Air Relative Humidity', 'Perimeter_mid_ZN_2'),
    'zone10_people_count': ('Zone People Occupant Count', 'Perimeter_mid_ZN_2'),
    'zone10_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_2'),
    'zone11_humidity': ('Zone Air Relative Humidity', 'Perimeter_mid_ZN_3'),
    'zone11_people_count': ('Zone People Occupant Count', 'Perimeter_mid_ZN_3'),
    'zone11_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_3'),
    'zone12_humidity': ('Zone Air Relative Humidity', 'Perimeter_mid_ZN_4'),
    'zone12_people_count': ('Zone People Occupant Count', 'Perimeter_mid_ZN_4'),
    'zone12_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_4'),
    'zone13_humidity': ('Zone Air Relative Humidity', 'Perimeter_top_ZN_1'),
    'zone13_people_count': ('Zone People Occupant Count', 'Perimeter_top_ZN_1'),
    'zone13_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_1'),
    'zone14_humidity': ('Zone Air Relative Humidity', 'Perimeter_top_ZN_2'),
    'zone14_people_count': ('Zone People Occupant Count', 'Perimeter_top_ZN_2'),
    'zone14_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_2'),
    'zone15_humidity': ('Zone Air Relative Humidity', 'Perimeter_top_ZN_3'),
    'zone15_people_count': ('Zone People Occupant Count', 'Perimeter_top_ZN_3'),
    'zone15_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_3'),
    'zone16_humidity': ('Zone Air Relative Humidity', 'Perimeter_top_ZN_4'),
    'zone16_people_count': ('Zone People Occupant Count', 'Perimeter_top_ZN_4'),
    'zone16_temperature': ('Zone Air Temperature', 'Perimeter_top_ZN_4'),
    'zone17_humidity': ('Zone Air Relative Humidity', 'GroundFloor_Plenum'),
    'zone17_temperature': ('Zone Air Temperature', 'GroundFloor_Plenum'),
    'zone18_humidity': ('Zone Air Relative Humidity', 'MidFloor_Plenum'),
    'zone18_temperature': ('Zone Air Temperature', 'MidFloor_Plenum'),
    'zone19_humidity': ('Zone Air Relative Humidity', 'TopFloor_Plenum'),
    'zone19_temperature': ('Zone Air Temperature', 'TopFloor_Plenum'),
    'zone1_humidity': ('Zone Air Relative Humidity', 'Basement'),
    'zone1_people_count': ('Zone People Occupant Count', 'Basement'),
    'zone1_temperature': ('Zone Air Temperature', 'Basement'),
    'zone2_humidity': ('Zone Air Relative Humidity', 'core_bottom'),
    'zone2_people_count': ('Zone People Occupant Count', 'core_bottom'),
    'zone2_temperature': ('Zone Air Temperature', 'core_bottom'),
    'zone3_humidity': ('Zone Air Relative Humidity', 'core_mid'),
    'zone3_people_count': ('Zone People Occupant Count', 'core_mid'),
    'zone3_temperature': ('Zone Air Temperature', 'core_mid'),
    'zone4_humidity': ('Zone Air Relative Humidity', 'core_top'),
    'zone4_people_count': ('Zone People Occupant Count', 'core_top'),
    'zone4_temperature': ('Zone Air Temperature', 'core_top'),
    'zone5_humidity': ('Zone Air Relative Humidity', 'Perimeter_bot_ZN_1'),
    'zone5_people_count': ('Zone People Occupant Count', 'Perimeter_bot_ZN_1'),
    'zone5_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_1'),
    'zone6_humidity': ('Zone Air Relative Humidity', 'Perimeter_bot_ZN_2'),
    'zone6_people_count': ('Zone People Occupant Count', 'Perimeter_bot_ZN_2'),
    'zone6_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_2'),
    'zone7_humidity': ('Zone Air Relative Humidity', 'Perimeter_bot_ZN_3'),
    'zone7_people_count': ('Zone People Occupant Count', 'Perimeter_bot_ZN_3'),
    'zone7_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_3'),
    'zone8_humidity': ('Zone Air Relative Humidity', 'Perimeter_bot_ZN_4'),
    'zone8_people_count': ('Zone People Occupant Count', 'Perimeter_bot_ZN_4'),
    'zone8_temperature': ('Zone Air Temperature', 'Perimeter_bot_ZN_4'),
    'zone9_humidity': ('Zone Air Relative Humidity', 'Perimeter_mid_ZN_1'),
    'zone9_people_count': ('Zone People Occupant Count', 'Perimeter_mid_ZN_1'),
    'zone9_temperature': ('Zone Air Temperature', 'Perimeter_mid_ZN_1')
}

DEFAULT_OFFICEGRID_METERS = {}

DEFAULT_OFFICEGRID_ACTUATORS = {
    'Heating_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'HTGSETP_SCH'),
    'Cooling_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'CLGSETP_SCH'),
    'Charge_Rate_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Charge Schedule'),
    'Discharge_Rate_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Discharge Schedule')
}

DEFAULT_OFFICEGRID_ACTION_MAPPING = {
    0: (15, 30, 0.0, 0.0),
    1: (16, 29, 0.0, 0.0),
    2: (17, 28, 0.0, 0.0),
    3: (18, 27, 0.0, 0.0),
    4: (19, 26, 0.0, 0.0),
    5: (20, 25, 0.0, 0.0),
    6: (21, 24, 0.0, 0.0),
    7: (22, 23, 0.0, 0.0),
    8: (22, 22, 0.0, 0.0),
    9: (21, 21, 0.0, 0.0)
}

DEFAULT_OFFICEGRID_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_OFFICEGRID_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5, 0.0, 0.0], dtype=np.float32),
    high=np.array([22.5, 30.0, 1.0, 1.0], dtype=np.float32),
    shape=(4,),
    dtype=np.float32
)

# ----------------------------------SHOP--------------------- #

# Kibam is the name of ElectricLoadCenter:Storage:Battery object
DEFAULT_SHOP_VARIABLES = {
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building'),
    'clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'ZN_1_FLR_1_SEC_5'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
    'htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'ZN_1_FLR_1_SEC_5'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'storage_battery_charge_state': ('Electric Storage Battery Charge State', 'Kibam'),
    'storage_charge_energy': ('Electric Storage Charge Energy', 'Kibam'),
    'storage_charge_power': ('Electric Storage Charge Power', 'Kibam'),
    'storage_discharge_energy': ('Electric Storage Discharge Energy', 'Kibam'),
    'storage_discharge_power': ('Electric Storage Discharge Power', 'Kibam'),
    'storage_thermal_loss_energy': ('Electric Storage Thermal Loss Energy', 'Kibam'),
    'storage_thermal_loss_rate': ('Electric Storage Thermal Loss Rate', 'Kibam'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'zone1_humidity': ('Zone Air Relative Humidity', 'ZN_1_FLR_1_SEC_1'),
    'zone1_people_count': ('Zone People Occupant Count', 'ZN_1_FLR_1_SEC_1'),
    'zone1_temperature': ('Zone Air Temperature', 'ZN_1_FLR_1_SEC_1'),
    'zone2_humidity': ('Zone Air Relative Humidity', 'ZN_1_FLR_1_SEC_2'),
    'zone2_people_count': ('Zone People Occupant Count', 'ZN_1_FLR_1_SEC_2'),
    'zone2_temperature': ('Zone Air Temperature', 'ZN_1_FLR_1_SEC_2'),
    'zone3_humidity': ('Zone Air Relative Humidity', 'ZN_1_FLR_1_SEC_3'),
    'zone3_people_count': ('Zone People Occupant Count', 'ZN_1_FLR_1_SEC_3'),
    'zone3_temperature': ('Zone Air Temperature', 'ZN_1_FLR_1_SEC_3'),
    'zone4_humidity': ('Zone Air Relative Humidity', 'ZN_1_FLR_1_SEC_4'),
    'zone4_people_count': ('Zone People Occupant Count', 'ZN_1_FLR_1_SEC_4'),
    'zone4_temperature': ('Zone Air Temperature', 'ZN_1_FLR_1_SEC_4'),
    'zone5_humidity': ('Zone Air Relative Humidity', 'ZN_1_FLR_1_SEC_5'),
    'zone5_people_count': ('Zone People Occupant Count', 'ZN_1_FLR_1_SEC_5'),
    'zone5_temperature': ('Zone Air Temperature', 'ZN_1_FLR_1_SEC_5')}

DEFAULT_SHOP_METERS = {}

DEFAULT_SHOP_ACTUATORS = {
    'Heating_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'HTGSETP_SCH'),
    'Cooling_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'CLGSETP_SCH')}

DEFAULT_SHOP_ACTION_MAPPING = {
    0: (15, 30),
    1: (16, 29),
    2: (17, 28),
    3: (18, 27),
    4: (19, 26),
    5: (20, 25),
    6: (21, 24),
    7: (22, 23),
    8: (22, 22),
    9: (21, 21)
}

DEFAULT_SHOP_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

DEFAULT_SHOP_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
    low=np.array([15.0, 22.5], dtype=np.float32),
    high=np.array([22.5, 30.0], dtype=np.float32),
    shape=(2,),
    dtype=np.float32
)


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
