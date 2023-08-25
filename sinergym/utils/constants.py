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
#                          Normalization dictionaries                          #
# ---------------------------------------------------------------------------- #
RANGES_5ZONE = {
    'Cooling_Setpoint_RL': [22.500061, 29.999975],
    'HVAC_electricity_demand_rate': [0.0, 9266.941],
    'Heating_Setpoint_RL': [15.000055, 22.499983],
    'abs_comfort': [0.0, 11.988519049141097],
    'air_humidity': [6.678406, 100.0],
    'air_temperature': [14.573951, 35.488518],
    'clg_setpoint': [22.500061, 40.0],
    'co2_emission': [0.0, 0.0],
    'comfort_penalty': [-11.988519049141097, -0.0],
    'day_of_month': [1.0, 31.0],
    'diffuse_solar_radiation': [0.0, 566.0],
    'direct_solar_radiation': [0.0, 943.0],
    'hour': [0.0, 23.0],
    'htg_setpoint': [12.8, 22.499983],
    'month': [1.0, 12.0],
    'outdoor_humidity': [11.0, 100.0],
    'outdoor_temperature': [-13.9, 36.7],
    'people_occupant': [0.0, 11.0],
    'power_penalty': [-0.92669415425972, -0.0],
    'reward': [-6.0418286092765525, -0.0],
    'total_electricity_HVAC': [0.0, 8340247.5],
    'wind_direction': [0.0, 357.5],
    'wind_speed': [0.0, 17.5],
    # previous wrapper variables
    'air_temperature_previous': [14.573951, 35.488518],
    'htg_setpoint_previous': [12.8, 22.499983],
    'clg_setpoint_previous': [22.500061, 40.0],
}


RANGES_DATACENTER = {
    'Cooling_Setpoint_RL': [22.500013, 29.999939],
    'HVAC_electricity_demand_rate': [2588.566, 63001.07],
    'Heating_Setpoint_RL': [15.000068, 22.499886],
    'abs_comfort': [0.4147307441036325, 4.02247370086085],
    'comfort_penalty': [-4.02247370086085, -0.4147307441036325],
    'day_of_month': [1.0, 31.0],
    'diffuse_solar_radiation': [0.0, 566.0],
    'direct_solar_radiation': [0.0, 943.0],
    'east_clg_setpoint': [22.500013, 29.999939],
    'east_htg_setpoint': [15.000068, 22.499886],
    'east_people_air_temperature': [27.415401, 28.036936],
    'east_zone_air_humidity': [2.3427522, 55.571835],
    'east_zone_people_occupant': [0.0, 0.0],
    'east_zone_temperature': [27.41473, 28.040743],
    'east_zone_thermal_comfort_clothing_value': [0.0, 0.0],
    'east_zone_thermal_comfort_fanger_model_ppd': [10.543119, 49.092926],
    'east_zone_thermal_comfort_mean_radiant_temperature': [22.390726, 28.061495],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'outdoor_humidity': [11.0, 100.0],
    'outdoor_temperature': [-13.9, 36.7],
    'power_penalty': [-6.300107141158778, -0.2588565918381311],
    'reward': [-4.95668713459597, -0.3658607819916097],
    'west_clg_setpoint': [22.500013, 29.999939],
    'west_htg_setpoint': [15.000068, 22.499886],
    'west_people_air_temperature': [22.556389, 30.005949],
    'west_zone_air_humidity': [1.8639423, 65.87973],
    'west_zone_people_occupant': [0.0, 0.0],
    'west_zone_temperature': [22.50559, 30.00286],
    'west_zone_thermal_comfort_clothing_value': [0.0, 0.0],
    'west_zone_thermal_comfort_fanger_model_ppd': [5.0, 50.85751],
    'west_zone_thermal_comfort_mean_radiant_temperature': [19.512312, 27.929258],
    'wind_direction': [0.0, 357.5],
    'wind_speed': [0.0, 17.5]
}

RANGES_OFFICE = {
    'HVAC_electricity_demand_rate': [128.04878, 92465.625],
    'Office_Cooling_RL': [22.500011, 29.999939],
    'Office_Heating_RL': [15.000026, 22.499958],
    'abs_comfort': [0.0, 141.59013437028364],
    'clg_setpoint': [24.0, 26.7],
    'comfort_penalty': [-141.59013437028364, -0.0],
    'day_of_month': [1.0, 31.0],
    'hour': [0.0, 23.0],
    'htg_setpoint': [15.6, 21.0],
    'month': [1.0, 12.0],
    'outdoor_temperature': [-13.9, 36.7],
    'power_penalty': [-9.246562464728546, -0.0128048780487804],
    'reward': [-70.8014696241662, -0.0064024390243902],
    'zone10_temperature': [6.9716387, 31.086294],
    'zone11_temperature': [10.463041, 27.016315],
    'zone12_temperature': [9.359555, 28.509354],
    'zone13_temperature': [6.379665, 28.701172],
    'zone14_temperature': [9.24301, 28.822401],
    'zone15_temperature': [10.043248, 27.654871],
    'zone16_temperature': [9.221822, 30.033983],
    'zone17_temperature': [9.950507, 29.45511],
    'zone18_temperature': [9.164494, 30.495295],
    'zone1_temperature': [15.462753, 29.714643],
    'zone2_temperature': [7.869569, 30.158476],
    'zone3_temperature': [11.482575, 27.35403],
    'zone4_temperature': [12.908837, 27.442966],
    'zone5_temperature': [14.349397, 27.854753],
    'zone6_temperature': [12.3087225, 27.843727],
    'zone7_temperature': [7.785018, 28.037645],
    'zone8_temperature': [7.0626063, 30.149975],
    'zone9_temperature': [7.707243, 29.920996]
}

RANGES_WAREHOUSE = {
    'BulkStorage_Heating_RL': [15.000057, 22.499998],
    'FineStorage_Cooling_RL': [22.500261, 29.99987],
    'FineStorage_Heating_RL': [15.000031, 22.49985],
    'HVAC_electricity_demand_rate': [481.0, 21038.592],
    'Office_Cooling_RL': [22.500006, 29.999893],
    'Office_Heating_RL': [15.0000925, 22.499968],
    'abs_comfort': [0.0, 11.402154878464914],
    'bstorage_htg_setpoint': [10.0, 22.499998],
    'bstorage_humidity': [3.7934682, 84.23336],
    'bstorage_temperature': [14.726301, 34.805294],
    'comfort_penalty': [-11.402154878464914, -0.0],
    'day_of_month': [1.0, 31.0],
    'diffuse_solar_radiation': [0.0, 566.0],
    'direct_solar_radiation': [0.0, 943.0],
    'fstorage_clg_setpoint': [22.500261, 29.99987],
    'fstorage_htg_setpoint': [15.000031, 22.49985],
    'fstorage_humidity': [2.913434, 88.18356],
    'fstorage_temperature': [11.026907, 29.340189],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'office_clg_setpoint': [22.500006, 29.999893],
    'office_htg_setpoint': [15.0000925, 22.499968],
    'office_humidity': [4.292086, 76.80151],
    'office_people_count': [0.0, 5.0],
    'office_temperature': [14.793758, 29.4],
    'outdoor_humidity': [11.0, 100.0],
    'outdoor_temperature': [-13.9, 36.7],
    'power_penalty': [-2.10385911413932, -0.0481],
    'reward': [-5.744479004780133, -0.02405],
    'wind_direction': [0.0, 357.5],
    'wind_speed': [0.0, 17.5]
}

RANGES_OFFICEGRID = {
    'Charge_Rate_RL': [1.5944242e-05, 0.9999935],
    'Cooling_Setpoint_RL': [22.500025, 29.999788],
    'Discharge_Rate_RL': [4.9471855e-06, 0.9999837],
    'HVAC_electricity_demand_rate': [21.755526, 900154.0],
    'Heating_Setpoint_RL': [15.0000515, 22.499897],
    'abs_comfort': [0.0, 137.96098825079804],
    'battery_charge_state': [400000000000.0, 500288000000.0],
    'clg_setpoint': [22.500025, 29.999788],
    'comfort_penalty': [-137.96098825079804, -0.0],
    'day_of_month': [1.0, 31.0],
    'diffuse_solar_radiation': [0.0, 566.0],
    'direct_solar_radiation': [0.0, 943.0],
    'hour': [0.0, 23.0],
    'htg_setpoint': [15.0000515, 22.499874],
    'month': [1.0, 12.0],
    'outdoor_humidity': [11.0, 100.0],
    'outdoor_temperature': [-13.9, 36.7],
    'power_penalty': [-90.01540189065497, -0.0021755525640112],
    'reward': [-68.9906213846732, -0.0010877762820056],
    'wind_direction': [0.0, 357.5],
    'wind_speed': [0.0, 17.5],
    'zone10_humidity': [3.9014182, 73.02324],
    'zone10_people_count': [0.0, 10.870567],
    'zone10_temperature': [11.428269, 32.261997],
    'zone11_humidity': [3.9780824, 73.0646],
    'zone11_people_count': [0.0, 16.86829],
    'zone11_temperature': [11.442675, 28.362448],
    'zone12_humidity': [3.9511242, 72.61227],
    'zone12_people_count': [0.0, 10.870567],
    'zone12_temperature': [11.334436, 32.889572],
    'zone13_humidity': [3.8352282, 76.35926],
    'zone13_people_count': [0.0, 16.86866],
    'zone13_temperature': [10.459512, 31.427008],
    'zone14_humidity': [3.9198968, 77.068375],
    'zone14_people_count': [0.0, 10.870567],
    'zone14_temperature': [10.360552, 32.254837],
    'zone15_humidity': [3.9320192, 77.19026],
    'zone15_people_count': [0.0, 16.86829],
    'zone15_temperature': [10.355516, 28.488447],
    'zone16_humidity': [3.9447205, 76.14643],
    'zone16_people_count': [0.0, 10.870567],
    'zone16_temperature': [10.258962, 33.11481],
    'zone17_humidity': [4.492952, 60.51695],
    'zone17_temperature': [15.765747, 26.851715],
    'zone18_humidity': [4.483258, 60.423626],
    'zone18_temperature': [15.129617, 26.888666],
    'zone19_humidity': [6.124, 88.705696],
    'zone19_temperature': [7.136532, 28.596289],
    'zone1_humidity': [4.2410755, 61.915527],
    'zone1_people_count': [0.0, 95.88552],
    'zone1_temperature': [17.81559, 27.655548],
    'zone2_humidity': [4.202735, 64.462685],
    'zone2_people_count': [0.0, 136.29295],
    'zone2_temperature': [16.350735, 29.005371],
    'zone3_humidity': [4.1808662, 64.95672],
    'zone3_people_count': [0.0, 136.29295],
    'zone3_temperature': [15.501164, 29.200987],
    'zone4_humidity': [3.666935, 64.85253],
    'zone4_people_count': [0.0, 136.29295],
    'zone4_temperature': [14.567434, 29.30396],
    'zone5_humidity': [3.8073199, 73.73933],
    'zone5_people_count': [0.0, 16.86866],
    'zone5_temperature': [12.664754, 30.622087],
    'zone6_humidity': [3.8307214, 74.27648],
    'zone6_people_count': [0.0, 10.870567],
    'zone6_temperature': [12.557228, 31.956497],
    'zone7_humidity': [3.8596928, 74.22123],
    'zone7_people_count': [0.0, 16.86829],
    'zone7_temperature': [12.642274, 28.245476],
    'zone8_humidity': [3.8424697, 73.86958],
    'zone8_people_count': [0.0, 10.870567],
    'zone8_temperature': [12.5359745, 31.856133],
    'zone9_humidity': [3.852498, 72.49596],
    'zone9_people_count': [0.0, 16.86866],
    'zone9_temperature': [11.540141, 31.43154]
}

RANGES_SHOP = {
    'Cooling_Setpoint_RL': [22.500103, 29.999979],
    'HVAC_electricity_demand_rate': [20.0, 24244.742],
    'Heating_Setpoint_RL': [15.000065, 22.49992],
    'abs_comfort': [0.0, 56.15920732802738],
    'clg_setpoint': [22.500103, 33.0],
    'comfort_penalty': [-56.15920732802738, -0.0],
    'day_of_month': [1.0, 31.0],
    'diffuse_solar_radiation': [0.0, 566.0],
    'direct_solar_radiation': [0.0, 943.0],
    'hour': [0.0, 23.0],
    'htg_setpoint': [13.0, 22.49992],
    'month': [1.0, 12.0],
    'outdoor_humidity': [11.0, 100.0],
    'outdoor_temperature': [-13.9, 36.7],
    'power_penalty': [-2.4244742973670714, -0.002],
    'reward': [-28.08060366401369, -0.001],
    'storage_battery_charge_state': [1442.5739, 8206.616],
    'storage_charge_energy': [0.0, 17039330.0],
    'storage_charge_power': [0.0, 18932.59],
    'storage_discharge_energy': [0.0, 17519884.0],
    'storage_discharge_power': [0.0, 26787.895],
    'storage_thermal_loss_energy': [0.00046225594, 1362685.2],
    'storage_thermal_loss_rate': [8.977186e-07, 4363.1284],
    'wind_direction': [0.0, 357.5],
    'wind_speed': [0.0, 17.5],
    'zone1_humidity': [4.6601014, 98.16939],
    'zone1_people_count': [0.0, 0.573705],
    'zone1_temperature': [8.25581, 29.887028],
    'zone2_humidity': [4.6608076, 97.824684],
    'zone2_people_count': [0.0, 0.44612],
    'zone2_temperature': [7.9952884, 29.953714],
    'zone3_humidity': [4.390302, 97.749565],
    'zone3_people_count': [0.0, 0.573705],
    'zone3_temperature': [8.27628, 29.967094],
    'zone4_humidity': [4.0773335, 97.9465],
    'zone4_people_count': [0.0, 0.44612],
    'zone4_temperature': [8.16939, 29.967093],
    'zone5_humidity': [4.3668585, 96.79506],
    'zone5_people_count': [0.0, 0.810445],
    'zone5_temperature': [11.081608, 29.691372]
}

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
