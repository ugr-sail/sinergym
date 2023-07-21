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
RANGES_5ZONE = {'Facility Total HVAC Electricity Demand Rate(Whole Building)': [173.6583692738386,
                                                                                32595.57259261767],
                'People Air Temperature(SPACE1-1 PEOPLE 1)': [0.0, 30.00826655379267],
                'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 588.0],
                'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 1033.0],
                'Site Outdoor Air Drybulb Temperature(Environment)': [-31.05437255409474,
                                                                      60.72839186915495],
                'Site Outdoor Air Relative Humidity(Environment)': [3.0, 100.0],
                'Site Wind Direction(Environment)': [0.0, 357.5],
                'Site Wind Speed(Environment)': [0.0, 23.1],
                'Space1-ClgSetP-RL': [21.0, 30.0],
                'Space1-HtgSetP-RL': [15.0, 22.49999],
                'Zone Air Relative Humidity(SPACE1-1)': [3.287277410867238,
                                                         87.60662171287048],
                'Zone Air Temperature(SPACE1-1)': [15.22565264653451, 30.00826655379267],
                'Zone People Occupant Count(SPACE1-1)': [0.0, 11.0],
                'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)': [21.0, 30.0],
                'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)': [15.0,
                                                                           22.49999046325684],
                'comfort_penalty': [-6.508266553792669, -0.0],
                'day': [1, 31],
                'done': [False, True],
                'hour': [0, 23],
                'month': [1, 12],
                'year': [1, 2022],
                'reward': [-3.550779087370951, -0.0086829184636919],
                'time (seconds)': [0, 31536000],
                'timestep': [0, 35040],
                'total_power_no_units': [-3.259557259261767, -0.0173658369273838],
                # Added previous variables:
                'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)_previous': [21.0, 30.0],
                'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)_previous': [15.0,
                                                                                    22.49999046325684],
                'Zone Air Temperature(SPACE1-1)_previous': [15.22565264653451, 30.00826655379267]}


RANGES_DATACENTER = {
    'East-ClgSetP-RL': [21.0, 30.0],
    'East-HtgSetP-RL': [15.0, 22.499973],
    'Facility Total HVAC Electricity Demand Rate(Whole Building)': [1763.7415,
                                                                    76803.016],
    'People Air Temperature(East Zone PEOPLE)': [0.0, 30.279287],
    'People Air Temperature(West Zone PEOPLE)': [0.0, 30.260946],
    'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 588.0],
    'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 1033.0],
    'Site Outdoor Air Drybulb Temperature(Environment)': [-16.049532, 42.0],
    'Site Outdoor Air Relative Humidity(Environment)': [3.0, 100.0],
    'Site Wind Direction(Environment)': [0.0, 357.5],
    'Site Wind Speed(Environment)': [0.0, 17.5],
    'West-ClgSetP-RL': [21.0, 30.0],
    'West-HtgSetP-RL': [15.0, 22.499998],
    'Zone Air Relative Humidity(East Zone)': [1.8851701, 67.184616],
    'Zone Air Relative Humidity(West Zone)': [1.8945858, 66.7946],
    'Zone Air Temperature(East Zone)': [21.003511, 30.279287],
    'Zone Air Temperature(West Zone)': [21.004263, 30.260946],
    'Zone People Occupant Count(East Zone)': [0.0, 7.0],
    'Zone People Occupant Count(West Zone)': [0.0, 11.0],
    'Zone Thermal Comfort Clothing Value(East Zone PEOPLE)': [0.0, 0.0],
    'Zone Thermal Comfort Clothing Value(West Zone PEOPLE)': [0.0, 0.0],
    'Zone Thermal Comfort Fanger Model PPD(East Zone PEOPLE)': [0.0, 66.75793],
    'Zone Thermal Comfort Fanger Model PPD(West Zone PEOPLE)': [0.0, 59.53962],
    'Zone Thermal Comfort Mean Radiant Temperature(East Zone PEOPLE)': [0.0,
                                                                        29.321169],
    'Zone Thermal Comfort Mean Radiant Temperature(West Zone PEOPLE)': [0.0,
                                                                        29.04933],
    'Zone Thermostat Cooling Setpoint Temperature(East Zone)': [21.0, 30.0],
    'Zone Thermostat Cooling Setpoint Temperature(West Zone)': [21.0, 30.0],
    'Zone Thermostat Heating Setpoint Temperature(East Zone)': [15.0, 22.499973],
    'Zone Thermostat Heating Setpoint Temperature(West Zone)': [15.0, 22.499998],
    'comfort_penalty': [-13.264959140712048, -0.0],
    'day': [1.0, 31.0],
    'done': [False, True],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'year': [1.0, 2022.0],
    'power_penalty': [-7.68030164869835, -0.1763741508343818],
    'reward': [-9.090902680780722, -0.0881870754171909],
    'time (seconds)': [0, 31536000],
    'timestep': [0, 35040]
}

RANGES_OFFICE = {
    'Facility Total HVAC Electricity Demand Rate(Whole Building)': [168.34105,
                                                                    126668.15],
    'Office_Cooling_RL': [22.500004, 29.99999],
    'Office_Heating_RL': [15.000012, 22.499994],
    'Site Outdoor Air Drybulb Temperature(Environment)': [-1.1, 42.0],
    'Zone Air Temperature(Core_bottom)': [20.70919, 27.480345],
    'Zone Air Temperature(Core_mid)': [21.514507, 26.98882],
    'Zone Air Temperature(Core_top)': [20.49858, 27.10577],
    'Zone Air Temperature(FirstFloor_Plenum)': [20.968946, 27.27035],
    'Zone Air Temperature(MidFloor_Plenum)': [21.107718, 27.317183],
    'Zone Air Temperature(Perimeter_bot_ZN_1)': [15.171554, 30.35336],
    'Zone Air Temperature(Perimeter_bot_ZN_2)': [18.939316, 30.110523],
    'Zone Air Temperature(Perimeter_bot_ZN_3)': [19.154911, 27.721031],
    'Zone Air Temperature(Perimeter_bot_ZN_4)': [18.955957, 31.069431],
    'Zone Air Temperature(Perimeter_mid_ZN_1)': [19.942469, 31.205746],
    'Zone Air Temperature(Perimeter_mid_ZN_2)': [19.25707, 30.427061],
    'Zone Air Temperature(Perimeter_mid_ZN_3)': [19.493652, 28.486017],
    'Zone Air Temperature(Perimeter_mid_ZN_4)': [19.37576, 31.639992],
    'Zone Air Temperature(Perimeter_top_ZN_1)': [19.346384, 31.139534],
    'Zone Air Temperature(Perimeter_top_ZN_2)': [18.630365, 30.676693],
    'Zone Air Temperature(Perimeter_top_ZN_3)': [18.760962, 29.120058],
    'Zone Air Temperature(Perimeter_top_ZN_4)': [18.678907, 32.390034],
    'Zone Air Temperature(TopFloor_Plenum)': [18.011152, 30.430445],
    'Zone Thermostat Cooling Setpoint Temperature(Core_bottom)': [22.500004,
                                                                  29.99999],
    'Zone Thermostat Heating Setpoint Temperature(Core_bottom)': [15.000012,
                                                                  22.499994],
    'abs_comfort': [0.0, 25.367293089923947],
    'comfort_penalty': [-25.367293089923947, -0.0],
    'day': [1.0, 31.0],
    'done': [False, True],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'power_penalty': [-12.66681481600682, -0.0168341048642297],
    'reward': [-12.69354202988205, -0.0084170524321148],
    'time (seconds)': [0, 31536000],
    'timestep': [0, 35040],
    'year': [1991.0, 1992.0]
}

RANGES_WAREHOUSE = {
    'BulkStorage_Heating_RL': [15.000002, 22.499958],
    'Facility Total HVAC Electricity Demand Rate(Whole Building)': [481.0,
                                                                    12384.411],
    'FineStorage_Cooling_RL': [22.500015, 29.999979],
    'FineStorage_Heating_RL': [15.000049, 22.499994],
    'Office_Cooling_RL': [22.500008, 29.999989],
    'Office_Heating_RL': [15.000006, 22.49995],
    'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 588.0],
    'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 1033.0],
    'Site Outdoor Air Drybulb Temperature(Environment)': [-1.1, 42.0],
    'Site Outdoor Air Relative Humidity(Environment)': [3.0, 100.0],
    'Site Wind Direction(Environment)': [0.0, 357.5],
    'Site Wind Speed(Environment)': [0.0, 13.9],
    'Zone Air Relative Humidity(Zone1 Office)': [6.253238, 100.0],
    'Zone Air Relative Humidity(Zone2 Fine Storage)': [4.621931, 100.0],
    'Zone Air Relative Humidity(Zone3 Bulk Storage)': [6.0304756, 95.72191],
    'Zone Air Temperature(Zone1 Office)': [15.067171, 29.446217],
    'Zone Air Temperature(Zone2 Fine Storage)': [13.976028, 29.089006],
    'Zone Air Temperature(Zone3 Bulk Storage)': [15.5426655, 34.394325],
    'Zone People Occupant Count(Zone1 Office)': [0.0, 5.0],
    'Zone Thermostat Cooling Setpoint Temperature(Zone1 Office)': [22.500008,
                                                                   29.999989],
    'Zone Thermostat Cooling Setpoint Temperature(Zone2 Fine Storage)': [22.500015,
                                                                         29.999979],
    'Zone Thermostat Heating Setpoint Temperature(Zone1 Office)': [15.000006,
                                                                   22.49995],
    'Zone Thermostat Heating Setpoint Temperature(Zone2 Fine Storage)': [15.000049,
                                                                         22.499994],
    'Zone Thermostat Heating Setpoint Temperature(Zone3 Bulk Storage)': [15.000002,
                                                                         22.499958],
    'abs_comfort': [0.0, 9.527248547862186],
    'comfort_penalty': [-9.527248547862186, -0.0],
    'day': [1.0, 31.0],
    'done': [False, True],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'power_penalty': [-1.2384410833362391, -0.0480999999999999],
    'reward': [-5.245662269856689, -0.0240499999999999],
    'time (seconds)': [0, 31536000],
    'timestep': [0, 35040],
    'year': [1991.0, 1992.0]
}

RANGES_OFFICEGRID = {'Charge_Rate_RL': [4.172325e-07, 0.9999925],
                     'Cooling_Setpoint_RL': [22.500011, 29.999977],
                     'Discharge_Rate_RL': [1.5497208e-06, 0.9999977],
                     'Electric Storage Simple Charge State(Battery)': [400000000000.0,
                                                                       500000000000.0],
                     'Facility Total HVAC Electricity Demand Rate(Whole Building)': [21.752993,
                                                                                     841044.94],
                     'Heating_Setpoint_RL': [15.000034, 22.499985],
                     'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 588.0],
                     'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 1033.0],
                     'Site Outdoor Air Drybulb Temperature(Environment)': [-1.1, 42.0],
                     'Site Outdoor Air Relative Humidity(Environment)': [3.0, 100.0],
                     'Site Wind Direction(Environment)': [0.0, 357.5],
                     'Site Wind Speed(Environment)': [0.0, 13.9],
                     'Zone Air Relative Humidity(Basement)': [5.7947288, 56.715904],
                     'Zone Air Relative Humidity(GroundFloor_Plenum)': [4.2359853, 58.988487],
                     'Zone Air Relative Humidity(MidFloor_Plenum)': [4.12756, 56.781883],
                     'Zone Air Relative Humidity(Perimeter_bot_ZN_1)': [2.834079, 70.3728],
                     'Zone Air Relative Humidity(Perimeter_bot_ZN_2)': [3.7915392, 73.45705],
                     'Zone Air Relative Humidity(Perimeter_bot_ZN_3)': [4.02541, 71.029976],
                     'Zone Air Relative Humidity(Perimeter_bot_ZN_4)': [3.2505984, 72.91185],
                     'Zone Air Relative Humidity(Perimeter_mid_ZN_1)': [2.497092, 68.21089],
                     'Zone Air Relative Humidity(Perimeter_mid_ZN_2)': [3.8115575, 72.54777],
                     'Zone Air Relative Humidity(Perimeter_mid_ZN_3)': [4.0045953, 69.21024],
                     'Zone Air Relative Humidity(Perimeter_mid_ZN_4)': [3.373578, 72.00627],
                     'Zone Air Relative Humidity(Perimeter_top_ZN_1)': [2.72939, 73.03404],
                     'Zone Air Relative Humidity(Perimeter_top_ZN_2)': [3.8250444, 76.22068],
                     'Zone Air Relative Humidity(Perimeter_top_ZN_3)': [3.977496, 74.55403],
                     'Zone Air Relative Humidity(Perimeter_top_ZN_4)': [3.4795659, 75.39192],
                     'Zone Air Relative Humidity(TopFloor_Plenum)': [4.359065, 78.65317],
                     'Zone Air Relative Humidity(core_bottom)': [4.4263396, 57.799427],
                     'Zone Air Relative Humidity(core_mid)': [4.330785, 56.90424],
                     'Zone Air Relative Humidity(core_top)': [4.434338, 62.41857],
                     'Zone Air Temperature(Basement)': [20.717112, 25.916859],
                     'Zone Air Temperature(GroundFloor_Plenum)': [20.279613, 26.802593],
                     'Zone Air Temperature(MidFloor_Plenum)': [20.11584, 27.07084],
                     'Zone Air Temperature(Perimeter_bot_ZN_1)': [18.225899, 31.426455],
                     'Zone Air Temperature(Perimeter_bot_ZN_2)': [17.352448, 31.083334],
                     'Zone Air Temperature(Perimeter_bot_ZN_3)': [17.537117, 28.363703],
                     'Zone Air Temperature(Perimeter_bot_ZN_4)': [17.152567, 32.238426],
                     'Zone Air Temperature(Perimeter_mid_ZN_1)': [17.900778, 32.82463],
                     'Zone Air Temperature(Perimeter_mid_ZN_2)': [16.500084, 31.902704],
                     'Zone Air Temperature(Perimeter_mid_ZN_3)': [16.987373, 29.222708],
                     'Zone Air Temperature(Perimeter_mid_ZN_4)': [16.419077, 33.204777],
                     'Zone Air Temperature(Perimeter_top_ZN_1)': [16.283703, 32.030056],
                     'Zone Air Temperature(Perimeter_top_ZN_2)': [15.7413025, 31.839926],
                     'Zone Air Temperature(Perimeter_top_ZN_3)': [15.905179, 30.020973],
                     'Zone Air Temperature(Perimeter_top_ZN_4)': [15.655823, 33.591072],
                     'Zone Air Temperature(TopFloor_Plenum)': [13.977393, 31.506895],
                     'Zone Air Temperature(core_bottom)': [20.273703, 27.046078],
                     'Zone Air Temperature(core_mid)': [20.271896, 27.318834],
                     'Zone Air Temperature(core_top)': [18.143255, 28.293093],
                     'Zone People Occupant Count(Basement)': [0.0, 91.09125],
                     'Zone People Occupant Count(Perimeter_bot_ZN_1)': [0.0, 16.025227],
                     'Zone People Occupant Count(Perimeter_bot_ZN_2)': [0.0, 10.32704],
                     'Zone People Occupant Count(Perimeter_bot_ZN_3)': [0.0, 16.024876],
                     'Zone People Occupant Count(Perimeter_bot_ZN_4)': [0.0, 10.32704],
                     'Zone People Occupant Count(Perimeter_mid_ZN_1)': [0.0, 16.025227],
                     'Zone People Occupant Count(Perimeter_mid_ZN_2)': [0.0, 10.32704],
                     'Zone People Occupant Count(Perimeter_mid_ZN_3)': [0.0, 16.024876],
                     'Zone People Occupant Count(Perimeter_mid_ZN_4)': [0.0, 10.32704],
                     'Zone People Occupant Count(Perimeter_top_ZN_1)': [0.0, 16.025227],
                     'Zone People Occupant Count(Perimeter_top_ZN_2)': [0.0, 10.32704],
                     'Zone People Occupant Count(Perimeter_top_ZN_3)': [0.0, 16.024876],
                     'Zone People Occupant Count(Perimeter_top_ZN_4)': [0.0, 10.32704],
                     'Zone People Occupant Count(core_bottom)': [0.0, 129.47832],
                     'Zone People Occupant Count(core_mid)': [0.0, 129.47832],
                     'Zone People Occupant Count(core_top)': [0.0, 129.47832],
                     'Zone Thermostat Cooling Setpoint Temperature(Basement)': [22.500011,
                                                                                29.999977],
                     'Zone Thermostat Heating Setpoint Temperature(Basement)': [15.000034,
                                                                                22.499985],
                     'abs_comfort': [0.0, 71.6801302145776],
                     'comfort_penalty': [-71.6801302145776, -0.0],
                     'day': [1.0, 31.0],
                     'done': [False, True],
                     'hour': [0.0, 23.0],
                     'month': [1.0, 12.0],
                     'power_penalty': [-84.10449604920616, -0.0021752993221097],
                     'reward': [-46.28584879011768, -0.0010878016155496],
                     'time (seconds)': [0, 31536000],
                     'timestep': [0, 35040],
                     'year': [1991.0, 1992.0]}

RANGES_SHOP = {'Cooling_Setpoint_RL': [22.500051, 29.999975],
               'Electric Storage Battery Charge State(Kibam)': [1445.8286, 8271.093],
               'Electric Storage Charge Energy(Kibam)': [0.0, 29168426.0],
               'Electric Storage Charge Power(Kibam)': [0.0, 32409.363],
               'Electric Storage Discharge Energy(Kibam)': [0.0, 27021258.0],
               'Electric Storage Discharge Power(Kibam)': [0.0, 30023.621],
               'Electric Storage Thermal Loss Energy(Kibam)': [3.5280268e-06, 3713192.2],
               'Electric Storage Thermal Loss Rate(Kibam)': [3.9200296e-09, 4125.769],
               'Facility Total HVAC Electricity Demand Rate(Whole Building)': [20.0,
                                                                               25360.148],
               'Heating_Setpoint_RL': [15.000007, 22.499907],
               'Site Diffuse Solar Radiation Rate per Area(Environment)': [0.0, 459.0],
               'Site Direct Solar Radiation Rate per Area(Environment)': [0.0, 880.0],
               'Site Outdoor Air Drybulb Temperature(Environment)': [-6.0, 30.0],
               'Site Outdoor Air Relative Humidity(Environment)': [20.0, 100.0],
               'Site Wind Direction(Environment)': [0.0, 357.5],
               'Site Wind Speed(Environment)': [0.0, 10.8],
               'Zone Air Relative Humidity(ZN_1_FLR_1_SEC_1)': [11.591304, 73.92276],
               'Zone Air Relative Humidity(ZN_1_FLR_1_SEC_2)': [9.241573, 71.74454],
               'Zone Air Relative Humidity(ZN_1_FLR_1_SEC_3)': [9.275727, 71.79277],
               'Zone Air Relative Humidity(ZN_1_FLR_1_SEC_4)': [10.065296, 72.80681],
               'Zone Air Relative Humidity(ZN_1_FLR_1_SEC_5)': [9.850535, 71.36196],
               'Zone Air Temperature(ZN_1_FLR_1_SEC_1)': [11.983476, 28.926315],
               'Zone Air Temperature(ZN_1_FLR_1_SEC_2)': [14.000444, 29.523012],
               'Zone Air Temperature(ZN_1_FLR_1_SEC_3)': [14.188374, 29.621378],
               'Zone Air Temperature(ZN_1_FLR_1_SEC_4)': [13.587406, 28.7227],
               'Zone Air Temperature(ZN_1_FLR_1_SEC_5)': [14.719373, 28.520697],
               'Zone People Occupant Count(ZN_1_FLR_1_SEC_1)': [0.0, 0.573705],
               'Zone People Occupant Count(ZN_1_FLR_1_SEC_2)': [0.0, 0.44612],
               'Zone People Occupant Count(ZN_1_FLR_1_SEC_3)': [0.0, 0.573705],
               'Zone People Occupant Count(ZN_1_FLR_1_SEC_4)': [0.0, 0.44612],
               'Zone People Occupant Count(ZN_1_FLR_1_SEC_5)': [0.0, 0.810445],
               'Zone Thermostat Cooling Setpoint Temperature(ZN_1_FLR_1_SEC_5)': [22.500051,
                                                                                  29.999975],
               'Zone Thermostat Heating Setpoint Temperature(ZN_1_FLR_1_SEC_5)': [15.000007,
                                                                                  22.499907],
               'abs_comfort': [0.0, 32.76872646263317],
               'comfort_penalty': [-32.76872646263317, -0.0],
               'day': [1.0, 31.0],
               'done': [False, True],
               'hour': [0.0, 23.0],
               'month': [1.0, 12.0],
               'power_penalty': [-2.5360149114627824, -0.002],
               'reward': [-16.633018306008985, -0.001],
               'time (seconds)': [0, 31536000],
               'timestep': [0, 35040],
               'year': [1991.0, 1992.0]}

# ---------------------------------------------------------------------------- #
#                       Default Eplus Environments values                      #
# ---------------------------------------------------------------------------- #

# ---------------------------------- GENERAL --------------------------------- #
DEFAULT_TIME_VARIABLES = ['year', 'month', 'day_of_month', 'hour']

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
        'Whole Building')}

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
