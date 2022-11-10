"""Constants used in whole project."""

import os

import gym
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
                'Zone Thermal Comfort Clothing Value(SPACE1-1 PEOPLE 1)': [0.0, 1.0],
                'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)': [0.0,
                                                                             98.37141259444684],
                'Zone Thermal Comfort Mean Radiant Temperature(SPACE1-1 PEOPLE 1)': [0.0,
                                                                                     35.98853496778508],
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
                'total_power_no_units': [-3.259557259261767, -0.0173658369273838]}


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

# ---------------------------------------------------------------------------- #
#                       Default Eplus Environments values                      #
# ---------------------------------------------------------------------------- #

# -------------------------------------5ZONE---------------------------------- #

DEFAULT_5ZONE_OBSERVATION_VARIABLES = [
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Site Outdoor Air Relative Humidity(Environment)',
    'Site Wind Speed(Environment)',
    'Site Wind Direction(Environment)',
    'Site Diffuse Solar Radiation Rate per Area(Environment)',
    'Site Direct Solar Radiation Rate per Area(Environment)',
    'Zone Thermostat Heating Setpoint Temperature(SPACE1-1)',
    'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)',
    'Zone Air Temperature(SPACE1-1)',
    'Zone Thermal Comfort Mean Radiant Temperature(SPACE1-1 PEOPLE 1)',
    'Zone Air Relative Humidity(SPACE1-1)',
    'Zone Thermal Comfort Clothing Value(SPACE1-1 PEOPLE 1)',
    'Zone Thermal Comfort Fanger Model PPD(SPACE1-1 PEOPLE 1)',
    'Zone People Occupant Count(SPACE1-1)',
    'People Air Temperature(SPACE1-1 PEOPLE 1)',
    'Facility Total HVAC Electricity Demand Rate(Whole Building)'
]

DEFAULT_5ZONE_ACTION_VARIABLES = [
    'Heating_Setpoint_RL',
    'Cooling_Setpoint_RL',
]

DEFAULT_5ZONE_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(DEFAULT_5ZONE_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32)

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
    low=np.array([15.0, 22.5]),
    high=np.array([22.5, 30.0]),
    shape=(2,),
    dtype=np.float32
)

DEFAULT_5ZONE_ACTION_DEFINITION = {
    'Htg-SetP-Sch': {'name': 'Heating_Setpoint_RL', 'initial_value': 21},
    'Clg-SetP-Sch': {'name': 'Cooling_Setpoint_RL', 'initial_value': 25},
}

# ----------------------------------DATACENTER--------------------------------- #
DEFAULT_DATACENTER_OBSERVATION_VARIABLES = [
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Site Outdoor Air Relative Humidity(Environment)',
    'Site Wind Speed(Environment)',
    'Site Wind Direction(Environment)',
    'Site Diffuse Solar Radiation Rate per Area(Environment)',
    'Site Direct Solar Radiation Rate per Area(Environment)',
    'Zone Thermostat Heating Setpoint Temperature(West Zone)',
    'Zone Thermostat Cooling Setpoint Temperature(West Zone)',
    'Zone Air Temperature(West Zone)',
    'Zone Thermal Comfort Mean Radiant Temperature(West Zone PEOPLE)',
    'Zone Air Relative Humidity(West Zone)',
    'Zone Thermal Comfort Clothing Value(West Zone PEOPLE)',
    'Zone Thermal Comfort Fanger Model PPD(West Zone PEOPLE)',
    'Zone People Occupant Count(West Zone)',
    'People Air Temperature(West Zone PEOPLE)',
    'Zone Thermostat Heating Setpoint Temperature(East Zone)',
    'Zone Thermostat Cooling Setpoint Temperature(East Zone)',
    'Zone Air Temperature(East Zone)',
    'Zone Thermal Comfort Mean Radiant Temperature(East Zone PEOPLE)',
    'Zone Air Relative Humidity(East Zone)',
    'Zone Thermal Comfort Clothing Value(East Zone PEOPLE)',
    'Zone Thermal Comfort Fanger Model PPD(East Zone PEOPLE)',
    'Zone People Occupant Count(East Zone)',
    'People Air Temperature(East Zone PEOPLE)',
    'Facility Total HVAC Electricity Demand Rate(Whole Building)'
]

DEFAULT_DATACENTER_ACTION_VARIABLES = [
    'Heating_Setpoint_RL',
    'Cooling_Setpoint_RL',
]

DEFAULT_DATACENTER_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(DEFAULT_DATACENTER_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32)

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
    low=np.array([15.0, 22.5]),
    high=np.array([22.5, 30.0]),
    shape=(2,),
    dtype=np.float32)

DEFAULT_DATACENTER_ACTION_DEFINITION = {
    'Heating Setpoints': {'name': 'Heating_Setpoint_RL', 'initial_value': 21},
    'Cooling Setpoints': {'name': 'Cooling_Setpoint_RL', 'initial_value': 25}
}

# ----------------------------------WAREHOUSE--------------------------------- #
DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES = [
    'Site Outdoor Air Drybulb Temperature(Environment)',
    'Site Outdoor Air Relative Humidity(Environment)',
    'Site Wind Speed(Environment)',
    'Site Wind Direction(Environment)',
    'Site Diffuse Solar Radiation Rate per Area(Environment)',
    'Site Direct Solar Radiation Rate per Area(Environment)',
    'Zone Thermostat Heating Setpoint Temperature(Zone1 Office)',
    'Zone Thermostat Cooling Setpoint Temperature(Zone1 Office)',
    'Zone Air Temperature(Zone1 Office)',
    'Zone Air Relative Humidity(Zone1 Office)',
    'Zone People Occupant Count(Zone1 Office)',
    'Zone Thermostat Heating Setpoint Temperature(Zone2 Fine Storage)',
    'Zone Thermostat Cooling Setpoint Temperature(Zone2 Fine Storage)',
    'Zone Air Temperature(Zone2 Fine Storage)',
    'Zone Air Relative Humidity(Zone2 Fine Storage)',
    'Zone Thermostat Heating Setpoint Temperature(Zone3 Bulk Storage)',
    'Zone Air Temperature(Zone3 Bulk Storage)',
    'Zone Air Relative Humidity(Zone3 Bulk Storage)',
    'Facility Total HVAC Electricity Demand Rate(Whole Building)']

DEFAULT_WAREHOUSE_ACTION_VARIABLES = [
    'Office_Heating_RL',
    'Office_Cooling_RL',
    'FineStorage_Heating_RL',
    'FineStorage_Cooling_RL',
    'BulkStorage_Heating_RL'

]

DEFAULT_WAREHOUSE_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32)

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
    low=np.array([15.0, 22.5, 15.0, 22.5, 15.0]),
    high=np.array([22.5, 30.0, 22.5, 30.0, 22.5]),
    shape=(5,),
    dtype=np.float32)

DEFAULT_WAREHOUSE_ACTION_DEFINITION = {
    'Office Heating Schedule': {
        'name': 'Office_Heating_RL',
        'initial_value': 21},
    'Office Cooling Schedule': {
        'name': 'Office_Cooling_RL',
        'initial_value': 25},
    'Fine Storage Heating Setpoint Schedule': {
        'name': 'FineStorage_Heating_RL',
                'initial_value': 21},
    'Fine Storage Cooling Setpoint Schedule': {
        'name': 'FineStorage_Cooling_RL',
        'initial_value': 25},
    'Bulk Storage Heating Setpoint Schedule': {
        'name': 'BulkStorage_Heating_RL',
        'initial_value': 21},
}

# ----------------------------------OFFICE--------------------------------- #

DEFAULT_OFFICE_OBSERVATION_VARIABLES = [
    'Zone Thermostat Heating Setpoint Temperature(Core_bottom)',
    'Zone Thermostat Cooling Setpoint Temperature(Core_bottom)',
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
    'Zone Air Temperature(Perimeter_mid_ZN_4)',
    'Facility Total HVAC Electricity Demand Rate(Whole Building)',
    'Site Outdoor Air Drybulb Temperature(Environment)'
]

DEFAULT_OFFICE_ACTION_VARIABLES = [
    'Office_Heating_RL',
    'Office_Cooling_RL'
]

DEFAULT_OFFICE_OBSERVATION_SPACE = gym.spaces.Box(
    low=-5e6,
    high=5e6,
    shape=(len(DEFAULT_OFFICE_OBSERVATION_VARIABLES) + 4,),
    dtype=np.float32)

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
    low=np.array([15.0, 22.5]),
    high=np.array([22.5, 30.0]),
    shape=(2,),
    dtype=np.float32)

DEFAULT_OFFICE_ACTION_DEFINITION = {
    'HTGSETP_SCH_YES_OPTIMUM': {
        'name': 'Office_Heating_RL',
        'initial_value': 21},
    'CLGSETP_SCH_YES_OPTIMUM': {
        'name': 'Office_Cooling_RL',
        'initial_value': 25}}

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
#     low=np.array([15.0, 22.5]),
#     high=np.array([22.5, 30.0]),
#     shape=(2,),
#     dtype=np.float32)

# DEFAULT_HOSPITAL_ACTION_DEFINITION = {
#     '': {'name': '', 'initial_value': 21},
#     '': {'name': '', 'initial_value': 25}
# }
