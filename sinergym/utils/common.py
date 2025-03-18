"""Common utilities."""

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import xlsxwriter
from eppy.modeleditor import IDF

try:
    from stable_baselines3.common.noise import NormalActionNoise
except ImportError:
    pass

import sinergym
from sinergym.utils.constants import LOG_COMMON_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.rewards import *

logger = TerminalLogger().getLogger(
    name='COMMON',
    level=LOG_COMMON_LEVEL)

# ---------------------------------------------------------------------------- #
#                                   WRAPPERS                                   #
# ---------------------------------------------------------------------------- #


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    Args:
        env (Type[gym.Env]): Environment to check
        wrapper_class (Type[gym.Wrapper]): Wrapper class to look for

    Returns:
        bool: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_wrapper(env: gym.Env,
                   wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a wrapper object by recursively searching.

    Args:
        env (gym.Env): Environment to unwrap
        wrapper_class (Type[gym.Wrapper]): Wrapper to look for

    Returns:
        Optional[gym.Wrapper]: Wrapper object if found, else None
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp.env
        env_tmp = env_tmp.env
    return None

# ---------------------------------------------------------------------------- #
#                               BUILDING MODELING                              #
# ---------------------------------------------------------------------------- #


def get_delta_seconds(
        st_year: int,
        st_mon: int,
        st_day: int,
        end_year: int,
        end_mon: int,
        end_day: int) -> float:
    """Returns the delta seconds between st year:st mon:st day:0:0:0 and
    end year:end mon:end day:24:0:0.

    Args:
        st_year (int): Start year.
        st_mon (int): Start month.
        st_day (int): Start day.
        end_year (int): End year.
        end_mon (int): End month.
        end_day (int): End day.

    Returns:
        float: Time difference in seconds.

    """
    start_time = datetime(st_year, st_mon, st_day)
    end_time = datetime(end_year, end_mon, end_day) + timedelta(days=1)
    return (end_time - start_time).total_seconds()


def eppy_element_to_dict(element: IDF) -> Dict[str, Dict[str, str]]:
    """Converts an eppy element into a dictionary following the EnergyPlus epJSON standard.

    Args:
        element (IDF): eppy element to be converted.

    Returns:
        Dict[str,Dict[str,str]]: Python dictionary with epJSON format of eppy element.
    """
    fields = {
        fieldname.lower().replace('drybulb', 'dry_bulb'):
        'WetBulb' if (value := element[fieldname]) == 'Wetbulb' else value
        for fieldname in element.fieldnames
        if fieldname not in {'Name', 'key'} and element[fieldname] != ''
    }

    return {getattr(element, 'Name', '').lower(): fields}


def export_schedulers_to_excel(
        schedulers: Dict[str, Dict[str, Union[str, Dict[str, str]]]], path: str) -> None:  # pragma: no cover
    """Exports scheduler information from a dictionary to an Excel file.

    Args:
        schedulers (Dict[str, Dict[str, Union[str, Dict[str, str]]]]): Dictionary with the correct format.
        path (str): Relative path where the Excel file will be created.
    """

    # Creating workbook and sheet
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()

    # Creating cell format configuration
    keys_format = workbook.add_format(
        {'bold': True, 'font_size': 20, 'align': 'center', 'bg_color': 'gray', 'border': True})
    cells_format = workbook.add_format({'align': 'center'})
    actuator_format = workbook.add_format(
        {'bold': True, 'align': 'center', 'bg_color': 'gray'})

    # Headers
    worksheet.write(0, 0, 'Name', keys_format)
    worksheet.write(0, 1, 'Type', keys_format)

    current_row = 1
    max_col = 1

    for key, info in schedulers.items():
        worksheet.write(current_row, 0, key, actuator_format)
        worksheet.write(
            current_row, 1, info.get(
                'Type', 'Unknown'), cells_format)

        col_offset = 2  # Offset inicial después de 'Type'
        for object_name, values in info.items():
            if isinstance(values, dict):
                worksheet.write(
                    current_row,
                    col_offset,
                    f'Name: {object_name}')
                worksheet.write(
                    current_row,
                    col_offset + 1,
                    f'Field: {
                        values.get(
                            "field_name",
                            "N/A")}')
                worksheet.write(
                    current_row,
                    col_offset + 2,
                    f'Table type: {
                        values.get(
                            "table_name",
                            "N/A")}')
                col_offset += 3  # Avanzar columnas según los datos escritos

        max_col = max(max_col, col_offset)
        current_row += 1

    # Adjusting column width
    worksheet.set_column(0, max_col, 40)

    # Add object columns
    for i, col in enumerate(range(2, max_col, 3), start=1):
        worksheet.merge_range(0, col, 0, col + 2, f'OBJECT {i}', keys_format)

    workbook.close()

# ---------------------------------------------------------------------------- #
#                          ORNSTEIN UHLENBECK PROCCESS                         #
# ---------------------------------------------------------------------------- #


def ornstein_uhlenbeck_process(
        data: pd.DataFrame,
        variability_config: Dict[str, Tuple[float, float, float]]) -> pd.DataFrame:
    """Add noise to the data using the Ornstein-Uhlenbeck process.

    Args:
        data (pd.DataFrame): Data to be modified.
        variability_config (Dict[str, Tuple[float, float, float]]): Dictionary with the variability configuration for each variable (sigma, mu and tau constants).

    Returns:
        pd.DataFrame: Data with noise added.
    """

    # deepcopy for weather_data
    data_mod = deepcopy(data)

    # Total time.
    T = 1.
    # get first column of df
    n = data_mod.shape[0]
    # dt = T / n  # tau defined as percentage of epw
    dt = T / 1.0  # tau defined as epw rows (hours)
    # t = np.linspace(0., T, n)  # Vector of times.

    # Sigma = standard deviation.
    # Mu = mean.
    # Tau = time constant.

    for variable, (sigma, mu, tau) in variability_config.items():
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        # Create noise
        noise = np.zeros(n)
        for i in range(n - 1):
            noise[i + 1] = noise[i] + dt * (-(noise[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()

        # Add noise
        data_mod[variable] += noise

    return data_mod

# ---------------------------------------------------------------------------- #
#                    READING YAML ENVIRONMENT CONFIGURATION                    #
# ---------------------------------------------------------------------------- #


def parse_variables_settings(
        variables: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Read variables dictionary (from Sinergym yaml settings) and adapt it to the
       EnergyPlus format. More information about Sinergym YAML configuration format
       in documentation.

    Args:
        variables (Dict[str, Any]): Dictionary from Sinergym YAML configuration with variables information.

    Returns:
        Dict[str, Tuple[str, str]]: Dictionary with variables information in EnergyPlus API format.
    """

    output = {}

    for variable, specification in variables.items():

        if isinstance(specification['variable_names'], str):

            if isinstance(specification['keys'], str):
                output[specification['variable_names']] = (
                    variable, specification['keys'])

            elif isinstance(specification['keys'], list):
                for key in specification['keys']:
                    prename = key.lower()
                    prename = prename.replace(' ', '_')
                    prename = prename + '_'
                    output[prename +
                           specification['variable_names']] = (variable, key)

            else:
                raise RuntimeError

        elif isinstance(specification['variable_names'], list):

            if isinstance(specification['keys'], str):
                raise RuntimeError

            elif isinstance(specification['keys'], list):
                if len(specification['variable_names']) != len(
                        specification['keys']):
                    logger.error(
                        f'variable names and keys must have the same len in {variable}')
                    raise ValueError
                for variable_name, key_name in list(
                        zip(specification['variable_names'], specification['keys'])):
                    output[variable_name] = (variable, key_name)

            else:
                raise RuntimeError

        else:

            raise RuntimeError

    return output


def parse_meters_settings(meters: Dict[str, str]) -> Dict[str, str]:
    """Read meters dictionary (from Sinergym YAML settings) and adapt it to the
       EnergyPlus format. More information about Sinergym YAML configuration format
       in documentation.

    Args:
        meters (Dict[str, str]): Dictionary from Sinergym YAML configuration with meters information.

    Returns:
        Dict[str, str]: Dictionary with meters information in EnergyPlus API format.
    """

    output = {}

    for meter_name, variable_name in meters.items():
        output[variable_name] = meter_name

    return output


def parse_actuators_settings(
        actuators: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[str, str, str]]:
    """Read actuators dictionary (from Sinergym YAML settings) and adapt it to the
       EnergyPlus format. More information about Sinergym YAML configuration format
       in documentation.

    Args:
        actuators (Dict[str, Dict[str, str]]): Dictionary from Sinergym YAML configuration with actuators information.

    Returns:
        Dict[str, Tuple[str, str, str]]: Dictionary with actuators information in EnergyPlus API format.
    """

    output = {}

    for actuator_name, specification in actuators.items():
        output[specification['variable_name']] = (
            specification['element_type'], specification['value_type'], actuator_name)

    return output


def convert_conf_to_env_parameters(
        conf: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a configuration from YAML format (sinergym/data/default_configuration/file.yaml) in a dictionary of all possible environments as dictionary with id as key and env_kwargs as value.
       More information about Sinergym environment configuration in YAML format in documentation.

    Args:
        conf (Dict[str, Any]): Dictionary from read YAML setting file (sinergym/data/default_configuration/file.yaml).

    Returns:
        Dict[str,[Dict[str, Any]]: All possible Sinergym environment constructor kwargs.
    """

    configurations = {}

    variables = parse_variables_settings(conf['variables'])
    meters = parse_meters_settings(conf['meters'])
    actuators = parse_actuators_settings(conf['actuators'])
    context = parse_actuators_settings(conf['context'])

    if len(conf['weather_specification']['weather_files']) != len(
            conf['weather_specification']['keys']):
        logger.error(
            'Weather files and id keys must have the same len\n' f'{
                len(
                    conf["weather_specification"]["weather_files"])} weather files != {
                len(
                    conf["weather_specification"]["keys"])} keys')
        raise ValueError

    weather_info = list(zip(conf['weather_specification']['keys'],
                        conf['weather_specification']['weather_files']))

    weather_variability = conf.get('weather_variability')

    for weather_id, weather_file in weather_info:

        id = 'Eplus-' + conf['id_base'] + '-' + weather_id + '-continuous-v1'

        env_kwargs = {
            'building_file': conf['building_file'],
            'weather_files': weather_file,
            'action_space': eval(conf['action_space']),
            'time_variables': conf['time_variables'],
            'variables': variables,
            'meters': meters,
            'actuators': actuators,
            'context': context,
            'initial_context': conf.get('initial_context'),
            'reward': eval(conf['reward']),
            'reward_kwargs': conf['reward_kwargs'],
            'max_ep_data_store_num': conf['max_ep_data_store_num'],
            'env_name': id,
            'config_params': conf.get('config_params')
        }
        configurations[id] = env_kwargs

        if weather_variability:

            # Cast weather variability variation from list to tuple
            weather_variability = {
                var_name: tuple(
                    tuple(param) if isinstance(param, list) else param
                    for param in var_params
                )
                for var_name, var_params in weather_variability.items()
            }

            id = 'Eplus-' + conf['id_base'] + '-' + \
                weather_id + '-continuous-stochastic-v1'
            env_kwargs = {
                'building_file': conf['building_file'],
                'weather_files': weather_file,
                'action_space': eval(conf['action_space']),
                'time_variables': conf['time_variables'],
                'variables': variables,
                'meters': meters,
                'actuators': actuators,
                'context': context,
                'initial_context': conf.get('initial_context'),
                'weather_variability': weather_variability,
                'reward': eval(conf['reward']),
                'reward_kwargs': conf['reward_kwargs'],
                'max_ep_data_store_num': conf['max_ep_data_store_num'],
                'env_name': id,
                'config_params': conf.get('config_params')
            }
            configurations[id] = env_kwargs

    return configurations

# ---------------------------------------------------------------------------- #
#                          Process YAML configurations                         #
# ---------------------------------------------------------------------------- #


def process_environment_parameters(env_params: dict) -> dict:  # pragma: no cover
    # Transform required str's into Callables or lists in tuples
    if env_params.get('action_space'):
        env_params['action_space'] = eval(
            env_params['action_space'])

    if env_params.get('variables'):
        for variable_name, components in env_params['variables'].items():
            env_params['variables'][variable_name] = tuple(components)

    if env_params.get('actuators'):
        for actuator_name, components in env_params['actuators'].items():
            env_params['actuators'][actuator_name] = tuple(components)

    if env_params.get('weather_variability'):
        env_params['weather_variability'] = {
            var_name: tuple(
                tuple(param) if isinstance(param, list) else param
                for param in var_params
            )
            for var_name, var_params in env_params['weather_variability'].items()
        }

    if env_params.get('reward'):
        env_params['reward'] = eval(env_params['reward'])

    if env_params.get('reward_kwargs'):
        for reward_param_name, reward_value in env_params.items():
            if reward_param_name in [
                'range_comfort_winter',
                'range_comfort_summer',
                'summer_start',
                    'summer_final']:
                env_params['reward_kwargs'][reward_param_name] = tuple(
                    reward_value)

    if env_params.get('config_params'):
        if env_params['config_params'].get('runperiod'):
            env_params['config_params']['runperiod'] = tuple(
                env_params['config_params']['runperiod'])
    # Add more keys if needed...

    return env_params


def process_algorithm_parameters(alg_params: dict):  # pragma: no cover

    # Transform required str's into Callables or list in tuples
    if alg_params.get('train_freq') and isinstance(
            alg_params.get('train_freq'), list):
        alg_params['train_freq'] = tuple(alg_params['train_freq'])

    if alg_params.get('action_noise'):
        alg_params['action_noise'] = eval(alg_params['action_noise'])
    # Add more keys if needed...

    return alg_params

 # --------------------- Functions that are no longer used -------------------- #

# def ranges_getter(output_path: str,
#                   last_result: Optional[Dict[str, List[float]]] = None
#                   ) -> Dict[str, List[float]]:  # pragma: no cover
#     """Given a path with simulations outputs, this function is used to extract max and min absolute values of all episodes in each variable. If a dict ranges is given, will be updated.

#     Args:
#         output_path (str): Path with simulation output (Eplus-env-<env_name>).
# last_result (Optional[Dict[str, List[float]]], optional): Last ranges
# dict to be updated. This will be created if it is not given.

#     Returns:
#         Dict[str, List[float]]: list min,max of each variable as a key.

#     """

#     if last_result is not None:
#         result = last_result
#     else:
#         result = {}

#     content = os.listdir(output_path)
#     for episode_path in content:
#         if os.path.isdir(
#             output_path +
#             '/' +
#                 episode_path) and episode_path.startswith('Eplus-env'):
#             simulation_content = os.listdir(output_path + '/' + episode_path)

#             if os.path.isdir(
#                 output_path +
#                 '/' +
#                     episode_path):
#                 monitor_path = output_path + '/' + episode_path + '/monitor.csv'
#                 print('Reading ' + monitor_path + ' limits.')
#                 data = pd.read_csv(monitor_path)

#                 if len(result) == 0:
#                     for column in data:
#                         # variable : [min,max]
#                         result[column] = [np.inf, -np.inf]

#                 for column in data:
#                     if np.min(data[column]) < result[column][0]:
#                         result[column][0] = np.min(data[column])
#                     if np.max(data[column]) > result[column][1]:
#                         result[column][1] = np.max(data[column])
#     return result
