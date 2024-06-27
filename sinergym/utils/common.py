"""Common utilities."""

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import xlsxwriter
from eppy.modeleditor import IDF
from opyplus.epgm.record import Record

import sinergym
from sinergym.utils.constants import YEAR
from sinergym.utils.rewards import *

# --------------------- Sinergym environment information --------------------- #


def get_ids(start='Eplus') -> List[str]:
    """
    Returns a list of environment IDs created by Sinergym (starting by Eplus).

    Parameters:
        start (str): The prefix to filter the environment IDs. Defaults to 'Eplus'.

    Returns:
        List[str]: A list of Sinergym environment IDs.

    """
    envs_id = [env_id for env_id in gym.envs.registration.registry.keys()
               if env_id.startswith(start)]
    return envs_id

# --------------------------------- Wrappers --------------------------------- #


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def unwrap_wrapper(env: gym.Env,
                   wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp.env
        env_tmp = env_tmp.env
    return None

# ----------------------------- Building modeling ---------------------------- #


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
    startTime = datetime(st_year, st_mon, st_day, 0, 0, 0)
    endTime = datetime(end_year, end_mon, end_day,
                       23, 0, 0) + timedelta(0, 3600)
    delta_sec = (endTime - startTime).total_seconds()
    return delta_sec


def ranges_getter(output_path: str,
                  last_result: Optional[Dict[str, List[float]]] = None
                  ) -> Dict[str, List[float]]:  # pragma: no cover
    """Given a path with simulations outputs, this function is used to extract max and min absolute values of all episodes in each variable. If a dict ranges is given, will be updated.

    Args:
        output_path (str): Path with simulation output (Eplus-env-<env_name>).
        last_result (Optional[Dict[str, List[float]]], optional): Last ranges dict to be updated. This will be created if it is not given.

    Returns:
        Dict[str, List[float]]: list min,max of each variable as a key.

    """

    if last_result is not None:
        result = last_result
    else:
        result = {}

    content = os.listdir(output_path)
    for episode_path in content:
        if os.path.isdir(
            output_path +
            '/' +
                episode_path) and episode_path.startswith('Eplus-env'):
            simulation_content = os.listdir(output_path + '/' + episode_path)

            if os.path.isdir(
                output_path +
                '/' +
                    episode_path):
                monitor_path = output_path + '/' + episode_path + '/monitor.csv'
                print('Reading ' + monitor_path + ' limits.')
                data = pd.read_csv(monitor_path)

                if len(result) == 0:
                    for column in data:
                        # variable : [min,max]
                        result[column] = [np.inf, -np.inf]

                for column in data:
                    if np.min(data[column]) < result[column][0]:
                        result[column][0] = np.min(data[column])
                    if np.max(data[column]) > result[column][1]:
                        result[column][1] = np.max(data[column])
    return result


def get_record_keys(record: Record) -> List[str]:
    """Given an opyplus Epm Record (one element from opyplus.epm object) this function returns list of keys (opyplus hasn't got this functionality explicitly)

     Args:
        record (opyplus.Epm.Record): Element from Epm object.

     Returns:
        List[str]: Key list from record.
    """
    return [field.ref for field in record._table._dev_descriptor._field_descriptors]


def eppy_element_to_dict(element: IDF) -> Dict[str, Dict[str, str]]:
    """Given a eppy element, this function will create a dictionary using the name as key and the rest of fields as value. Following de EnergyPlus epJSON standard.

    Args:
        element (IDF): eppy element to be converted.

    Returns:
        Dict[str,Dict[str,str]]: Python dictionary with epJSON format of eppy element.
    """
    fields = {}
    for fieldname in element.fieldnames:
        fieldname_fixed = fieldname.lower().replace(
            'drybulb', 'dry_bulb')
        if fieldname != 'Name' and fieldname != 'key':
            if element[fieldname] != '':
                if element[fieldname] == 'Wetbulb':
                    fields[fieldname_fixed] = 'WetBulb'
                else:
                    fields[fieldname_fixed] = element[fieldname]
    return {element.Name.lower(): fields}


def export_schedulers_to_excel(
        schedulers: Dict[str, Dict[str, Union[str, Dict[str, str]]]], path: str) -> None:  # pragma: no cover
    """Given a python dictionary with schedulers from modeling format, this method export that information in a excel file

    Args:
        schedulers (Dict[str, Dict[str, Union[str, Dict[str, str]]]]): Python dictionary with the format correctly.
        path (str): Relative path where excel file will be created.
    """

    # Creating workbook and sheet
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet()
    # Creating cells format configuration
    keys_format = workbook.add_format({'bold': True,
                                       'font_size': 20,
                                       'align': 'center',
                                       'bg_color': 'gray',
                                       'border': True})
    cells_format = workbook.add_format(
        {'align': 'center'})
    actuator_format = workbook.add_format(
        {'bold': True, 'align': 'center', 'bg_color': 'gray'})
    # Indicating cell position within sheet
    current_row = 0
    current_col = 0
    # Anotate max_column in order to know excel extension
    max_col = 1

    worksheet.write(current_row, current_col, 'Name', keys_format)
    worksheet.write(current_row, current_col + 1, 'Type', keys_format)
    current_row += 1

    for key, info in schedulers.items():
        worksheet.write(current_row, current_col, key, actuator_format)
        current_col += 1
        worksheet.write(current_row, current_col, info['Type'], cells_format)
        current_col += 1
        for object_name, values in info.items():
            if isinstance(values, dict):
                worksheet.write(
                    current_row,
                    current_col,
                    'Name: ' + object_name)
                current_col += 1
                worksheet.write(
                    current_row,
                    current_col,
                    'Field: ' +
                    values['field_name'])
                current_col += 1
                worksheet.write(
                    current_row,
                    current_col,
                    'Table type: ' +
                    values['table_name'])
                current_col += 1
        # Update max column if it is necessary
        if current_col > max_col:
            max_col = current_col

        current_row += 1
        current_col = 0

    current_row = 0
    object_num = 1
    # Updating columns extension
    worksheet.set_column(0, max_col, 40)

    for i in range(2, max_col, 3):
        worksheet.merge_range(
            current_row,
            i,
            current_row,
            i + 2,
            'OBJECT' + str(object_num),
            keys_format)
        object_num += 1
    workbook.close()

# ------------------ Reading JSON environment configuration ------------------ #


def json_to_variables(variables: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """Read variables dictionary (from Sinergym JSON conf) and adapt it to the
       EnergyPlus format. More information about Sinergym JSON configuration format
       in documentation.

    Args:
        variables (Dict[str, Any]): Dictionary from Sinergym JSON configuration with variables information.

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
                assert len( specification['variable_names']) == len(
                    specification['keys']), 'variable names and keys must have the same len in {}'.format(variable)
                for variable_name, key_name in list(
                        zip(specification['variable_names'], specification['keys'])):
                    output[variable_name] = (variable, key_name)

            else:
                raise RuntimeError

        else:

            raise RuntimeError

    return output


def json_to_meters(meters: Dict[str, str]) -> Dict[str, str]:
    """Read meters dictionary (from Sinergym JSON conf) and adapt it to the
       EnergyPlus format. More information about Sinergym JSON configuration format
       in documentation.

    Args:
        meters (Dict[str, str]): Dictionary from Sinergym JSON configuration with meters information.

    Returns:
        Dict[str, str]: Dictionary with meters information in EnergyPlus API format.
    """

    output = {}

    for meter_name, variable_name in meters.items():
        output[variable_name] = meter_name

    return output


def json_to_actuators(
        actuators: Dict[str, Dict[str, str]]) -> Dict[str, Tuple[str, str, str]]:
    """Read actuators dictionary (from Sinergym JSON conf) and adapt it to the
       EnergyPlus format. More information about Sinergym JSON configuration format
       in documentation.

    Args:
        actuators (Dict[str, Dict[str, str]]): Dictionary from Sinergym JSON configuration with actuators information.

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
    """Convert a conf from json format (sinergym/data/default_configuration/file.json) in a dictionary of all possible environments as dictionary with id as key and env_kwargs as value.
       More information about Sinergym environment configuration in JSON format in documentation.

    Args:
        conf (Dict[str, Any]): Dictionary from read json configuration file (sinergym/data/default_configuration/file.json).

    Returns:
        Dict[str,[Dict[str, Any]]: All possible Sinergym environment constructor kwargs.
    """

    configurations = {}

    variables = json_to_variables(conf['variables'])
    meters = json_to_meters(conf['meters'])
    actuators = json_to_actuators(conf['actuators'])

    assert len(conf['weather_specification']['weather_files']) == len(
        conf['weather_specification']['keys']), 'Weather files and id keys must have the same len'

    weather_info = list(zip(conf['weather_specification']['keys'],
                        conf['weather_specification']['weather_files']))

    variation = conf.get('variation')

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
            'reward': eval(conf['reward']),
            'reward_kwargs': conf['reward_kwargs'],
            'max_ep_data_store_num': conf['max_ep_data_store_num'],
            'env_name': id.replace('Eplus-', ''),
            'config_params': conf.get('config_params')
        }
        configurations[id] = env_kwargs

        if variation:

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
                'weather_variability': tuple(variation),
                'reward': eval(conf['reward']),
                'reward_kwargs': conf['reward_kwargs'],
                'max_ep_data_store_num': conf['max_ep_data_store_num'],
                'env_name': id.replace('Eplus-', ''),
                'config_params': conf.get('config_params')
            }
            configurations[id] = env_kwargs

    return configurations
