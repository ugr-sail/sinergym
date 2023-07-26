"""Common utilities."""

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import xlsxwriter
from eppy.modeleditor import IDF
from opyplus.epm.record import Record

from sinergym.utils.constants import YEAR


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


def get_season_comfort_range(
        year: int, month: int, day: int) -> Tuple[float, float]:
    """Get comfort temperature range depending on season. The comfort ranges are those
    defined by ASHRAE in Standard 55—Thermal Environmental Conditions for Human Occupancy (2004).

    Args:
        year (int): current year
        month (int): current month
        day (int): current day

    Returns:
        Tuple[float, float]: Comfort temperature from the correct season.
    """

    summer_start_date = datetime(year, 6, 1)
    summer_final_date = datetime(year, 9, 30)

    range_comfort_summer = (23.0, 26.0)
    range_comfort_winter = (20.0, 23.5)

    current_dt = datetime(year, month, day)

    if current_dt >= summer_start_date and current_dt <= summer_final_date:
        comfort = range_comfort_summer
    else:
        comfort = range_comfort_winter

    return comfort


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
