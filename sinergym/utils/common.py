"""Common utilities."""

import csv
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pydoc import locate
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas as pd
from opyplus import Epm, WeatherData
from opyplus.epm.record import Record

# NORMALIZATION RANGES
RANGES_5ZONE = {'Facility Total HVAC Electricity Demand Rate (Whole Building)': [173.6583692738386,
                                                                                 32595.57259261767],
                'People Air Temperature (SPACE1-1 PEOPLE 1)': [0.0, 30.00826655379267],
                'Site Diffuse Solar Radiation Rate per Area (Environment)': [0.0, 588.0],
                'Site Direct Solar Radiation Rate per Area (Environment)': [0.0, 1033.0],
                'Site Outdoor Air Drybulb Temperature (Environment)': [-31.05437255409474,
                                                                       60.72839186915495],
                'Site Outdoor Air Relative Humidity (Environment)': [3.0, 100.0],
                'Site Wind Direction (Environment)': [0.0, 357.5],
                'Site Wind Speed (Environment)': [0.0, 23.1],
                'Space1-ClgSetP-RL': [21.0, 30.0],
                'Space1-HtgSetP-RL': [15.0, 22.49999],
                'Zone Air Relative Humidity (SPACE1-1)': [3.287277410867238,
                                                          87.60662171287048],
                'Zone Air Temperature (SPACE1-1)': [15.22565264653451, 30.00826655379267],
                'Zone People Occupant Count (SPACE1-1)': [0.0, 11.0],
                'Zone Thermal Comfort Clothing Value (SPACE1-1 PEOPLE 1)': [0.0, 1.0],
                'Zone Thermal Comfort Fanger Model PPD (SPACE1-1 PEOPLE 1)': [0.0,
                                                                              98.37141259444684],
                'Zone Thermal Comfort Mean Radiant Temperature (SPACE1-1 PEOPLE 1)': [0.0,
                                                                                      35.98853496778508],
                'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)': [21.0, 30.0],
                'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)': [15.0,
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

RANGES_IW = {
    "Site Outdoor Air Drybulb Temperature": [-13.0, 26.0],
    "Site Outdoor Air Relative Humidity": [0.0, 100.0],
    "Site Wind Speed": [0.0, 11.0],
    "Site Wind Direction": [0.0, 360.0],
    "Site Diffuse Solar Radiation Rate per Area": [0.0, 378.0],
    "Site Direct Solar Radiation Rate per Area": [0.0, 1000.0],
    "IW Hot Water System OA Enable Flag OA Setpoint": [-30.0, 30.0],
    "IW Average PPD": [0.0, 100.0],
    "IW Effective Zone Air Temperature Setpoint": [18.0, 25.0],
    "IW North Zone Average Temperature": [18.0, 25.0],
    "IW Effective IAT Setpoint by Logics": [18.0, 25.0],
    "IW Occupy Mode Flag": [0.0, 1.0],
    "IW Calculated Heating Demand": [0.0, 85.0],
    'day': [1.0, 31.0],
    'done': [False, True],
    'hour': [0.0, 23.0],
    'month': [1.0, 12.0],
    'year': [1.0, 2022.0],
}

RANGES_DATACENTER = {
    'East-ClgSetP-RL': [21.0, 30.0],
    'East-HtgSetP-RL': [15.0, 22.499973],
    'Facility Total HVAC Electricity Demand Rate (Whole Building)': [1763.7415,
                                                                     76803.016],
    'People Air Temperature (East Zone PEOPLE)': [0.0, 30.279287],
    'People Air Temperature (West Zone PEOPLE)': [0.0, 30.260946],
    'Site Diffuse Solar Radiation Rate per Area (Environment)': [0.0, 588.0],
    'Site Direct Solar Radiation Rate per Area (Environment)': [0.0, 1033.0],
    'Site Outdoor Air Drybulb Temperature (Environment)': [-16.049532, 42.0],
    'Site Outdoor Air Relative Humidity (Environment)': [3.0, 100.0],
    'Site Wind Direction (Environment)': [0.0, 357.5],
    'Site Wind Speed (Environment)': [0.0, 17.5],
    'West-ClgSetP-RL': [21.0, 30.0],
    'West-HtgSetP-RL': [15.0, 22.499998],
    'Zone Air Relative Humidity (East Zone)': [1.8851701, 67.184616],
    'Zone Air Relative Humidity (West Zone)': [1.8945858, 66.7946],
    'Zone Air Temperature (East Zone)': [21.003511, 30.279287],
    'Zone Air Temperature (West Zone)': [21.004263, 30.260946],
    'Zone People Occupant Count (East Zone)': [0.0, 7.0],
    'Zone People Occupant Count (West Zone)': [0.0, 11.0],
    'Zone Thermal Comfort Clothing Value (East Zone PEOPLE)': [0.0, 0.0],
    'Zone Thermal Comfort Clothing Value (West Zone PEOPLE)': [0.0, 0.0],
    'Zone Thermal Comfort Fanger Model PPD (East Zone PEOPLE)': [0.0, 66.75793],
    'Zone Thermal Comfort Fanger Model PPD (West Zone PEOPLE)': [0.0, 59.53962],
    'Zone Thermal Comfort Mean Radiant Temperature (East Zone PEOPLE)': [0.0,
                                                                         29.321169],
    'Zone Thermal Comfort Mean Radiant Temperature (West Zone PEOPLE)': [0.0,
                                                                         29.04933],
    'Zone Thermostat Cooling Setpoint Temperature (East Zone)': [21.0, 30.0],
    'Zone Thermostat Cooling Setpoint Temperature (West Zone)': [21.0, 30.0],
    'Zone Thermostat Heating Setpoint Temperature (East Zone)': [15.0, 22.499973],
    'Zone Thermostat Heating Setpoint Temperature (West Zone)': [15.0, 22.499998],
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


def get_current_time_info(
        epm: Epm, sec_elapsed: float) -> List[int]:
    """Returns the current day, month and hour given the seconds elapsed since the simulation started.

    Args:
        epm (opyplus.Epm): EnergyPlus model object.
        sec_elapsed (float): Seconds elapsed since the start of the simulation

    Returns:
        List[int]: A List composed by the current year, day, month and hour in the simulation.

    """
    start_date = datetime(
        year=int(epm.RunPeriod[0]['begin_year']),
        month=int(epm.RunPeriod[0]['begin_month']),
        day=int(epm.RunPeriod[0]['begin_day_of_month'])
    )

    current_date = start_date + timedelta(seconds=sec_elapsed)

    return [
        int(current_date.year),
        int(current_date.month),
        int(current_date.day),
        int(current_date.hour),
    ]


def parse_variables(var_file: str) -> Dict[str, List[str]]:
    """Parse observation and action to dictionary.

    Args:
        var_file (str): Variables file path.

    Returns:
        Dict[str, List[str]]: observation and action keys; a list with the name of the observation <variables> (<zone>) and a list with the name of the action <variables> respectively.
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables = {}
    observation = []
    action = []
    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            observation.append(var[0].attrib['type'] +
                               ' (' + var[0].attrib['name'] + ')')
        if var.attrib['source'] == 'Ptolemy':
            action.append(var[0].attrib['schedule'])

    variables['observation'] = observation
    variables['action'] = action

    return variables


def parse_observation_action_space(space_file: str) -> Dict[str, Any]:
    """Parse observation space definition to gym env.

    Args:
        space_file (str): Observation space definition file path.

    Raises:
        RuntimeError: If root XML tag is not 'space'.

    Returns:
        Dict[str, Any]: 'observation', 'discrete_action' and 'continuous_action' keys. tuple for gym.spaces.Box() arguments, dictionary action mapping for gym.spaces.Discrete() and tuple for gym.spaces.Box()
    """

    tree = ET.parse(space_file)
    root = tree.getroot()
    if(root.tag != 'space'):
        raise RuntimeError(
            'Failed to open environment action observation space (Check XML definition)')

    # Observation and action spaces
    observation_space = root.find('observation-space')
    action_space = root.find('action-space')
    assert observation_space is not None and action_space is not None

    discrete_action_space = action_space.find('discrete')
    continuous_action_space = action_space.find('continuous')
    assert discrete_action_space is not None and continuous_action_space is not None

    action_shape = action_space.find('shape')
    assert action_shape is not None
    action_shape = int(action_shape.attrib['value'])

    # Observation space values
    dtype = observation_space.find('dtype')
    assert dtype is not None
    dtype = locate(dtype.attrib['value'])
    assert callable(dtype)
    low = observation_space.find('low')
    high = observation_space.find('high')
    shape = observation_space.find('shape')
    assert low is not None and high is not None and shape is not None
    low = dtype(low.attrib['value'])
    high = dtype(high.attrib['value'])
    shape = int(shape.attrib['value'])
    observation = (low, high, (shape,), dtype)

    # discrete action values
    discrete_action = {}
    for element in discrete_action_space:
        # element mapping index
        index = int(element.attrib['index'])
        # element action values
        actions = tuple([float(element.attrib['action' + str(i)])
                         for i in range(action_shape)])

        discrete_action[index] = actions

    # continuous actions values
    actions_dtype = locate(
        continuous_action_space.find('dtype').attrib['value'])
    low_ranges = continuous_action_space.find('low-ranges')
    high_ranges = continuous_action_space.find('high-ranges')
    low_action = [actions_dtype(element.attrib['value'])
                  for element in low_ranges]
    high_action = [actions_dtype(element.attrib['value'])
                   for element in high_ranges]

    continuous_action = (low_action, high_action,
                         (action_shape,), actions_dtype)

    # return final output
    result = {}
    result['observation'] = observation
    result['discrete_action'] = discrete_action
    result['continuous_action'] = continuous_action
    return result


def create_variable_weather(
        weather_data: WeatherData,
        original_epw_file: str,
        columns: List[str] = ['drybulb'],
        variation: Optional[Tuple[float, float, float]] = None) -> Optional[str]:
    """Create a new weather file using Ornstein-Uhlenbeck process.

    Args:
        weather_data (opyplus.WeatherData): Opyplus object with the weather for the simulation.
        original_epw_file (str): Path to the original EPW file.
        columns (List[str], optional): List of columns to be affected. Defaults to ['drybulb'].
        variation (Optional[Tuple[float, float, float]], optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.

    Returns:
        Optional[str]: Name of the file created in the same location as the original one.
    """

    if variation is None:
        return None
    else:
        # Get dataframe with weather series
        df = weather_data.get_weather_series()

        sigma = variation[0]  # Standard deviation.
        mu = variation[1]  # Mean.
        tau = variation[2]  # Time constant.

        T = 1.  # Total time.
        # All the columns are going to have the same num of rows since they are
        # in the same dataframe
        n = len(df[columns[0]])
        dt = T / n
        # t = np.linspace(0., T, n)  # Vector of times.

        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)

        x = np.zeros(n)

        # Create noise
        for i in range(n - 1):
            x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
                sigma_bis * sqrtdt * np.random.randn()

        for column in columns:
            # Add noise
            df[column] += x

        # Save new weather data
        weather_data.set_weather_series(df)
        filename = original_epw_file.split('.epw')[0]
        filename += '_Random_%s_%s_%s.epw' % (str(sigma), str(mu), str(tau))
        weather_data.to_epw(filename)
        return filename


def ranges_getter(output_path: str,
                  last_result: Optional[Dict[str, List[float]]] = None
                  ) -> Dict[str, List[float]]:
    """Given a path with simulations outputs, this function is used to extract max and min absolute values of all episodes in each variable. If a dict ranges is given, will be updated.

    Args:
        output_path (str): path with simulations directories (Eplus-env-<env_name>).
        last_result (Optional[Dict[str, List[float]]], optional): Last ranges dict to be updated. This will be created if it is not given.

    Returns:
        Dict[str, List[float]]: list min,max of each variable as a key.

    """

    if last_result is not None:
        result = last_result
    else:
        result = {}

    content = os.listdir(output_path)
    for simulation in content:
        if os.path.isdir(
            output_path +
            '/' +
                simulation) and simulation.startswith('Eplus-env'):
            simulation_content = os.listdir(output_path + '/' + simulation)
            for episode_dir in simulation_content:
                if os.path.isdir(
                    output_path +
                    '/' +
                    simulation +
                    '/' +
                        episode_dir):
                    monitor_path = output_path + '/' + simulation + '/' + episode_dir + '/monitor.csv'
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


def setpoints_transform(action: Union[int,
                                      float,
                                      np.integer,
                                      np.ndarray,
                                      List[Any],
                                      Tuple[Any]],
                        action_space: gym.spaces.Box,
                        setpoints_space: List[List[Union[int, float]]]
                        ) -> List[Union[int, float]]:
    """Given an action inner gym action_space, this will be converted into an action inner setpoints_space (Sinergym Simulation).

     Args:
        action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action of a step in gym simulation.
        action_space (gym.spaces.Box): Gym action space
        setpoints_space (List[List[Union[int, float]]]): Sinergym simulation action space

     Returns:
        List[Union[int, float]]: Action transformed into simulation action space.

    """

    action_ = []

    for i, value in enumerate(action):
        if action_space.low[i] <= value <= action_space.high[i]:
            a_max_min = action_space.high[i] - \
                action_space.low[i]
            sp_max_min = setpoints_space[i][1] - \
                setpoints_space[i][0]

            action_.append(
                setpoints_space[i][0] + (value - action_space.low[i]) * sp_max_min / a_max_min)
        else:
            # If action is outer action_space already, it don't need
            # transformation
            action_.append(value)

    return action_


def get_record_keys(record: Record) -> List[str]:
    """Given an opyplus Epm Record (one element from opyplus.epm object) this function returns list of keys (opyplus hasn't got this functionality explicitly)

     Args:
        record (opyplus.Epm.Record): Element from Epm object.

     Returns:
        List[str]: Key list from record.
    """
    return [field.ref for field in record._table._dev_descriptor._field_descriptors]


def prepare_batch_from_records(records: List[Record]) -> List[Dict[str, Any]]:
    """Prepare a list of dictionaries in order to use Epm.add_batch directly

    Args:
        records List[opyplus.Epm.Record]: List of records which will be converted to dictionary batch.

    Returns:
        List[Dict[str, Any]]: List of dicts where each dictionary is a record element.
    """

    batch = []
    for record in records:
        aux_dict = {}
        for key in get_record_keys(record):
            aux_dict[key] = record[key]
        batch.append(aux_dict)

    return batch


def get_season_comfort_range(year, month, day):
    """Get comfort temperature range depending on season. The comfort ranges are those
    defined by ASHRAE in Standard 55—Thermal Environmental Conditions for Human Occupancy (2004).

    Args:
        year (int): current year
        month (int): current month
        day (int): current day
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


class Logger():
    """Sinergym terminal logger for simulation executions.
    """

    def getLogger(
            self,
            name: str,
            level: str,
            formatter: str) -> logging.Logger:
        """Return Sinergym logger for the progress output in terminal.

        Args:
            name (str): logger name
            level (str): logger level
            formatter (str): logger formatter

        Returns:
            logging.logger

        """
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger


class CSVLogger(object):
    """CSV Logger for agent interaction with environment.

        :param monitor_header: CSV header for sub_run_N/monitor.csv which record interaction step by step.
        :param progress_header: CSV header for res_N/progress.csv which record main data episode by episode.
        :param log_file: log_file path for monitor.csv, there will be one CSV per episode.
        :param log_progress_file: log_file path for progress.csv, there will be only one CSV per whole simulation.
        :param flag: This flag is used to activate (True) or deactivate (False) Logger in real time.
        :param steps_data, rewards, powers, etc: These arrays are used to record steps data to elaborate main data for progress.csv later.
        :param total_timesteps: Current episode timesteps executed.
        :param total_time_elapsed: Current episode time elapsed (simulation seconds).
        :param comfort_violation_timesteps: Current episode timesteps whose comfort_penalty!=0.
        :param steps_data: It is a array of str's. Each element belong to a step data.

    """

    def __init__(
            self,
            monitor_header: str,
            progress_header: str,
            log_progress_file: str,
            log_file: Optional[str] = None,
            flag: bool = True):

        self.monitor_header = monitor_header
        self.progress_header = progress_header + '\n'
        self.log_file = log_file
        self.log_progress_file = log_progress_file
        self.flag = flag

        # episode data
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self.comfort_penalties = []
        self.power_penalties = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def log_step(
            self,
            timestep: int,
            observation: List[Any],
            action: Union[List[Union[int, float]], List[None]],
            simulation_time: float,
            reward: Optional[float],
            total_power_no_units: Optional[float],
            comfort_penalty: Optional[float],
            power: Optional[float],
            done: bool) -> None:
        """Log step information and store it in steps_data attribute.

        Args:
            timestep (int): Current episode timestep in simulation.
            observation (list): Values that belong to current observation.
            action (list): Values that belong to current action.
            simulation_time (float): Total time elapsed in current episode (seconds).
            reward (float): Current reward achieved.
            total_power_no_units (float): Power consumption penalty depending on reward function.
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            power (float): Power consumption in current step (W).
            done (bool): It specifies if this step terminates episode or not.

        """
        if self.flag:
            row_contents = [timestep] + list(observation) + \
                list(action) + [simulation_time, reward,
                                total_power_no_units, comfort_penalty, done]
            self.steps_data.append(row_contents)

            # Store step information for episode
            self._store_step_information(
                reward,
                power,
                comfort_penalty,
                total_power_no_units,
                timestep,
                simulation_time)
        else:
            pass

    def log_step_normalize(
            self,
            timestep: int,
            observation: List[Any],
            action: Union[List[Union[int, float]], List[None]],
            simulation_time: float,
            reward: Optional[float],
            total_power_no_units: Optional[float],
            comfort_penalty: Optional[float],
            done: bool) -> None:
        """Log step information and store it in steps_data_normalized attribute.

        Args:
            timestep (int): Current episode timestep in simulation.
            observation (List[Any]): Values that belong to current observation.
            action (List[Union[int, float]]): Values that belong to current action.
            simulation_time (float): Total time elapsed in current episode (seconds).
            reward (float): Current reward achieved.
            total_power_no_units (float): Power consumption penalty depending on reward function.
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            done (bool): It specifies if this step terminates episode or not.
        """
        if self.flag:
            row_contents = [timestep] + list(observation) + \
                list(action) + [simulation_time, reward,
                                total_power_no_units, comfort_penalty, done]
            self.steps_data_normalized.append(row_contents)
        else:
            pass

    def log_episode(self, episode: int) -> None:
        """Log episode main information using steps_data param.

        Args:
            episode (int): Current simulation episode number.

        """
        if self.flag:
            # statistics metrics for whole episode
            ep_mean_reward = np.mean(self.rewards)
            ep_cumulative_reward = np.sum(self.rewards)
            ep_cumulative_power = np.sum(self.powers)
            ep_mean_power = np.mean(self.powers)
            ep_cumulative_comfort_penalty = np.sum(self.comfort_penalties)
            ep_mean_comfort_penalty = np.mean(self.comfort_penalties)
            ep_cumulative_power_penalty = np.sum(self.power_penalties)
            ep_mean_power_penalty = np.mean(self.power_penalties)
            try:
                comfort_violation = (
                    self.comfort_violation_timesteps /
                    self.total_timesteps *
                    100)
            except ZeroDivisionError:
                comfort_violation = np.nan

            # write steps_info in monitor.csv
            with open(self.log_file, 'w', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerows(self.steps_data)

            # Write normalize steps_info in monitor_normalized.csv
            if len(self.steps_data_normalized) > 1:
                with open(self.log_file[:-4] + '_normalized.csv', 'w', newline='') as file_obj:
                    # Create a writer object from csv module
                    csv_writer = csv.writer(file_obj)
                    # Add contents of list as last row in the csv file
                    csv_writer.writerows(self.steps_data_normalized)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [
                episode,
                ep_cumulative_reward,
                ep_mean_reward,
                ep_cumulative_power,
                ep_mean_power,
                ep_cumulative_comfort_penalty,
                ep_mean_comfort_penalty,
                ep_cumulative_power_penalty,
                ep_mean_power_penalty,
                comfort_violation,
                self.total_timesteps,
                self.total_time_elapsed]
            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            # Reset episode information
            self._reset_logger()
        else:
            pass

    def set_log_file(self, new_log_file: str) -> None:
        """Change log_file path for monitor.csv when an episode ends.

        Args:
            new_log_file (str): New log path depending on simulation.

        """
        if self.flag:
            self.log_file = new_log_file
            if self.log_file:
                with open(self.log_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.monitor_header)
        else:
            pass

    def _store_step_information(
            self,
            reward: float,
            power: float,
            comfort_penalty: float,
            power_penalty: float,
            timestep: int,
            simulation_time: float) -> None:
        """Store relevant data to episode summary in progress.csv.

        Args:
            reward (float): Current reward achieved.
            power (float): Power consumption in current step (W).
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            power_penalty (float): Power consumption penalty depending on reward function.
            timestep (int): Current episode timestep in simulation.
            simulation_time (float): Total time elapsed in current episode (seconds).

        """
        if reward is not None:
            self.rewards.append(reward)
        if power is not None:
            self.powers.append(power)
        if comfort_penalty is not None:
            self.comfort_penalties.append(comfort_penalty)
        if power_penalty is not None:
            self.power_penalties.append(power_penalty)
        if comfort_penalty != 0:
            self.comfort_violation_timesteps += 1
        self.total_timesteps = timestep
        self.total_time_elapsed = simulation_time

    def _reset_logger(self) -> None:
        """Reset relevant data to next episode summary in progress.csv.
        """
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self. comfort_penalties = []
        self.power_penalties = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def activate_flag(self) -> None:
        """Activate Sinergym CSV logger
        """
        self.flag = True

    def deactivate_flag(self) -> None:
        """Deactivate Sinergym CSV logger
        """
        self.flag = False
