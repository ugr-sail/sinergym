"""Common utilities."""

import os
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pydoc import locate
import csv
import pandas as pd

from datetime import datetime, timedelta

# NORMALIZATION RANGES
RANGES_5ZONE = {'Facility Total HVAC Electric Demand Power (Whole Building)': [173.6583692738386,
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
                'reward': [-3.550779087370951, -0.0086829184636919],
                'time (seconds)': [0, 31536000],
                'timestep': [0, 35040],
                'total_power_no_units': [-3.259557259261767, -0.0173658369273838]}


def get_delta_seconds(year, st_mon, st_day, end_mon, end_day):
    """Returns the delta seconds between `year:st_mon:st_day:0:0:0` and
    `year:end_mon:end_day:24:0:0`.

    Args:
        st_year (int): Year.
        st_mon (int): Start month.
        st_day (int): Start day.
        end_mon (int): End month.
        end_day (int): End day.

    Returns:
        float: Time difference in seconds.
    """

    startTime = datetime(year, st_mon, st_day, 0, 0, 0)
    endTime = datetime(year, end_mon, end_day, 23, 0, 0) + timedelta(0, 3600)
    delta_sec = (endTime - startTime).total_seconds()
    return delta_sec


def get_current_time_info(epm, sec_elapsed, sim_year=1991):
    """Returns the current day, month and hour given the seconds elapsed since the simulation started.

    Args:
        epm (opyplus.Epm): EnergyPlus model object.
        sec_elapsed (int): Seconds elapsed since the start of the simulation
        sim_year (int, optional): Year of the simulation. Defaults to 1991.

    Returns:
        (int, int, int): A tuple composed by the current day, month and hour in the simulation.
    """

    start_date = datetime(
        year=sim_year,  # epm.RunPeriod[0]['start_year'],
        month=epm.RunPeriod[0]['begin_month'],
        day=epm.RunPeriod[0]['begin_day_of_month']
    )

    current_date = start_date + timedelta(seconds=sec_elapsed)

    return (current_date.day, current_date.month, current_date.hour)


def parse_variables(var_file):
    """Parse observation and action to dictionary.

    Args:
        var_file (str): Variables file path.

    Returns:
        dict:
            {'observation': A list with the name of the observation <variables> (<zone>) \n
            'action'      : A list with the name og the action <variables>}.
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables = {}
    observation = []
    action = []
    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            observation.append(var[0].attrib['type'] +
                               ' ('+var[0].attrib['name']+')')
        if var.attrib['source'] == 'Ptolemy':
            action.append(var[0].attrib['schedule'])

    variables['observation'] = observation
    variables['action'] = action

    return variables


def parse_observation_action_space(space_file):
    """Parse observation space definition to gym env.

    Args:
        space_file (str): Observation space definition file path.

    Returns:
        dictionary:
                {'observation'     : tupple for gym.spaces.Box() arguments, \n
                'discrete_action'  : dictionary action mapping for gym.spaces.Discrete(), \n
                'continuos_action' : tuple for gym.spaces.Box()}
    """
    tree = ET.parse(space_file)
    root = tree.getroot()
    if(root.tag != 'space'):
        raise RuntimeError(
            'Failed to open environment action observation space (Check XML definition)')

    # Observation and action spaces
    observation_space = root.find('observation-space')
    action_space = root.find('action-space')
    discrete_action_space = action_space.find('discrete')
    continuous_action_space = action_space.find('continuous')

    action_shape = int(action_space.find('shape').attrib['value'])

    # Observation space values
    dtype = locate(observation_space.find('dtype').attrib['value'])
    low = dtype(observation_space.find('low').attrib['value'])
    high = dtype(observation_space.find('high').attrib['value'])
    shape = int(observation_space.find('shape').attrib['value'])
    observation = (low, high, (shape,), dtype)

    # discrete action values
    discrete_action = {}
    for element in discrete_action_space:
        # element mapping index
        index = int(element.attrib['index'])
        # element action values
        actions = tuple([float(element.attrib['action'+str(i)])
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


def create_variable_weather(weather_data, original_epw_file, columns: list = ['drybulb'], variation: tuple = None):
    """Create a new weather file adding gaussian noise to the original one.

    Args:
        weather_data (opyplus.WeatherData): Opyplus object with the weather for the simulation.
        original_epw_file (str): Path to the original EPW file.
        columns (list, optional): List of columns to be affected. Defaults to ['drybulb'].
        variation (tuple, optional): Tuple with the mean and standard desviation of the Gaussian noise. Defaults to None.

    Returns:
        str: Name of the file created in the same location as the original one.
    """

    if variation is None:
        return None
    else:
        # Get dataframe with weather series
        df = weather_data.get_weather_series()

        # Generate random noise
        shape = (df.shape[0], len(columns))
        mu, std = variation
        noise = np.random.normal(mu, std, shape)
        df[columns] += noise

        # Save new weather data
        weather_data.set_weather_series(df)
        filename = original_epw_file.split('.epw')[0]
        filename += '_Random_%s_%s.epw' % (str(mu), str(std))
        weather_data.to_epw(filename)
        return filename


def ranges_getter(output_path, last_result=None):
    """Given a path with simulations outputs, this function is used to extract max and min absolute valors of all episodes in each variable. If a dict ranges is given, will be updated

    Args:
        output_path (str): path with simulations directories (Eplus-env-*).
        last_result (dict): Last ranges dict to be updated. This will be created if it is not given.

    Returns:
        dict: list min,max of each variable as a key.
    """
    if last_result is not None:
        result = last_result
    else:
        result = {}

    content = os.listdir(output_path)
    for simulation in content:
        if os.path.isdir(output_path+'/'+simulation) and simulation.startswith('Eplus-env'):
            simulation_content = os.listdir(output_path+'/'+simulation)
            for episode_dir in simulation_content:
                if os.path.isdir(output_path+'/'+simulation+'/'+episode_dir):
                    monitor_path = output_path+'/'+simulation+'/'+episode_dir+'/monitor.csv'
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


class Logger():
    def getLogger(self, name, level, formatter):
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger


class CSVLogger(object):
    def __init__(self, monitor_header, progress_header, log_progress_file, log_file=None, flag=True):

        self.monitor_header = monitor_header
        self.progress_header = progress_header+'\n'
        self.log_file = log_file
        self.log_progress_file = log_progress_file
        self.flag = flag

        # episode data
        self.steps_data = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def log_step(self, timestep, date, observation, action, simulation_time, reward, total_power_no_units, comfort_penalty, power, done):
        if self.flag:
            row_contents = [timestep] + list(date) + list(observation) + \
                list(action) + [simulation_time, reward,
                                total_power_no_units, comfort_penalty,  done]
            self.steps_data.append(row_contents)

            # Store step information for episode
            self._store_step_information(
                reward, power, comfort_penalty, timestep, simulation_time)
        else:
            pass

    def log_episode(self, episode):
        if self.flag:
            # statistics metrics for whole episode
            ep_mean_reward = np.mean(self.rewards)
            ep_total_reward = np.sum(self.rewards)
            ep_mean_power = np.mean(self.powers)
            comfort_violation = (
                self.comfort_violation_timesteps/self.total_timesteps*100)

            # write steps_info in monitor.csv
            with open(self.log_file, 'w', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerows(self.steps_data)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [episode, ep_total_reward, ep_mean_reward, ep_mean_power, comfort_violation,
                            self.total_timesteps, self.total_time_elapsed]
            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            # Reset episode information
            self._reset_logger()
        else:
            pass

    def set_log_file(self, new_log_file):
        if self.flag:
            self.log_file = new_log_file
            if self.log_file:
                with open(self.log_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.monitor_header)
        else:
            pass

    def _store_step_information(self, reward, power, comfort_penalty, timestep, simulation_time):
        if reward is not None:
            self.rewards.append(reward)
        if power is not None:
            self.powers.append(power)
        if comfort_penalty != 0:
            self.comfort_violation_timesteps += 1
        self.total_timesteps = timestep
        self.total_time_elapsed = simulation_time

    def _reset_logger(self):
        self.steps_data = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def activate_flag(self):
        self.flag = True

    def deactivate_flag(self):
        self.flag = False
