"""Common utilities."""

import os
import logging
import numpy as np
import xml.etree.ElementTree as ET
from pydoc import locate

from datetime import datetime, timedelta


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
            {"observation": A list with the name of the observation <variables> (<zone>) \n
            "action"      : A list with the name og the action <variables>}.
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables={}
    observation=[]
    action =[]
    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            observation.append(var[0].attrib['type']+" ("+var[0].attrib['name']+")")
        if var.attrib['source'] == 'Ptolemy':
            action.append(var[0].attrib['schedule']) 

    variables["observation"]=observation
    variables["action"]=action

    return variables

def parse_observation_action_space(space_file):
    """Parse observation space definition to gym env.

    Args:
        space_file (str): Observation space definition file path.

    Returns:
        dictionary: 
                {"observation"     : tupple for gym.spaces.Box() arguments, \n
                "discrete_action"  : dictionary action mapping for gym.spaces.Discrete(), \n
                "continuos_action" : tuple for gym.spaces.Box()}
    """
    tree = ET.parse(space_file)
    root = tree.getroot()
    if(root.tag!="space"):
        raise RuntimeError("Failed to open environment action observation space (Check XML definition)")

    #Observation and action spaces
    observation_space=root.find('observation-space')
    action_space=root.find("action-space")
    discrete_action_space=action_space.find("discrete")
    continuous_action_space=action_space.find("continuous")

    action_shape=int(action_space.find("shape").attrib["value"])

    #Observation space values
    dtype=locate(observation_space.find("dtype").attrib["value"])
    low=dtype(observation_space.find("low").attrib["value"])
    high=dtype(observation_space.find("high").attrib["value"])
    shape=int(observation_space.find("shape").attrib["value"])
    observation=(low, high, (shape,), dtype)

    #discrete action values
    discrete_action={}
    for element in discrete_action_space:
        #element mapping index
        index=int(element.attrib["index"])
        #element action values
        actions=tuple([float(element.attrib["action"+str(i)]) for i in range(action_shape)])

        discrete_action[index]=actions

    #continuous actions values
    actions_dtype=locate(continuous_action_space.find("dtype").attrib["value"])
    low_ranges=continuous_action_space.find("low-ranges")
    high_ranges=continuous_action_space.find("high-ranges")
    low_action=[actions_dtype(element.attrib["value"]) for element in low_ranges]
    high_action=[actions_dtype(element.attrib["value"]) for element in high_ranges] 

    continuous_action=(low_action, high_action, (action_shape,) , actions_dtype)

    #return final output
    result={}
    result["observation"]=observation
    result["discrete_action"]=discrete_action
    result["continuous_action"]=continuous_action
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


class Logger():
    def getLogger(self, name, level, formatter):
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger

# based on https://gist.github.com/robdmc/d78d48467e3daea22fe6
# class CSVLogger(object):
#     def __init__(self, name, log_file=None, level='info', needs_header=True, header=None, formatter):
#         # create logger on the current module and set its level
#         self.logger = logging.getLogger(name)
#         self.logger.setLevel(logging.INFO)
#         self.logger.setLevel(getattr(logging, level.upper()))
#         self.needs_header = needs_header
#         self.header=header

#         # create a formatter that creates a single line of json with a comma at the end
#         #'%(created)s,%(name)s,"%(utc_time)s","%(eastern_time)s",%(levelname)s,"%(message)s"'
#         self.formatter = logging.Formatter(formatter)      

#         self.log_file = log_file
#         if self.log_file:
#             # create a channel for handling the logger (stderr) and set its format
#             consoleHandler = logging.FileHandler(log_file)
#         else:
#             # create a channel for handling the logger (stderr) and set its format
#             consoleHandler = logging.StreamHandler()
#         consoleHandler.setFormatter(self.formatter)

#         # connect the logger to the channel
#         self.logger.addHandler(consoleHandler)

#         # Create CSV file with header if it's required
#         if self.needs_header:
#             if self.log_file:
#                 with open(self.log_file, 'a') as file_obj:
#                     if self.needs_header:
#                         file_obj.write(self.header)
#             else:
#                 if self.needs_header:
#                     sys.stderr.write(self.header)

#     def log(self, observation, action, reward, temperature, power, timestep, simulation_time, level='info'):
#         #Aquí debería crear las entradas de observaciones y acciones de forma dinámica
#         info = {
#             'observation': observation,
#             'action': action,
#             'reward': reward,
#             'temperature': temperature,
#             'power': power,
#             'timestep': timestep,
#             'simulation_time': simulation_time
#         }
#         func = getattr(self.logger, level)
#         func(msg, extra=info)