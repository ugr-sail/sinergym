import os
import numpy as np
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


YEAR = 1991
WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}


def get_delta_seconds(st_year, st_mon, st_day, end_mon, end_day):
    """
    Return the delta seconds between st_year:st_mon:st_day:0:0:0 and
    st_year:end_mon:end_day:24:0:0
    
    Args:
        st_year, st_mon, st_day, end_mon, end_day: int
            The start year, start month, start day, end month, end day.
            
    Return: float
        Time difference in seconds.
    """
    
    startTime = datetime(st_year, st_mon, st_day, 0, 0, 0)
    endTime = datetime(st_year, end_mon, end_day, 23, 0, 0) + \
              timedelta(0, 3600)
    delta_sec = (endTime - startTime).total_seconds()
    return delta_sec


def get_current_time_info(epm, sec_elapsed, sim_year = 1991):
    """
    Returns the current day, month and hour given the seconds elapsed since the simulation started

    Parameters
    ----------
    epm : Epm
        Energyplus model object from opyplus
    sec_elapsed : int
        seconds elapsed since the start of the simulation
    sim_year : int (optional)
        Year of the simulation

    Return
    ----------
    tuple : (int, int, int)
        a tuple composed by the current day, month and hour in the simulation
    """

    start_date = datetime(
        year = sim_year, # epm.RunPeriod[0]['start_year'],
        month = epm.RunPeriod[0]['begin_month'],
        day = epm.RunPeriod[0]['begin_day_of_month']
    )

    current_date = start_date + timedelta(seconds=sec_elapsed)

    return (current_date.day, current_date.month, current_date.hour)


def parse_variables(var_file):
    """
    Parse observation to dictionary

    Parameters
    ----------
    var_file : string 
        variables file path

    Returns
    -------
    list
        a list with the name of the variables
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables = []
    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            variables.append(var[0].attrib['type'])

    return variables


def create_variable_weather(weather_data, original_epw_file, columns: list = ['drybulb'], variation: tuple = None):
    """
    Create a new weather file adding gaussian noise to the original one.

    Parameters
    ----------
    weather_data : opyplus.WeatherData
        Opyplus object with the weather for the simulation
    original_epw_file : str
        Path to the original EPW file
    columns : list
        List of columns to be affected
    variation : tuple
        (mean, std) of the Gaussian noise

    Return
    ------
    str
        Name of the file created in the same location as the original one.
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