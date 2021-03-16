import os
import logging
import opyplus as op

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


def get_current_time_info(idf_file, sec_elapsed):
    """
    Returns the current day, month and hour given the seconds elapsed since the simulation started

    Parameters
    ----------
    idf file : IDF
        IDF file with simulation context information
    sec_elapsed : int
        seconds elapsed since the start of the simulation

    Return
    ----------
    tuple : (int, int, int)
        a tuple composed by the current day, month and hour in the simulation
    """

    epm = op.Epm.from_idf(idf_file)

    start_date = datetime(
        year = YEAR, # epm.RunPeriod[0]['start_year'],
        month = epm.RunPeriod[0]['begin_month'],
        day = epm.RunPeriod[0]['begin_day_of_month']
    )

    current_date = start_date + timedelta(seconds=sec_elapsed)

    return (current_date.day, current_date.month, current_date.hour)


def parse_observation(var_file, obs):
    """
    Parse observation to dictionary

    Parameters
    ----------
    var_file : string 
        variables file path
    obs : np.array
        observation vector with variables values

    Returns
    -------
    dict
        a dictionary with key = the name of the variable, 
        and value = value of the observed variable
    """

    tree = ET.parse(var_file)
    root = tree.getroot()

    variables = []

    for var in root.findall('variable'):
        if var.attrib['source'] == 'EnergyPlus':
            variables.append(var[0].attrib['type'])

    return dict(zip(variables, obs))
    

class Logger():  
    def getLogger(self, name, level, formatter):
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger