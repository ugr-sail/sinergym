import logging
import datetime


MON_DAYS_MAP = {1:31, 2:28, 3:31,
               4:30, 5:31, 6:30,
               7:31, 8:31, 9:30,
               10:31, 11:30, 12:31};

WEEKDAY_ENCODING = {'monday':0, 'tuesday':1, 'wednesday':2, 'thursday':3,
                    'friday':4, 'saturday': 5, 'sunday':6}


def get_hours_to_now(month, day):
    """
    The function returns the number of hours by the start of the month:day.
    
    Args:
        month: int
            Month.
    
    Return: int
        The hours from Jan 1st 00:00:00 to the start of this month:day.
    """
    ret = 0
    for mon in range(1, month):
        ret += 24 * MON_DAYS_MAP[mon]
    ret += 24 * (day - 1)
    return ret


def get_time_string(start_year, start_mon, start_day, timestep_second):
    """"""
    startTime = datetime.datetime(start_year, start_mon, start_day, 0, 0, 0)
    retTime = startTime + datetime.timedelta(0, timestep_second)
    return str(retTime)


def get_delta_seconds(st_year, st_mon, st_day, ed_mon, ed_day):
    """
    Return the delta seconds between st_year:st_mon:st_day:0:0:0 and
    st_year:ed_mon:ed_day:24:0:0
    
    Args:
        st_year, st_mon, st_day, ed_mon, ed_day: int
            The start year, start month, start day, end month, end day.
            
    Return: float
        Time difference in seconds.
    """
    
    startTime = datetime.datetime(st_year, st_mon, st_day, 0, 0, 0)
    endTime = datetime.datetime(st_year, ed_mon, ed_day, 23, 0, 0) + \
              datetime.timedelta(0, 3600)
    delta_sec = (endTime - startTime).total_seconds()
    return delta_sec


def getSecondFromStartOfYear(nowDateTime):
    """
    Get the corresponding seconds from the start of the year.
    Args:
        nowDateTime: Python datatime object
    Return: int
    """
    startDateTime = nowDateTime.replace(month = 1, day = 1, hour = 0, minute = 0, second = 0, microsecond = 0)
    return int((nowDateTime - startDateTime).total_seconds())


class Logger():
    
    def getLogger(self, name, level, formatter):
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger