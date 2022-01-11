"""Class and utilities for set up extra configuration in experiments with Sinergym"""
import os
from opyplus import Epm, WeatherData, Idd


class Config(object):
    """Config object to manage extra configuration in Sinergym experiments.

        :param _idf_path: IDF path origin for apply extra configuration.
        :param _timesteps_per_hour: Timesteps generated in a simulation hour.
        :param building: opyplus object to read/modify idf building.
    """

    def __init__(
            self,
            idf_path,
            **kwargs):

        self._idf_path = idf_path
        self.config = kwargs
        idd = Idd(os.path.join(os.environ['EPLUS_PATH'], 'Energy+.idd'))
        self.building = Epm.from_idf(
            self._idf_path,
            idd_or_version=idd,
            check_length=False)

    def set_conf(self):
        """Set configuration and store idf in a new path.
        """
        if self.config.get('timesteps_per_hour'):
            self.building.timestep[0].number_of_timesteps_per_hour = self.config['timesteps_per_hour']

        new_idf_path = self._idf_path.split('.idf')[0] + '_extra.idf'
        self.building.save(new_idf_path)
        self._idf_path = new_idf_path
