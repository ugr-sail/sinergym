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
            weather_path,
            **kwargs):

        self._idf_path = idf_path
        self._weather_path = weather_path
        # DDY path is deducible using weather_path (only change .epw by .ddy)
        self._ddy_path = self._weather_path.split('.epw')[0] + '.ddy'
        self.config = kwargs

        # Opyplus objects
        self._idd = Idd(os.path.join(os.environ['EPLUS_PATH'], 'Energy+.idd'))
        self.building = Epm.from_idf(
            self._idf_path,
            idd_or_version=self._idd,
            check_length=False)
        self.ddy_model = Epm.from_idf(
            self._ddy_path,
            idd_or_version=self._idd,
            check_length=False)
        self.weather_data = WeatherData.from_epw(self._weather_path)

    def set_extra_conf(self):
        """Set configuration and store idf in a new path.
        """
        if self.config.get('timesteps_per_hour'):
            self.building.timestep[0].number_of_timesteps_per_hour = self.config['timesteps_per_hour']

    def save_building_model(self, working_dir_path: str = None):

        # If no path specified, then use idf_path to save it.
        if working_dir_path is None:
            new_idf_path = self._idf_path
        else:
            new_idf_path = self._idf_path.split('.idf')[0] + '_extra.idf'
            self.building.save(new_idf_path)
            self._idf_path = new_idf_path
