"""Class and utilities for set up extra configuration in experiments with Sinergym"""
import os
from opyplus import Epm, WeatherData, Idd
from sinergym.utils.common import get_record_keys, prepare_batch_from_records


class Config(object):
    """Config object to manage extra configuration in Sinergym experiments.

        :param _idf_path: IDF path origin for apply extra configuration.
        :param _weather_path: EPW path origin for apply weather to simulation.
        :param _ddy_path: DDY path origin for get DesignDays and weather Location
        :param config: Dict config with extra configuration which is required to modify IDF model (may be None)
        :param _idd: IDD opyplus object to set up Epm
        :param building: opyplus Epm object with IDF model
        :param ddy_model: opyplus Epm object with DDY model
        :param weather_data: opyplus WeatherData object with EPW data
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

    def adapt_idf_to_epw(self,
                         summerday: str = 'Afb Ann Clg .4% Condns DB=>MWB',
                         winterday: str = 'Afb Ann Htg 99.6% Condns DB'):
        """Given a summer day name and winter day name from DDY file, this method modify IDF Location and DesingDay's in order to adapt IDF to EPW.

        Args:
            summerday (str): Design day for summer day specifically (DDY has several of them).
            winterday (str): Design day for winter day specifically (DDY has several of them).
        """

        old_location = self.building.site_location[0]
        old_designdays = self.building.SizingPeriod_DesignDay

        # Adding the new location and designdays based on ddy file
        # LOCATION
        new_location = prepare_batch_from_records(
            [self.ddy_model.site_location[0]])
        # DESIGNDAYS
        winter_designday = self.ddy_model.SizingPeriod_DesignDay.one(
            lambda designday: winterday.lower() in designday.name.lower())
        summer_designday = self.ddy_model.SizingPeriod_DesignDay.one(
            lambda designday: summerday.lower() in designday.name.lower())
        new_designdays = prepare_batch_from_records(
            [winter_designday, summer_designday])

        # Deleting the old location and old DesignDays from Epm
        old_location.delete()
        old_designdays.delete()

        # Added New Location and DesignDays to Epm
        self.building.site_location.batch_add(new_location)
        self.building.SizingPeriod_DesignDay.batch_add(new_designdays)

    def save_building_model(self, working_dir_path: str = None):

        # If no path specified, then use idf_path to save it.
        if working_dir_path is None:
            new_idf_path = self._idf_path
        else:
            new_idf_path = self._idf_path.split('.idf')[0] + '_extra.idf'
            self.building.save(new_idf_path)
            self._idf_path = new_idf_path
