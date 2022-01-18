"""Class and utilities for set up extra configuration in experiments with Sinergym (extra params, weather_variability, building model modification and files management)"""
from copy import deepcopy
import os
from opyplus import Epm, WeatherData, Idd
from sinergym.utils.common import prepare_batch_from_records, get_delta_seconds
import numpy as np

WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
YEAR = 1991  # Non leap year


class Config(object):
    """Config object to manage extra configuration in Sinergym experiments.

        :param _idf_path: IDF path origin for apply extra configuration.
        :param _weather_path: EPW path origin for apply weather to simulation.
        :param _ddy_path: DDY path origin for get DesignDays and weather Location
        :param _env_working_dir_parent: Path for Sinergym experiment output
        :param _env_working_dir: Path for Sinergym specific episode (before first simulator reset this param is None)
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
            env_working_dir_parent,
            extra_config):

        self._idf_path = idf_path
        self._weather_path = weather_path
        # DDY path is deducible using weather_path (only change .epw by .ddy)
        self._ddy_path = self._weather_path.split('.epw')[0] + '.ddy'
        self._env_working_dir_parent = env_working_dir_parent
        self._env_working_dir = None

        self.config = extra_config

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

    # ---------------------------------------------------------------------------- #
    #                       IDF and Building model management                      #
    # ---------------------------------------------------------------------------- #

    def adapt_idf_to_epw(self,
                         summerday: str = 'Ann Clg .4% Condns DB=>MWB',
                         winterday: str = 'Ann Htg 99.6% Condns DB'):
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

    def apply_extra_conf(self):
        """Set extra configuration in building model
        """
        if self.config is not None:
            if self.config.get('timesteps_per_hour'):
                self.building.timestep[0].number_of_timesteps_per_hour = self.config['timesteps_per_hour']

    def save_building_model(self):
        """Take current building model and save as IDF in current env_working_dir episode folder.

        Returns:
            str: Path of IDF file stored (episode folder).
        """
        # If no path specified, then use idf_path to save it.
        if self._env_working_dir is not None:
            episode_idf_path = self._env_working_dir + \
                '/' + self._idf_path.split('/')[-1]
            self.building.save(episode_idf_path)
            return episode_idf_path
        else:
            raise Exception

    # ---------------------------------------------------------------------------- #
    #                        EPW and Weather Data management                       #
    # ---------------------------------------------------------------------------- #

    def apply_weather_variability(
            self,
            columns: list = ['drybulb'],
            variation: tuple = None):
        """Modify weather data using Ornstein-Uhlenbeck process.

        Args:
            columns (list, optional): List of columns to be affected. Defaults to ['drybulb'].
            variation (tuple, optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.

        Returns:
            str: New EPW file path generated in simulator working path in that episode
        """
        if variation is None:
            return self._weather_path
        else:
            # deepcopy for weather_data
            weather_data_mod = deepcopy(self.weather_data)
            # Get dataframe with weather series
            df = weather_data_mod.get_weather_series()

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
            weather_data_mod.set_weather_series(df)

            filename = self._weather_path.split('/')[-1]
            filename = filename.split('.epw')[0]
            filename += '_Random_%s_%s_%s.epw' % (
                str(sigma), str(mu), str(tau))
            episode_weather_path = self._env_working_dir + '/' + filename
            weather_data_mod.to_epw(episode_weather_path)
            return episode_weather_path

    # ---------------------------------------------------------------------------- #
    #                        Model and Config Functionality                        #
    # ---------------------------------------------------------------------------- #

    def _get_eplus_run_info(self):
        """This method read the building model from config and finds the running start month, start day, start year, end month, end day, end year, start weekday and the number of steps in a hour simulation. If any value is Unknown, then value will be 0. If step per hour is < 1, then default value will be 4.

        Returns:
            (int, int, int, int, int, int, int, int): A tuple with: the start month, start day, start year, end month, end day, end year, start weekday and number of steps in a hour simulation.
        """
        # Get runperiod object inner IDF
        runperiod = self.building.RunPeriod[0]

        start_month = int(
            0 if runperiod.begin_month is None else runperiod.begin_month)
        start_day = int(
            0 if runperiod.begin_day_of_month is None else runperiod.begin_day_of_month)
        start_year = int(
            0 if runperiod.begin_year is None else runperiod.begin_year)
        end_month = int(
            0 if runperiod.end_month is None else runperiod.end_month)
        end_day = int(
            0 if runperiod.end_day_of_month is None else runperiod.end_day_of_month)
        end_year = int(0 if runperiod.end_year is None else runperiod.end_year)
        start_weekday = WEEKDAY_ENCODING[runperiod.day_of_week_for_start_day.lower(
        )]
        n_steps_per_hour = self.building.timestep[0].number_of_timesteps_per_hour
        if n_steps_per_hour < 1 or n_steps_per_hour is None:
            n_steps_per_hour = 4  # default value

        return (
            start_month,
            start_day,
            start_year,
            end_month,
            end_day,
            end_year,
            start_weekday,
            n_steps_per_hour)

    def _get_one_epi_len(self):
        """Gets the length of one episode (an EnergyPlus process run to the end) depending on the config of simulation.

        Returns:
            int: The simulation time step in which the simulation ends.
        """
        # Get runperiod object inner IDF
        runperiod = self.building.RunPeriod[0]
        start_month = int(
            0 if runperiod.begin_month is None else runperiod.begin_month)
        start_day = int(
            0 if runperiod.begin_day_of_month is None else runperiod.begin_day_of_month)
        end_month = int(
            0 if runperiod.end_month is None else runperiod.end_month)
        end_day = int(
            0 if runperiod.end_day_of_month is None else runperiod.end_day_of_month)

        return get_delta_seconds(
            YEAR,
            start_month,
            start_day,
            end_month,
            end_day)

    def set_working_dir(self, new_env_working_dir: str):
        """Set env_working_dir attribute in conf for current episode, this method is called in simulator reset.

        Args:
            new_env_working_dir (str): New path working dir for new simulator episode.
        """
        # Create the Eplus working directory
        os.makedirs(new_env_working_dir)
        # Set attribute config
        self._env_working_dir = new_env_working_dir

    @property
    def start_year(self):
        """Returns the EnergyPlus simulation year.

        Returns:
            int: Simulation year.
        """

        return self.YEAR
