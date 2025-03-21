"""Class and utilities for backend modeling in Python with Sinergym (extra params, weather_variability, building model modification and files management)"""
import fcntl
import json
import os
import re
from copy import deepcopy
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from eppy.modeleditor import IDF
from epw.weather import Weather

from sinergym.utils.common import (
    eppy_element_to_dict,
    get_delta_seconds,
    ornstein_uhlenbeck_process,
)
from sinergym.utils.constants import (
    CWD,
    LOG_MODEL_LEVEL,
    PKG_DATA_PATH,
    WEEKDAY_ENCODING,
    YEAR,
)
from sinergym.utils.logger import TerminalLogger


class ModelJSON(object):
    """Class to manage backend models (building, weathers...) and folders in Sinergym (JSON version).

        :param _json_path: JSON path origin to create the building model.
        :param weather_files: Available weather files for each episode.
        :param _weather_path: EPW path origin for apply weather to simulation in current episode.
        :param _ddy_path: DDY path origin for get DesignDays and weather Location.
        :param _idd: IDD eppy object to set up Epm.
        :param _variables: Output:Variable(s) information about building model.
        :param _meters: Output:Meter(s) information about building model.
        :param experiment_path: Path for Sinergym experiment output.
        :param episode_path: Path for Sinergym specific episode (before first simulator reset this param is None).
        :param max_ep_store: Number of episodes directories will be stored in experiment_path.
        :param config: Dict config with extra configuration which is required to modify building model (may be None).
        :param building: Building model (Dictionary extracted from JSON).
        :param ddy_model: eppy object with DDY model.
        :param weather_data: epw module Weather class instance with EPW data.
        :param zone_names: List of the zone names available in the building.
        :param schedulers: Information in Dict format about all building schedulers.
        :param runperiod: Information in Dict format about runperiod that determine an episode.
        :param episode_length: Time in seconds that an episode has.
        :param step_size: Time in seconds that an step has.
        :param timestep_per_episode: Timestep in a runperiod (simulation episode).
    """

    logger = TerminalLogger().getLogger(
        name='MODEL',
        level=LOG_MODEL_LEVEL)

    def __init__(
            self,
            env_name: str,
            json_file: str,
            weather_files: List[str],
            variables: Dict[str, Tuple[str, str]],
            meters: Dict[str, str],
            max_ep_store: int,
            extra_config: Dict[str, Any]):
        """Constructor. Variables and meters are required to update building model scheme.

        Args:
            env_name (str): Name of the environment, required for Sinergym output management.
            json_file (str): Json file name, path is calculated by the constructor.
            weather_files (List[str]): List of the weather file names, one of them will be select randomly, path will be calculated by the constructor.
            variables (Dict[str, Tuple[str, str]]): Specification for EnergyPlus Output:Variable. The key name is custom, then tuple must be the original variable name and the output variable key.
            meters (Dict[str, str]): Specification for EnergyPlus Output:Meter. The key name is custom, then value is the original EnergyPlus Meters name.
            max_ep_store (int): Number of episodes directories will be stored in experiment_path.
            extra_config (Dict[str, Any]): Dict config with extra configuration which is required to modify building model (may be None).
        """
        self.pkg_data_path = PKG_DATA_PATH
        # ----------------------- Transform filenames in paths ----------------------- #

        # JSON
        self._json_path = os.path.join(
            self.pkg_data_path, 'buildings', json_file)

        # EPW
        self.weather_files = weather_files

        # IDD
        self._idd = os.path.join(os.environ['EPLUS_PATH'], 'Energy+.idd')

        # Select one weather randomly (if there are more than one)
        self._weather_path = os.path.join(
            self.pkg_data_path, 'weather', np.random.choice(
                self.weather_files))

        # DDY path is deducible using weather_path (only change .epw by .ddy)
        self._ddy_path = self._weather_path.split('.epw')[0] + '.ddy'

        # -------------------------------- File Models ------------------------------- #

        # Building model object (Python dictionary from epJSON file)
        with open(self._json_path) as json_f:
            self.building = json.load(json_f)

        # DDY model (eppy object)
        IDF.setiddname(self._idd)
        self.ddy_model = IDF(self._ddy_path)

        # Weather data (epw.weather object)
        self.weather_data = Weather()
        self.weather_data.read(self._weather_path)
        # Weather variability if exists
        self.weather_variability_config = None

        # ----------------------------- Other attributes ----------------------------- #

        # Output paths and config
        self.experiment_path = self._set_experiment_working_dir(env_name)
        self.episode_path: Optional[str] = None
        self.max_ep_store = max_ep_store
        self.config = extra_config

        # Input/Output varibles
        self._variables = variables
        self._meters = meters

        # Extract building zones
        self.zone_names = list(self.building['Zone'].keys())
        # Extract schedulers available in building model
        self.schedulers = self.get_schedulers()

        # ------------------------ Checking config definition ------------------------ #

        # Check config definition
        self._check_eplus_config()
        self.logger.info('Model Config is correct.')

        # ------------- Apply adaptations in building model automatically ------------ #
        self.adapt_building_to_variables()
        self.adapt_building_to_meters()
        self.adapt_building_to_config()

        # Runperiod information
        self.runperiod = self.get_eplus_runperiod()
        self.episode_length = self.get_runperiod_len()
        self.step_size = 3600 / self.runperiod['n_steps_per_hour']
        self.timestep_per_episode = int(
            self.episode_length / self.step_size)

        self.logger.info('Runperiod established.')
        self.logger.debug(f'Runperiod: {self.runperiod}')
        self.logger.info(f'Episode length (seconds): {self.episode_length}')
        self.logger.info(f'timestep size (seconds): {self.step_size}')
        self.logger.info(f'timesteps per episode: {self.timestep_per_episode}')

    # ---------------------------------------------------------------------------- #
    #                 Variables and Building model adaptation                      #
    # ---------------------------------------------------------------------------- #

    def adapt_building_to_epw(
            self,
            summerday: str = 'Ann Clg .4% Condns DB=>MWB',
            winterday: str = 'Ann Htg 99.6% Condns DB') -> None:
        """Given a summer day name and winter day name from DDY file, this method modify Location and DesingDay's in order to adapt building model to EPW.

        Args:
            summerday (str): Design day for summer day specifically (DDY has several of them).
            winterday (str): Design day for winter day specifically (DDY has several of them).
        """

        # Getting the new location and designdays based on ddy file (Records
        # must be converted to dictionary)

        # LOCATION
        self.building['Site:Location'] = eppy_element_to_dict(
            next(location for location in self.ddy_model.idfobjects['Site:Location']))

        # DESIGNDAYS
        ddy_designdays = self.ddy_model.idfobjects['SizingPeriod:DesignDay']

        try:
            summer_designday = next(
                dd for dd in ddy_designdays if summerday in dd.Name)
            winter_designday = next(
                dd for dd in ddy_designdays if winterday in dd.Name)
        except StopIteration:
            self.logger.error(
                f"Design day not found: Summer='{summerday}', Winter='{winterday}'")
            raise ValueError

        self.building['SizingPeriod:DesignDay'] = {
            **eppy_element_to_dict(winter_designday),
            **eppy_element_to_dict(summer_designday)
        }

        self.logger.info('Adapting weather to building model.')
        self.logger.debug(f'Weather path: {self._weather_path.split('/')[-1]}')

    def adapt_building_to_variables(self) -> None:
        """Replaces the default Output:Variable entries in the building model with custom variables.

        This method removes all default Output:Variable entries and adds new ones based on
        the variables stored in `self._variables`. Each variable is assigned a key, a name,
        and a reporting frequency of 'Timestep'.
        """

        # Remove existing Output:Variables
        self.building['Output:Variable'] = {}

        # Construct and assign new Output:Variable dictionary
        self.building['Output:Variable'] = {
            f'Output:Variable {i}': {
                'key_value': variable_key,
                'variable_name': variable_name,
                'reporting_frequency': 'Timestep'} for i,
            (variable_name,
             variable_key) in enumerate(
                self._variables.values(),
                start=1)}

        self.logger.info(
            'Building model Output:Variable updated with defined variable names.')

    def adapt_building_to_meters(self) -> None:
        """Reads all meters and updates the building model with Output:Meter fields.
        """

        # Remove existing Output:Meters
        self.building['Output:Meter'] = {}

        # Construct and assign new Output:Meter dictionary
        self.building['Output:Meter'] = {
            f'Output:Meter {i}': {
                'key_name': meter_name,
                'reporting_frequency': 'Timestep'} for i,
            meter_name in enumerate(self._meters.values(), start=1)
        }

        self.logger.info(
            'Updated building model Output:Meter with meter names.')

    def adapt_building_to_config(self) -> None:
        """Set extra configuration in building model
        """

        if not self.config:
            return

        # Timesteps processed in a simulation hour
        timesteps = self.config.get('timesteps_per_hour')
        if timesteps:
            next(iter(self.building['Timestep'].values()), {})[
                'number_of_timesteps_per_hour'] = self.config['timesteps_per_hour']

            self.logger.debug(
                f'Extra config: timesteps_per_hour set up to {
                    self.config['timesteps_per_hour']}')

        # Runperiod datetimes --> Tuple(start_day, start_month, start_year,
        # end_day, end_month, end_year)
        runperiod = self.config.get('runperiod')
        if runperiod:
            next(iter(self.building['RunPeriod'].values()), {}).update({
                'begin_day_of_month': int(runperiod[0]),
                'begin_month': int(runperiod[1]),
                'begin_year': int(runperiod[2]),
                'end_day_of_month': int(runperiod[3]),
                'end_month': int(runperiod[4]),
                'end_year': int(runperiod[5])
            })

            # Update runperiod and episode related attributes
            self.runperiod = self.get_eplus_runperiod()
            self.episode_length = self.get_runperiod_len()
            self.step_size = 3600 / self.runperiod['n_steps_per_hour']
            self.timestep_per_episode = int(
                self.episode_length / self.step_size)

            # Log updated values in terminal
            self.logger.info(
                f'Extra config: runperiod updated to {self.runperiod}')
            self.logger.info(
                f'Updated episode length (seconds): {self.episode_length}')
            self.logger.info(
                f'Updated timestep size (seconds): {self.step_size}')
            self.logger.info(
                f'Updated timesteps per episode: {
                    self.timestep_per_episode}')

    def save_building_model(self) -> str:
        """Take current building model and save as epJSON in current episode path folder.

        Returns:
            str: Path of epJSON file stored (episode folder).
        """

        # If no path specified, then use json_path to save it.
        if not self.episode_path:
            self.logger.error(
                'Episode path should be set before saving building model.')
            raise RuntimeError

        episode_json_path = os.path.join(self.episode_path,
                                         os.path.basename(self._json_path))

        try:
            with open(episode_json_path, 'w') as outfile:
                json.dump(self.building, outfile, indent=4)
            self.logger.debug(f'Building model saved at: {episode_json_path}')

        except OSError:
            self.logger.error(
                f'Failed to save building model: {episode_json_path}')
            raise OSError

        return episode_json_path

    # ---------------------------------------------------------------------------- #
    #                        EPW and Weather Data management                       #
    # ---------------------------------------------------------------------------- #

    def update_weather_path(self) -> None:
        """When this method is called, weather file is changed randomly and building model is adapted to new one.
        """

        if not self.weather_files:
            self.logger.error('No weather files available to choose from.')
            raise ValueError

        # Select a new weather file randomly
        self._weather_path = os.path.join(
            self.pkg_data_path, 'weather', np.random.choice(
                self.weather_files))

        # Verify file exists
        if not os.path.isfile(self._weather_path):
            self.logger.error(
                f'Weather file {self._weather_path} does not exist.')
            raise FileNotFoundError

        # Update ddy path for the same than weather
        self._ddy_path = os.path.splitext(self._weather_path)[0] + '.ddy'

        # Load new DDY and Weather instances
        self.ddy_model = IDF(self._ddy_path)
        self.weather_data.read(self._weather_path)

        self.logger.info(
            f'Weather file {self._weather_path.split('/')[-1]} used.')

    def apply_weather_variability(
            self,
            weather_variability: Optional[Dict[str, Tuple[
            Union[float, Tuple[float, float]],
            Union[float, Tuple[float, float]],
            Union[float, Tuple[float, float]]
            ]]] = None) -> str:
        """Modify weather data using Ornstein-Uhlenbeck process according to the variation specified in the weather_variability dictionary.

        Args:
            weather_variability (Optional[Dict[str,Tuple[Union[float,Tuple[float,float]],Union[float,Tuple[float,float]],Union[float,Tuple[float,float]]]]]): Dictionary with the variation for each column in the weather data. Defaults to None. The key is the column name and the value is a tuple with the sigma, mean and tau for OU process.

        Returns:
            str: New EPW file path generated in simulator working path in that episode or current EPW path if variation is not defined.
        """

        base_filename, ext = os.path.splitext(
            os.path.basename(self._weather_path))
        weather_data_mod = deepcopy(self.weather_data)

        # Apply variation to EPW if exists
        if weather_variability:

            # Generate variability configuration
            self.weather_variability_config = {
                weather_var: tuple(
                    np.random.uniform(param[0], param[1]) if isinstance(param, tuple) else param
                    for param in params
                )
                for weather_var, params in weather_variability.items()
            }

            # Apply Ornstein-Uhlenbeck process to weather data
            weather_data_mod.dataframe = ornstein_uhlenbeck_process(
                data=self.weather_data.dataframe,
                variability_config=self.weather_variability_config)

            self.logger.info(
                f'Weather noise applied to columns: {
                    list(self.weather_variability_config.keys())}')

            # Modify filename to reflect noise addition
            base_filename += '_OU_Noise'

        # Define output path
        episode_weather_path = os.path.join(
            self.episode_path, f'{base_filename}{ext}')
        # Write new weather file
        weather_data_mod.write(episode_weather_path)

        self.logger.debug(
            f'Saved modified weather file: {episode_weather_path}')

        return episode_weather_path

# ---------------------------------------------------------------------------- #
#                          Schedulers info extraction                          #
# ---------------------------------------------------------------------------- #

    def get_schedulers(self) -> Dict[str,
                                     Dict[str, Union[str, Dict[str, str]]]]:
        """Extract all schedulers available in the building model to be controlled.

        Returns:
            Dict[str, Dict[str, Any]]: Python Dictionary: For each scheduler found, it shows type value and where this scheduler is present (table, object and field).
        """
        result = {}
        schedules = {}
        # Mount a dict with only building schedulers
        if 'Schedule:Compact' in self.building:
            schedules.update(self.building['Schedule:Compact'])
        if 'Schedule:Year' in self.building:
            schedules.update(self.building['Schedule:Year'])

        for sch_name, sch_info in schedules.items():
            # Write sch_name and data type in output
            result[sch_name] = {
                'Type': sch_info['schedule_type_limits_name'],
            }
            # We are going to search where that scheduler appears in whole
            # building model
            for table, elements in self.building.items():
                # Don't include themselves
                if table != 'Schedule:Compact' and table != 'Schedule:Year':
                    # For each object of a table type
                    for element_name, element_fields in elements.items():
                        # For each field in a object
                        for field_key, field_value in element_fields.items():
                            # If a field value is the schedule name
                            if field_value == sch_name:
                                # We annotate the object name as key and the
                                # field name where sch name appears and the
                                # table where belong to
                                result[sch_name][element_name] = {
                                    'field_name': field_key,
                                    'table_name': table
                                }
        return result

# ---------------------------------------------------------------------------- #
#                           Runperiod info extraction                          #
# ---------------------------------------------------------------------------- #

    def get_eplus_runperiod(
            self) -> Dict[str, int]:
        """This method reads building runperiod information and returns it.

        Returns:
            Dict[str,int]: A Dict with: the start month, start day, start year, end month, end day, end year, start weekday and number of steps in a hour simulation.
        """
        # Get runperiod object inner building model
        runperiod = next(iter(self.building['RunPeriod'].values()), {})

        # Extract information about runperiod
        start_month = runperiod.get('begin_month', 0)
        start_day = runperiod.get('begin_day_of_month', 0)
        start_year = runperiod.get('begin_year', YEAR)
        end_month = runperiod.get('end_month', 0)
        end_day = runperiod.get('end_day_of_month', 0)
        end_year = runperiod.get('end_year', YEAR)

        start_weekday = WEEKDAY_ENCODING[runperiod['day_of_week_for_start_day'].lower(
        )]

        n_steps_per_hour = list(self.building['Timestep'].values())[
            0]['number_of_timesteps_per_hour']

        return {
            'start_day': start_day,
            'start_month': start_month,
            'start_year': start_year,
            'end_day': end_day,
            'end_month': end_month,
            'end_year': end_year,
            'start_weekday': start_weekday,
            'n_steps_per_hour': n_steps_per_hour}

    def get_runperiod_len(self) -> float:
        """Gets the length of runperiod (an EnergyPlus process run to the end) depending on the config of simulation.

        Returns:
            float: The simulation time in which the simulation ends (seconds).
        """
        # Get runperiod object inner building model

        return get_delta_seconds(
            self.runperiod['start_year'],
            self.runperiod['start_month'],
            self.runperiod['start_day'],
            self.runperiod['end_year'],
            self.runperiod['end_month'],
            self.runperiod['end_day'])

    # ---------------------------------------------------------------------------- #
    #                  Working Folder for Simulation Management                    #
    # ---------------------------------------------------------------------------- #

    def _set_experiment_working_dir(self, env_name: str) -> str:
        """Set experiment working dir path like config attribute for current simulation.

        Args:
            env_name (str): simulation env name to define a name in directory

        Returns:
            str: Experiment path for directory created.
        """
        # lock file for paralell execution
        lock_file = os.path.join(CWD, '.lock')

        # CRITICAL SECTION: Avoid race conditions when generating the directory
        with open(lock_file, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Generate experiment_path
                experiment_path = self._get_working_folder(
                    directory_path=CWD,
                    base_name=f'{env_name}-res'
                )

                # Create directory
                os.makedirs(experiment_path)

            finally:
                # Release the lock
                fcntl.flock(f, fcntl.LOCK_UN)

        # Set path as an instance attribute
        self.experiment_path = experiment_path

        self.logger.info(
            'Experiment working directory created.')
        self.logger.info(
            f'Working directory: {experiment_path}')

        return experiment_path

    def _get_working_folder(
            self,
            directory_path: str,
            base_name: str = 'sinergym-run') -> str:
        """Create a working folder path from path_folder using base_name, returning the absolute result path.
           Assumes folders in parent_dir have suffix <env_name>-run{run_number}. Finds the highest run number and sets the output folder to that number + 1.

        Args:
            path_folder (str): Path when working dir will be created.
            base_name (str, optional): Base name used to name the new folder inner path_folder. Defaults to '-run'.

        Returns:
            str: Path to the working directory.

        """

        # Ensure directory exists
        os.makedirs(directory_path, exist_ok=True)

        # Regular expression to match folders with the base_name followed by a
        # number
        pattern = re.compile(rf'^{re.escape(base_name)}(\d+)$')

        # Extract valid numbers from existing folder names
        existing_numbers = [int(match.group(1)) for folder in os.listdir(directory_path) if (
            match := pattern.match(folder)) and os.path.isdir(os.path.join(directory_path, folder))]

        # Determine the next experiment ID
        experiment_id = max(existing_numbers, default=0) + 1

        working_dir = os.path.join(
            directory_path, f'{base_name}{experiment_id}')

        return working_dir

    def set_episode_working_dir(self) -> str:
        """Set episode working dir path like config attribute for current simulation execution.

        Raises:
            Exception: If experiment path (parent folder) has not be created previously.

        Returns:
            str: Episode path for directory created.
        """
        # Generate episode dir path if experiment dir path has been created
        # previously
        if self.experiment_path is None:
            self.logger.error('Experiment path is not specified.')
            raise Exception
        else:
            episode_path = self._get_working_folder(
                directory_path=self.experiment_path,
                base_name='episode-')
            # Create directory
            os.makedirs(episode_path)
            # set path like config attribute
            self.episode_path = episode_path

            # Remove redundant past working directories
            self._rm_past_history_dir(episode_path, 'episode-')

            self.logger.info(
                'Episode directory created.')
            self.logger.debug(
                f'Episode directory path: {episode_path}')

            return episode_path

    def _rm_past_history_dir(self, episode_path: str, base_name: str) -> None:
        """Removes old simulation result directories beyond the max storage limit.

        Args:
            episode_path (str): Path of the current episode output.
            base_name (str): Base name used to detect episode output IDs.
        """
        # Extract directory ID safely using regex
        match = re.search(rf"{re.escape(base_name)}(\d+)$", episode_path)
        if not match:
            self.logger.error(
                f"Could not extract episode ID from: {episode_path}")
            raise ValueError

        try:
            cur_dir_id = int(match.group(1))
        except ValueError:
            self.logger.error(
                f"Invalid episode ID extracted from: {episode_path}")
            raise ValueError

        # Compute the directory ID to remove
        rm_dir_id = cur_dir_id - self.max_ep_store
        if rm_dir_id <= 0:
            return  # Nothing to remove

        # Construct full path of the directory to remove
        rm_dir_full_name = os.path.join(
            os.path.dirname(episode_path),
            f"{base_name}{rm_dir_id}")

        # Safely remove only if the directory exists
        if os.path.exists(rm_dir_full_name) and os.path.isdir(
                rm_dir_full_name):
            rmtree(rm_dir_full_name)
            self.logger.debug(
                f"Deleted old episode directory: {rm_dir_full_name}")
        else:
            self.logger.warning(
                f"No old episode directory found to delete: {rm_dir_full_name}")

    # ---------------------------------------------------------------------------- #
    #                             Model class checker                              #
    # ---------------------------------------------------------------------------- #

    def _check_eplus_config(self) -> None:
        """Check Eplus Environment config definition is correct.
        """

        # COMMON
        # Check weather files exist
        for w_file in self.weather_files:
            w_path = os.path.join(
                self.pkg_data_path, 'weather', w_file)
            if not os.path.isfile(w_path):
                self.logger.critical(
                    f'Weather files: {w_file} is not a weather file available in Sinergym.')
                raise FileNotFoundError

        # EXTRA CONFIG
        if self.config is not None:
            for config_key in self.config.keys():
                # Check config parameters values
                # Timesteps
                if config_key == 'timesteps_per_hour':
                    if self.config[config_key] < 1:
                        self.logger.critical(
                            f'Extra Config: timestep_per_hour must be a positive int value, the value specified is {
                                self.config[config_key]}')
                        raise ValueError
                # Runperiod
                elif config_key == 'runperiod':
                    if not isinstance(self.config[config_key], tuple):
                        self.logger.critical(
                            f'Extra Config: Runperiod specified in extra configuration must be a tuple (type detected {
                                type(
                                    self.config[config_key])})')
                        raise TypeError
                    if len(self.config[config_key]) != 6:
                        self.logger.critical(
                            'Extra Config: Runperiod specified in extra configuration must have 6 elements.')
                        raise ValueError
                else:
                    self.logger.error(
                        f'Extra Config: Key name specified in config called [{config_key}] is not available in Sinergym, it will be ignored.')

    # ---------------------------------------------------------------------------- #
    #                                  Properties                                  #
    # ---------------------------------------------------------------------------- #

    @property  # pragma: no cover
    def building_path(self) -> str:
        return self._json_path

    @property  # pragma: no cover
    def weather_path(self) -> str:
        return self._weather_path

    @property  # pragma: no cover
    def ddy_path(self) -> Optional[str]:
        return self._ddy_path

    @property  # pragma: no cover
    def idd_path(self) -> Optional[str]:
        return self._idd
