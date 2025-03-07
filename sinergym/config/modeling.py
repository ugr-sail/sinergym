"""Class and utilities for backend modeling in Python with Sinergym (extra params, weather_variability, building model modification and files management)"""
import fcntl
import json
import os
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
        self.runperiod = self._get_eplus_runperiod()
        self.episode_length = self._get_runperiod_len()
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
        new_location = self.ddy_model.idfobjects['Site:Location'][0]
        new_location = eppy_element_to_dict(new_location)

        # DESIGNDAYS
        ddy_designdays = self.ddy_model.idfobjects['SizingPeriod:DesignDay']
        summer_designdays = list(
            filter(
                lambda designday: summerday in designday.Name,
                ddy_designdays))[0]
        winter_designdays = list(
            filter(
                lambda designday: winterday in designday.Name,
                ddy_designdays))[0]
        new_designdays = {}
        new_designdays.update(eppy_element_to_dict(winter_designdays))
        new_designdays.update(eppy_element_to_dict(summer_designdays))

        # Addeding new location and DesignDays to Building model
        self.building['Site:Location'] = new_location
        self.building['SizingPeriod:DesignDay'] = new_designdays

        self.logger.info('Adapting weather to building model.')
        self.logger.debug(f'Weather path: {self._weather_path.split('/')[-1]}')

    def adapt_building_to_variables(self) -> None:
        """This method reads all variables and write it in the building model as Output:Variable field.
        """
        output_variables = {}
        for i, (variable_name, variable_key) in enumerate(
                list(self._variables.values()), start=1):

            # Add element Output:Variable to the building model
            output_variables['Output:Variable ' + str(i)] = {'key_value': variable_key,
                                                             'variable_name': variable_name,
                                                             'reporting_frequency': 'Timestep'}

        self.logger.info(
            'Update building model Output:Variable with variable names.')

        # Delete default Output:Variables and added whole building variables to
        # Output:Variable field
        self.building['Output:Variable'] = output_variables

    def adapt_building_to_meters(self) -> None:
        """This method reads all meters and write it in the building model as Output:Meter field.
        """
        output_meters = {}
        for i, meter_name in enumerate(
                list(self._meters.values()), start=1):

            # Add element Output:Variable to the building model
            output_meters['Output:Meter ' +
                          str(i)] = {'key_name': meter_name, 'reporting_frequency': 'Timestep'}

        self.logger.info(
            'Update building model Output:Meter with meter names.')

        # Delete default Output:Variables and added whole building variables to
        # Output:Variable field
        self.building['Output:Meter'] = output_meters

    def adapt_building_to_config(self) -> None:
        """Set extra configuration in building model
        """
        if self.config is not None:

            # Timesteps processed in a simulation hour
            if self.config.get('timesteps_per_hour'):
                list(self.building['Timestep'].values())[
                    0]['number_of_timesteps_per_hour'] = self.config['timesteps_per_hour']

                self.logger.debug(
                    f'Extra config: timesteps_per_hour set up to {
                        self.config['timesteps_per_hour']}')

            # Runperiod datetimes --> Tuple(start_day, start_month, start_year,
            # end_day, end_month, end_year)
            if self.config.get('runperiod'):
                runperiod = list(self.building['RunPeriod'].values())[0]
                runperiod['begin_day_of_month'] = int(
                    self.config['runperiod'][0])
                runperiod['begin_month'] = int(self.config['runperiod'][1])
                runperiod['begin_year'] = int(self.config['runperiod'][2])
                runperiod['end_day_of_month'] = int(
                    self.config['runperiod'][3])
                runperiod['end_month'] = int(self.config['runperiod'][4])
                runperiod['end_year'] = int(self.config['runperiod'][5])

                # Update runperiod and episode related attributes
                self.runperiod = self._get_eplus_runperiod()
                self.episode_length = self._get_runperiod_len()
                self.step_size = 3600 / self.runperiod['n_steps_per_hour']
                self.timestep_per_episode = int(
                    self.episode_length / self.step_size)
                self.logger.info(
                    f'Extra config: runperiod updated to {runperiod}')
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
        if self.episode_path is not None:
            episode_json_path = os.path.join(self.episode_path,
                                             os.path.basename(self._json_path))
            with open(episode_json_path, "w") as outfile:
                json.dump(self.building, outfile, indent=4)

            self.logger.debug(
                f'Saving episode building model... [{episode_json_path}]')

            return episode_json_path
        else:
            self.logger.error(
                'Episode path should be set before saving building model.')
            raise RuntimeError

    # ---------------------------------------------------------------------------- #
    #                        EPW and Weather Data management                       #
    # ---------------------------------------------------------------------------- #

    def update_weather_path(self) -> None:
        """When this method is called, weather file is changed randomly and building model is adapted to new one.
        """
        self._weather_path = os.path.join(
            self.pkg_data_path, 'weather', np.random.choice(
                self.weather_files))
        self._ddy_path = self._weather_path.split('.epw')[0] + '.ddy'
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

        filename = self._weather_path.split('/')[-1]
        weather_data_mod = deepcopy(self.weather_data)

        # Apply variation to EPW if exists
        if weather_variability is not None:

            # Check if there are ranges specified in params and get a random
            # value
            self.weather_variability_config = {
                weather_var: tuple(
                    np.random.uniform(
                        param[0], param[1]) if isinstance(
                        param, tuple) else param
                    for param in params
                )
                for weather_var, params in weather_variability.items()
            }

            # Apply Ornstein-Uhlenbeck process to weather data
            weather_data_mod.dataframe = ornstein_uhlenbeck_process(
                data=self.weather_data.dataframe,
                variability_config=self.weather_variability_config)

            self.logger.info(
                f'Weather noise applied in columns: {
                    list(
                        self.weather_variability_config.keys())}')

            # Modify filename to reflect noise addition
            filename = f"{filename.split('.epw')[0]}_OU_Noise.epw"

        episode_weather_path = self.episode_path + '/' + filename
        weather_data_mod.write(episode_weather_path)

        self.logger.debug(
            f'Saving episode weather path... [{episode_weather_path}]')

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

    def _get_eplus_runperiod(
            self) -> Dict[str, int]:
        """This method reads building runperiod information and returns it.

        Returns:
            Dict[str,int]: A Dict with: the start month, start day, start year, end month, end day, end year, start weekday and number of steps in a hour simulation.
        """
        # Get runperiod object inner building model
        runperiod = list(self.building['RunPeriod'].values())[0]

        # Extract information about runperiod
        start_month = int(
            0 if runperiod['begin_month'] is None else runperiod['begin_month'])
        start_day = int(
            0 if runperiod['begin_day_of_month'] is None else runperiod['begin_day_of_month'])
        start_year = int(
            YEAR if runperiod['begin_year'] is None else runperiod['begin_year'])
        end_month = int(
            0 if runperiod['end_month'] is None else runperiod['end_month'])
        end_day = int(0 if runperiod['end_day_of_month']
                      is None else runperiod['end_day_of_month'])
        end_year = int(
            YEAR if runperiod['end_year'] is None else runperiod['end_year'])
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

    def _get_runperiod_len(self) -> float:
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

    def _set_experiment_working_dir(self, env_name: str) -> str:
        """Set experiment working dir path like config attribute for current simulation.

        Args:
            env_name (str): simulation env name to define a name in directory

        Returns:
            str: Experiment path for directory created.
        """
        # CRITICAL SECTION for paralell execution
        # In order to avoid several folders with the same name
        with open(CWD + '/.lock', 'w') as f:
            # Exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                # Generate experiment dir path
                experiment_path = self._get_working_folder(
                    directory_path=CWD,
                    base_name='%s-res' %
                    (env_name))
                # Create dir
                os.makedirs(experiment_path)
            finally:
                # Release lock
                fcntl.flock(f, fcntl.LOCK_UN)
        # Set path like config attribute
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

        # Create de rute if not exists
        os.makedirs(directory_path, exist_ok=True)
        experiment_id = 0
        # Iterate all elements inner path
        for folder_name in os.listdir(directory_path):
            if not os.path.isdir(os.path.join(directory_path, folder_name)):
                continue
            try:
                # Detect if there is another directory with the same base_name
                # and get number at final of the name
                folder_name = int(folder_name.split(base_name)[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except BaseException:
                pass
        # experiment_id number will be +1 from last name found out.
        experiment_id += 1

        working_dir = os.path.join(
            directory_path, '%s%d' %
            (base_name, experiment_id))
        return working_dir

    def _rm_past_history_dir(
            self,
            episode_path: str,
            base_name: str) -> None:
        """Removes the past simulation results from episode

        Args:
            episode_path (str): path for the current episide output
            base_name (str): base name for detect episode output id
        """

        cur_dir_name, cur_dir_id = episode_path.split(base_name)
        cur_dir_id = int(cur_dir_id)
        if cur_dir_id - self.max_ep_store > 0:
            rm_dir_id = cur_dir_id - self.max_ep_store
            rm_dir_full_name = cur_dir_name + base_name + str(rm_dir_id)
            rmtree(rm_dir_full_name)

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
