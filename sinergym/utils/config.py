"""Class and utilities for set up extra configuration in experiments with Sinergym (extra params, weather_variability, building model modification and files management)"""
import os
import xml.etree.cElementTree as ElementTree
from copy import deepcopy
from shutil import rmtree
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas
from opyplus import Epm, Idd, WeatherData
from opyplus.epm.record import Record

from sinergym.utils.common import (get_delta_seconds, get_record_keys,
                                   prepare_batch_from_records, to_idf)
from sinergym.utils.constants import CWD, PKG_DATA_PATH, WEEKDAY_ENCODING, YEAR


class Config(object):
    """Config object to manage extra configuration in Sinergym experiments.

        :param _idf_path: IDF path origin for apply extra configuration.
        :param _weather_path: EPW path origin for apply weather to simulation.
        :param _ddy_path: DDY path origin for get DesignDays and weather Location
        :param experiment_path: Path for Sinergym experiment output
        :param episode_path: Path for Sinergym specific episode (before first simulator reset this param is None)
        :param max_ep_store: Number of episodes directories will be stored in experiment_path
        :param config: Dict config with extra configuration which is required to modify IDF model (may be None)
        :param _idd: IDD opyplus object to set up Epm
        :param building: opyplus Epm object with IDF model
        :param ddy_model: opyplus Epm object with DDY model
        :param weather_data: opyplus WeatherData object with EPW data
        :param action_definition: Dict with action definition to automatic building model preparation.
    """

    def __init__(
            self,
            idf_path: str,
            weather_path: str,
            variables: Dict[str, List[str]],
            env_name: str,
            max_ep_store: int,
            action_definition: Optional[Dict[str, Any]],
            extra_config: Dict[str, Any]):

        self._idf_path = idf_path
        self._weather_path = weather_path
        # RDD file name is deducible using idf name (only change .idf by .rdd)
        self._rdd_path = os.path.join(
            PKG_DATA_PATH,
            'variables',
            self._idf_path.split('/')[-1].split('.idf')[0] +
            '.rdd')
        # DDY path is deducible using weather_path (only change .epw by .ddy)
        self._ddy_path = self._weather_path.split('.epw')[0] + '.ddy'
        self.experiment_path = self.set_experiment_working_dir(env_name)
        self.episode_path = None
        self.max_ep_store = max_ep_store

        # Set config and action definition as config attribute
        self.config = extra_config
        self.action_definition = action_definition

        # Variables XML Tree (empty at the beginning)
        self.variables = variables
        self.variables_tree = ElementTree.Element('BCVTB-variables')

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

        # Extract idf zone names
        self.idf_zone_names = []
        for idf_zone in self.building.Zone:
            self.idf_zone_names.append(idf_zone.name.lower())
        # Extract rdd observation variables names
        data = pandas.read_csv(self._rdd_path, skiprows=1)
        self.rdd_variables_names = list(map(
            lambda name: name.split(' [')[0],
            data['Variable Name [Units]'].tolist()))

        # Extract schedulers available in building model
        self.schedulers = self.get_schedulers()

        # Check observation variables definition
        self._check_observation_variables()
        # Check config definition
        self._check_eplus_config()

    # ---------------------------------------------------------------------------- #
    #            IDF, variables and Building model adaptation                      #
    # ---------------------------------------------------------------------------- #

    def adapt_idf_to_epw(self,
                         summerday: str = 'Ann Clg .4% Condns DB=>MWB',
                         winterday: str = 'Ann Htg 99.6% Condns DB') -> None:
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

    def adapt_variables_to_cfg_and_idf(self) -> None:
        """This method adds to XML variable tree all observation and action variables information. In addition, it modifies IDF Output:Variable in order to adapt to new observation variables set.
        """
        # OBSERVATION VARIABLES
        output_variables = []
        self.variables_tree.append(ElementTree.Comment(
            'Observation variables: Received from EnergyPlus'))
        for obs_var in self.variables['observation']:
            # obs_var = "<variable_name>(<variable_zone>)"
            var_elements = obs_var.split('(')
            var_name = var_elements[0]
            var_zone = var_elements[1][:-1]

            # Add obs name and zone to XML variables tree
            new_xml_obs = ElementTree.SubElement(
                self.variables_tree, 'variable', source='EnergyPlus')
            ElementTree.SubElement(
                new_xml_obs,
                'EnergyPlus',
                name=var_zone,
                type=var_name)

            # Add IDF record Output:Variable
            if var_zone.lower() == 'Environment'.lower(
            ) or var_zone.lower() == 'Whole Building'.lower():
                var_zone = '*'
            output_variables.append(
                dict(
                    key_value=var_zone,
                    variable_name=var_name,
                    reporting_frequency='timestep'))

        # Delete default Output:Variables and added observation_variables
        # specified
        self.building.output_variable.delete()
        self.building.output_variable.batch_add(output_variables)

        # ACTION VARIABLES
        self.variables_tree.append(
            ElementTree.Comment('Action variables: Sent to EnergyPlus'))
        for act_var in self.variables['action']:

            new_xml_variable = ElementTree.SubElement(
                self.variables_tree, 'variable', source='Ptolemy')
            ElementTree.SubElement(
                new_xml_variable,
                'EnergyPlus',
                schedule=act_var)

    def set_external_interface(self) -> None:
        """Set an empty external interface with Ptolemy server if is not in the current building
        """

        # If no ExternalInterface object found
        if len(self.building.ExternalInterface) == 0:
            # Create PtolemyServer interface in building
            self.building.ExternalInterface.add(
                name_of_external_interface='PtolemyServer')

    def apply_extra_conf(self) -> None:
        """Set extra configuration in building model
        """
        if self.config is not None:

            # Timesteps processed in a simulation hour
            if self.config.get('timesteps_per_hour'):
                self.building.timestep[0].number_of_timesteps_per_hour = self.config['timesteps_per_hour']

            # Runperiod datetimes --> Tuple(start_day, start_month, start_year,
            # end_day, end_month, end_year)
            if self.config.get('runperiod'):
                runperiod = self.building.RunPeriod[0]
                runperiod.begin_day_of_month = int(self.config['runperiod'][0])
                runperiod.begin_month = int(self.config['runperiod'][1])
                runperiod.begin_year = int(self.config['runperiod'][2])
                runperiod.end_day_of_month = int(self.config['runperiod'][3])
                runperiod.end_month = int(self.config['runperiod'][4])
                runperiod.end_year = int(self.config['runperiod'][5])

    def adapt_idf_to_action_definition(self) -> None:
        """Interpret action definition and apply changes in IDF in order to control schedulers specified.
        """
        if self.action_definition is not None:
            # Iterate in schedulers to control in action definition
            for original_sch_name, new_sch in self.action_definition.items():
                # Search original scheduler information in building model
                original_sch = self.schedulers[original_sch_name.lower()]
                # Add external interface to IDF
                self.building.ExternalInterface_Schedule.add(
                    name=new_sch['name'],
                    schedule_type_limits_name=original_sch['Type'],
                    initial_value=new_sch['initial_value'])
                # Look for scheduler elements where appear and substitute for new
                # one name
                for key, element in original_sch.items():
                    if isinstance(element, dict):
                        table_name = element['object_type'].replace(
                            ':', '_').replace(' ', '_').lower()
                        record = self.building._tables[table_name].one(
                            lambda record: record.name == element['object_name'])
                        record[element['object_field_name']] = new_sch['name']

    def save_variables_cfg(self) -> str:
        """This method saves current XML variables tree model into a variables.cfg file.

        Raises:
            RuntimeError: If this method is used without an episode_path generated (see reset method in simulator), this exception is raised.

        Returns:
            str: Path to the new saved variables.cfg used by BCVTB for Energyplus communication.
        """
        if self.episode_path is not None:

            episode_cfg_path = self.episode_path + \
                '/variables.cfg'

            ElementTree.indent(self.variables_tree)

            with open(episode_cfg_path, "wb") as f:
                f.write(
                    '<?xml version="1.0" encoding="ISO-8859-1"?>\n<!DOCTYPE BCVTB-variables SYSTEM "variables.dtd">\n'.encode('utf8'))
                ElementTree.ElementTree(self.variables_tree).write(f, 'utf-8')

            return episode_cfg_path

        else:
            raise RuntimeError(
                '[Simulator Config] Episode path should be set before saving variables.cfg.')

    def save_building_model(self) -> str:
        """Take current building model and save as IDF in current env_working_dir episode folder.

        Returns:
            str: Path of IDF file stored (episode folder).
        """
        # If no path specified, then use idf_path to save it.
        if self.episode_path is not None:
            episode_idf_path = os.path.join(self.episode_path,
                                            os.path.basename(self._idf_path))
            # self.building.save(episode_idf_path)
            to_idf(building=self.building, file_path=episode_idf_path)
            return episode_idf_path
        else:
            raise RuntimeError(
                '[Simulator Config] Episode path should be set before saving building model.')

    def get_schedulers(self) -> Dict[str, Any]:
        """Extract all schedulers available in the building model to be controlled.

        Returns:
            Dict[str, Any]: Python Dictionary: For each scheduler found, it shows type value and where this scheduler is present (Object name, Object field and Object type).
        """
        result = {}
        schedule_tables = [
            self.building.schedule_compact,
            self.building.schedule_year]
        for schedule_table in schedule_tables:
            for schedule in schedule_table:
                object_index = 1
                result[schedule.name] = {}
                result[schedule.name]['Type'] = schedule.schedule_type_limits_name.name
                # Extract objects where scheduler is present as value field
                for table in self.building:
                    if table.get_name() != 'Schedule:Compact' and table.get_name() != 'Schedule:Year':
                        for record in table:
                            for i, value in enumerate(record):
                                if isinstance(value, Record):
                                    if (value._table._dev_descriptor.table_name == 'Schedule:Compact' or value._table._dev_descriptor.table_name ==
                                            'Schedule:Year') and value.name == schedule.name:
                                        result[schedule.name]['Object' + str(object_index)] = {'object_name': record.name,
                                                                                               'object_field_name': record._table._dev_descriptor._field_descriptors[i].ref,
                                                                                               'object_type': record._table._dev_descriptor.table_name}
                                        object_index += 1

        return result

    # ---------------------------------------------------------------------------- #
    #                        EPW and Weather Data management                       #
    # ---------------------------------------------------------------------------- #

    def apply_weather_variability(
            self,
            columns: List[str] = ['drybulb'],
            variation: Optional[Tuple[float, float, float]] = None) -> str:
        """Modify weather data using Ornstein-Uhlenbeck process.

        Args:
            columns (List[str], optional): List of columns to be affected. Defaults to ['drybulb'].
            variation (Optional[Tuple[float, float, float]], optional): Tuple with the sigma, mean and tau for OU process. Defaults to None.

        Returns:
            str: New EPW file path generated in simulator working path in that episode or current EPW path if variation is not defined.
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
            episode_weather_path = self.episode_path + '/' + filename
            weather_data_mod.to_epw(episode_weather_path)
            return episode_weather_path

    # ---------------------------------------------------------------------------- #
    #                        Model and Config Functionality                        #
    # ---------------------------------------------------------------------------- #

    def _get_eplus_run_info(
            self) -> Tuple[int, int, int, int, int, int, int, int]:
        """This method read the building model from config and finds the running start month, start day, start year, end month, end day, end year, start weekday and the number of steps in a hour simulation. If any value is Unknown, then value will be 0. If step per hour is < 1, then default value will be 4.

        Returns:
            Tuple[int, int, int, int, int, int, int, int]: A tuple with: the start month, start day, start year, end month, end day, end year, start weekday and number of steps in a hour simulation.
        """
        # Get runperiod object inner IDF
        runperiod = self.building.RunPeriod[0]

        # Extract information about runperiod
        start_month = int(
            0 if runperiod.begin_month is None else runperiod.begin_month)
        start_day = int(
            0 if runperiod.begin_day_of_month is None else runperiod.begin_day_of_month)
        start_year = int(
            YEAR if runperiod.begin_year is None else runperiod.begin_year)
        end_month = int(
            0 if runperiod.end_month is None else runperiod.end_month)
        end_day = int(
            0 if runperiod.end_day_of_month is None else runperiod.end_day_of_month)
        end_year = int(
            YEAR if runperiod.end_year is None else runperiod.end_year)
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

    def _get_one_epi_len(self) -> float:
        """Gets the length of one episode (an EnergyPlus process run to the end) depending on the config of simulation.

        Returns:
            float: The simulation time step in which the simulation ends.
        """
        # Get runperiod object inner IDF
        runperiod = self.building.RunPeriod[0]
        start_year = int(
            YEAR if runperiod.begin_year is None else runperiod.begin_year)
        start_month = int(
            0 if runperiod.begin_month is None else runperiod.begin_month)
        start_day = int(
            0 if runperiod.begin_day_of_month is None else runperiod.begin_day_of_month)
        end_year = int(
            YEAR if runperiod.end_year is None else runperiod.end_year)
        end_month = int(
            0 if runperiod.end_month is None else runperiod.end_month)
        end_day = int(
            0 if runperiod.end_day_of_month is None else runperiod.end_day_of_month)

        return get_delta_seconds(
            start_year,
            start_month,
            start_day,
            end_year,
            end_month,
            end_day)

    # ---------------------------------------------------------------------------- #
    #                  Working Folder for Simulation Management                    #
    # ---------------------------------------------------------------------------- #

    def set_experiment_working_dir(self, env_name: str) -> str:
        """Set experiment working dir path like config attribute for current simulation.

        Args:
            env_name (str): simulation env name to define a name in directory

        Returns:
            str: Experiment path for directory created.
        """
        # Generate experiment dir path
        experiment_path = self._get_working_folder(
            directory_path=CWD,
            base_name='-%s-res' %
            (env_name))
        # Create dir
        os.makedirs(experiment_path)
        # set path like config attribute
        self.experiment_path = experiment_path
        return experiment_path

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
            raise Exception
        else:
            episode_path = self._get_working_folder(
                directory_path=self.experiment_path,
                base_name='-sub_run')
            # Create directoy
            os.makedirs(episode_path)
            # set path like config attribute
            self.episode_path = episode_path

            # Remove redundant past working directories
            self._rm_past_history_dir(episode_path, '-sub_run')
            return episode_path

    def _get_working_folder(
            self,
            directory_path: str,
            base_name: str = '-run') -> str:
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

        working_dir = os.path.join(directory_path, 'Eplus-env')
        working_dir = working_dir + '%s%d' % (base_name, experiment_id)
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

    @ property
    def start_year(self) -> int:
        """Returns the EnergyPlus simulation year.

        Returns:
            int: Simulation year.
        """

        return YEAR

    # ---------------------------------------------------------------------------- #
    #                             Config class checker                             #
    # ---------------------------------------------------------------------------- #

    def _check_eplus_config(self) -> None:
        """Check Eplus Environment config definition is correct.
        """

        # EXTRA CONFIG
        if self.config is not None:
            for config_key in self.config.keys():
                # Check config parameters values
                # Timesteps
                if config_key == 'timesteps_per_hour':
                    assert self.config[config_key] > 0, 'Extra Config: timestep_per_hour must be a positive int value.'
                # Runperiod
                elif config_key == 'runperiod':
                    assert isinstance(
                        self.config[config_key], tuple) and len(
                        self.config[config_key]) == 6, 'Extra Config: Runperiod specified in extra configuration has an incorrect format (tuple with 6 elements).'
                else:
                    raise KeyError(
                        F'Extra Config: Key name specified in config called [{config_key}] has no support in Sinergym.')
        # ACTION DEFINITION
        if self.action_definition is not None:
            for original_sch_name, new_sch in self.action_definition.items():
                # Check action definition format
                assert isinstance(
                    original_sch_name, str), 'Action definition: Keys must be str.'
                assert isinstance(
                    new_sch, dict), 'Action definition: New scheduler definition must be a dict.'
                assert set(
                    new_sch.keys()) == set(
                    ['name', 'initial_value']), 'Action definition: keys in new scheduler definition must be name and initial_value.'
                assert isinstance(
                    new_sch['name'], str), 'Action definition: Name field in new scheduler must be a str element.'
                # Check action definition component is in schedulers available
                # in building model
                assert original_sch_name.lower() in self.schedulers.keys(
                ), 'Action definition: Object called {} is not an existing component in building model.'.format(original_sch_name)
                # Check new variable is present in action variables
                assert new_sch['name'] in self.variables['action'], 'Action definition: {} external variable should be in action variables.'.format(
                    new_sch['name'])

    def _check_observation_variables(self) -> None:
        """This method checks whether observation variables zones are available in building model definition
        """
        for obs_var in self.variables['observation']:
            obs_name = obs_var.split('(')[0]
            obs_zone = obs_var.split('(')[1][:-1]

            # Check observarion variables names
            assert obs_name in self.rdd_variables_names, 'Observation variables: Variable called {} in observation variables is not valid for IDF building model'.format(
                obs_name)
            # Check observation variables zones
            if obs_zone.lower() != 'Environment'.lower(
            ) and obs_zone.lower() != 'Whole Building'.lower():
                # zones names with people 1 or lights 1, etc. The second name
                # is ignored, only check that zone is a substr from obs zone
                assert any(list(map(lambda zone: zone.lower() in obs_zone.lower(), self.idf_zone_names))
                           ), 'Observation variables: Zone called {} in observation variables does not exist in IDF building model.'.format(obs_zone)

    # ---------------------------------------------------------------------------- #
    #                                    OTHERS                                    #
    # ---------------------------------------------------------------------------- #

    # def _set_thermostat_zone_control(self,
    #                                  zone_control: Record,
    #                                  controller_type: str,
    #                                  controller: Dict[str,
    #                                                   Any]) -> None:
    #     """Modify Thermostat:ZoneControl object from IDF model in order to set a new controller

    #     Args:
    #         zone_control (Record): Zone Control to be modified.
    #         controller_type (str): Controller type you want to change.
    #         controller (Dict[str, Any]): controller to set in Zone Control.
    #     """
    #     # We iterate all record fields searching
    #     # controller_type value
    #     for i in range(len(get_record_keys(zone_control))):
    #         if isinstance(zone_control[i], str):
    #             if zone_control[i].lower() == controller_type.lower():
    #                 # Then, the next field will be always the DualSetpoint
    #                 # thermostat name, so we change it
    #                 zone_control[i + 1] = controller['name']
    #                 # We do not need to search more fields in that record
    #                 # specifically, so break it
    #                 break
