"""Sinergym Loggers"""

import csv
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pkg_resources

from sinergym.utils.constants import LOG_FORMAT

required = {'stable-baselines3', 'wandb'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if not missing:
    import wandb
    from stable_baselines3.common.logger import KVWriter


class CustomFormatter(logging.Formatter):
    """Custom logger format for terminal messages"""
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = LOG_FORMAT

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger():
    """Sinergym terminal logger for simulation executions.
    """

    def getLogger(
            self,
            name: str,
            level: str,
            formatter: Any = CustomFormatter()) -> logging.Logger:
        """Return Sinergym logger for the progress output in terminal.

        Args:
            name (str): logger name
            level (str): logger level
            formatter (Callable): logger formatter class

        Returns:
            logging.logger

        """
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger


class CSVLogger(object):
    """CSV Logger for agent interaction with environment.

        :param monitor_header: CSV header for sub_run_N/monitor.csv which record interaction step by step.
        :param progress_header: CSV header for res_N/progress.csv which record main data episode by episode.
        :param log_file: log_file path for monitor.csv, there will be one CSV per episode.
        :param log_progress_file: log_file path for progress.csv, there will be only one CSV per whole simulation.
        :param flag: This flag is used to activate (True) or deactivate (False) Logger in real time.
        :param steps_data, rewards, powers, etc: These arrays are used to record steps data to elaborate main data for progress.csv later.
        :param total_timesteps: Current episode timesteps executed.
        :param total_time_elapsed: Current episode time elapsed (simulation hours).
        :param comfort_violation_timesteps: Current episode timesteps whose comfort_penalty!=0.
        :param steps_data: It is a array of str's. Each element belong to a step data.

    """

    def __init__(
            self,
            monitor_header: str,
            progress_header: str,
            log_progress_file: str,
            log_file: Optional[str] = None,
            flag: bool = True):
        """CSVLogger constructor

        Args:
            monitor_header (str): CSV header for sub_run_N/monitor.csv which record interaction step by step.
            progress_header (str): CSV header for res_N/progress.csv which record main data episode by episode.
            log_progress_file (str): log_file path for progress.csv, there will be only one CSV per whole simulation.
            log_file (Optional[str], optional): log_file path for monitor.csv, there will be one CSV per episode. Defaults to None.
            flag (bool, optional): Activate (True) or deactivate (False) Logger in real time. Defaults to True.
        """

        self.monitor_header = monitor_header
        self.progress_header = progress_header + '\n'
        self.log_file = log_file
        self.log_progress_file = log_progress_file
        self.flag = flag

        # episode data
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.episode_data = {
            'rewards': [],
            'reward_energy_terms': [],
            'reward_comfort_terms': [],
            'abs_energy_penalties': [],
            'abs_comfort_penalties': [],
            'total_power_demands': [],
            'total_temperature_violations': [],
            'total_timesteps': 0,
            'total_time_elapsed': 0,
            'comfort_violation_timesteps': 0
        }

    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Dict[str, Any]) -> List:
        """Assemble the array data to log in the new row

        Args:
            obs (List[Any]): Observation from step.
            action (Union[int, np.ndarray, List[Any]]): Action done in step.
            terminated (bool): terminated flag in step.
            truncated (bool): truncated flag in step.
            info (Optional[Dict[str, Any]]): Extra info collected in step.

        Returns:
            List: Row content created in order to being logged.
        """
        return [
            info.get('timestep')] + list(obs) + list(action) + [
            info.get('time_elapsed(hours)'),
            info.get('reward'),
            info.get('energy_term'),
            info.get('comfort_term'),
            info.get('abs_energy_penalty'),
            info.get('abs_comfort_penalty'),
            info.get('total_power_demand'),
            info.get('total_temperature_violation'),
            terminated,
            truncated]

    def _store_step_information(
            self,
            info: Dict[str, Any]) -> None:
        """Store relevant data to episode summary in progress.csv.

        Args:
            info (Optional[Dict[str, Any]]): Extra info collected in step.
        """
        # In reset (timestep=1), some keys are not available in info
        if info['timestep'] > 1:

            self.episode_data['rewards'].append(info['reward'])
            self.episode_data['reward_energy_terms'].append(
                info['energy_term'])
            self.episode_data['reward_comfort_terms'].append(
                info['comfort_term'])
            self.episode_data['abs_energy_penalties'].append(
                info['abs_energy_penalty'])
            self.episode_data['abs_comfort_penalties'].append(
                info['abs_comfort_penalty'])
            self.episode_data['total_power_demands'].append(
                info['total_power_demand'])
            self.episode_data['total_temperature_violations'].append(
                info['total_temperature_violation'])
            if info['comfort_term'] < 0:
                self.episode_data['comfort_violation_timesteps'] += 1
            self.episode_data['total_time_elapsed'] = info['time_elapsed(hours)']
            self.episode_data['total_timesteps'] = info['timestep']

    def _reset_logger(self) -> None:
        """Reset relevant data to next episode summary in progress.csv.
        """
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.episode_data = {
            'rewards': [],
            'reward_energy_terms': [],
            'reward_comfort_terms': [],
            'abs_energy_penalties': [],
            'abs_comfort_penalties': [],
            'total_power_demands': [],
            'total_temperature_violations': [],
            'total_timesteps': 0,
            'total_time_elapsed': 0,
            'comfort_violation_timesteps': 0
        }

    def log_step(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Dict[str, Any]) -> None:
        """Log step information and store it in steps_data attribute.

        Args:
            obs (List[Any]): Observation from step.
            action (Union[int, np.ndarray, List[Any]]): Action done in step.
            terminated (bool): terminated flag in step.
            truncated (bool): truncated flag in step.
            info (Dict[str, Any]): Extra info collected in step.
        """
        if self.flag:
            self.steps_data.append(
                self._create_row_content(
                    obs, action, terminated, truncated, info))
            # Store step information for episode
            self._store_step_information(info)

    def log_step_normalize(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Dict[str, Any]) -> None:
        """Log step information and store it in steps_data attribute.

        Args:
            obs (List[Any]): Observation from step.
            action (Union[int, np.ndarray, List[Any]]): Action done in step.
            reward (Optional[float]): Reward returned in step.
            terminated (bool): terminated flag in step.
            truncated (bool): truncated flag in step.
            info (Optional[Dict[str, Any]]): Extra info collected in step.
        """
        if self.flag:
            self.steps_data_normalized.append(
                self._create_row_content(
                    obs, action, terminated, truncated, info))

    def log_episode(self, episode: int) -> None:
        """Log episode main information using steps_data param.

        Args:
            episode (int): Current simulation episode number.

        """
        if self.flag:

            # WRITE steps_info rows in monitor.csv
            with open(self.log_file, 'w', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerows(self.steps_data)

            # WRITE normalize steps_info rows in monitor_normalized.csv
            if len(self.steps_data_normalized) > 1:
                with open(self.log_file[:-4] + '_normalized.csv', 'w', newline='') as file_obj:
                    # Create a writer object from csv module
                    csv_writer = csv.writer(file_obj)
                    # Add contents of list as last row in the csv file
                    csv_writer.writerows(self.steps_data_normalized)

            # CREATE CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # CREATE progress.csv row to add with episode summary
            try:
                comfort_violation = (
                    self.episode_data['comfort_violation_timesteps'] /
                    self.episode_data['total_timesteps'] *
                    100)
            except ZeroDivisionError:
                comfort_violation = np.nan

            summary_row = [
                episode,
                np.sum(self.episode_data['rewards']),
                np.mean(self.episode_data['rewards']),
                np.sum(self.episode_data['reward_energy_terms']),
                np.mean(self.episode_data['reward_energy_terms']),
                np.sum(self.episode_data['reward_comfort_terms']),
                np.mean(self.episode_data['reward_comfort_terms']),
                np.sum(self.episode_data['abs_energy_penalties']),
                np.mean(self.episode_data['abs_energy_penalties']),
                np.sum(self.episode_data['abs_comfort_penalties']),
                np.mean(self.episode_data['abs_comfort_penalties']),
                np.sum(self.episode_data['total_power_demands']),
                np.mean(self.episode_data['total_power_demands']),
                np.sum(self.episode_data['total_temperature_violations']),
                np.mean(self.episode_data['total_temperature_violations']),
                comfort_violation,
                self.episode_data['total_timesteps'],
                self.episode_data['total_time_elapsed']]

            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(summary_row)

            # Reset episode information
            self._reset_logger()

    def set_log_file(self, new_log_file: str) -> None:
        """Change log_file path for monitor.csv when an episode ends.

        Args:
            new_log_file (str): New log path depending on simulation.

        """
        if self.flag:
            self.log_file = new_log_file
            if self.log_file:
                with open(self.log_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.monitor_header)

    def activate_flag(self) -> None:
        """Activate Sinergym CSV logger
        """
        self.flag = True

    def deactivate_flag(self) -> None:
        """Deactivate Sinergym CSV logger
        """
        self.flag = False


if not missing:
    class WandBOutputFormat(KVWriter):
        """
        Dumps key/value pairs onto WandB. This class is based on SB3 used in logger callback
        """

        def write(
            self,
            key_values: Dict[str, Any],
            key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
            step: int = 0,
        ) -> None:

            for (key, value), (_, excluded) in zip(
                sorted(key_values.items()), sorted(key_excluded.items())
            ):

                if excluded is not None and "wandb" in excluded:
                    continue

                if isinstance(value, np.ScalarType):
                    if not isinstance(value, str):
                        wandb.log({key: value}, step=step)
