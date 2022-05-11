"""Sinergym Loggers"""

import csv
import logging
import os
from typing import Any, List, Optional, Union

import numpy as np


class Logger():
    """Sinergym terminal logger for simulation executions.
    """

    def getLogger(
            self,
            name: str,
            level: str,
            formatter: str) -> logging.Logger:
        """Return Sinergym logger for the progress output in terminal.

        Args:
            name (str): logger name
            level (str): logger level
            formatter (str): logger formatter

        Returns:
            logging.logger

        """
        logger = logging.getLogger(name)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logging.Formatter(formatter))
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
        :param total_time_elapsed: Current episode time elapsed (simulation seconds).
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

        self.monitor_header = monitor_header
        self.progress_header = progress_header + '\n'
        self.log_file = log_file
        self.log_progress_file = log_progress_file
        self.flag = flag

        # episode data
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self.comfort_penalties = []
        self.power_penalties = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def log_step(
            self,
            timestep: int,
            observation: List[Any],
            action: Union[List[Union[int, float]], List[None]],
            simulation_time: float,
            reward: Optional[float],
            total_power_no_units: Optional[float],
            comfort_penalty: Optional[float],
            power: Optional[float],
            done: bool) -> None:
        """Log step information and store it in steps_data attribute.

        Args:
            timestep (int): Current episode timestep in simulation.
            observation (list): Values that belong to current observation.
            action (list): Values that belong to current action.
            simulation_time (float): Total time elapsed in current episode (seconds).
            reward (float): Current reward achieved.
            total_power_no_units (float): Power consumption penalty depending on reward function.
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            power (float): Power consumption in current step (W).
            done (bool): It specifies if this step terminates episode or not.

        """
        if self.flag:
            row_contents = [timestep] + list(observation) + \
                list(action) + [simulation_time, reward,
                                total_power_no_units, comfort_penalty, done]
            self.steps_data.append(row_contents)

            # Store step information for episode
            self._store_step_information(
                reward,
                power,
                comfort_penalty,
                total_power_no_units,
                timestep,
                simulation_time)
        else:
            pass

    def log_step_normalize(
            self,
            timestep: int,
            observation: List[Any],
            action: Union[List[Union[int, float]], List[None]],
            simulation_time: float,
            reward: Optional[float],
            total_power_no_units: Optional[float],
            comfort_penalty: Optional[float],
            done: bool) -> None:
        """Log step information and store it in steps_data_normalized attribute.

        Args:
            timestep (int): Current episode timestep in simulation.
            observation (List[Any]): Values that belong to current observation.
            action (List[Union[int, float]]): Values that belong to current action.
            simulation_time (float): Total time elapsed in current episode (seconds).
            reward (float): Current reward achieved.
            total_power_no_units (float): Power consumption penalty depending on reward function.
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            done (bool): It specifies if this step terminates episode or not.
        """
        if self.flag:
            row_contents = [timestep] + list(observation) + \
                list(action) + [simulation_time, reward,
                                total_power_no_units, comfort_penalty, done]
            self.steps_data_normalized.append(row_contents)
        else:
            pass

    def log_episode(self, episode: int) -> None:
        """Log episode main information using steps_data param.

        Args:
            episode (int): Current simulation episode number.

        """
        if self.flag:
            # statistics metrics for whole episode
            ep_mean_reward = np.mean(self.rewards)
            ep_cumulative_reward = np.sum(self.rewards)
            ep_cumulative_power = np.sum(self.powers)
            ep_mean_power = np.mean(self.powers)
            ep_cumulative_comfort_penalty = np.sum(self.comfort_penalties)
            ep_mean_comfort_penalty = np.mean(self.comfort_penalties)
            ep_cumulative_power_penalty = np.sum(self.power_penalties)
            ep_mean_power_penalty = np.mean(self.power_penalties)
            try:
                comfort_violation = (
                    self.comfort_violation_timesteps /
                    self.total_timesteps *
                    100)
            except ZeroDivisionError:
                comfort_violation = np.nan

            # write steps_info in monitor.csv
            with open(self.log_file, 'w', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerows(self.steps_data)

            # Write normalize steps_info in monitor_normalized.csv
            if len(self.steps_data_normalized) > 1:
                with open(self.log_file[:-4] + '_normalized.csv', 'w', newline='') as file_obj:
                    # Create a writer object from csv module
                    csv_writer = csv.writer(file_obj)
                    # Add contents of list as last row in the csv file
                    csv_writer.writerows(self.steps_data_normalized)

            # Create CSV file with header if it's required for progress.csv
            if not os.path.isfile(self.log_progress_file):
                with open(self.log_progress_file, 'a', newline='\n') as file_obj:
                    file_obj.write(self.progress_header)

            # building episode row
            row_contents = [
                episode,
                ep_cumulative_reward,
                ep_mean_reward,
                ep_cumulative_power,
                ep_mean_power,
                ep_cumulative_comfort_penalty,
                ep_mean_comfort_penalty,
                ep_cumulative_power_penalty,
                ep_mean_power_penalty,
                comfort_violation,
                self.total_timesteps,
                self.total_time_elapsed]
            with open(self.log_progress_file, 'a+', newline='') as file_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(file_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(row_contents)

            # Reset episode information
            self._reset_logger()
        else:
            pass

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
        else:
            pass

    def _store_step_information(
            self,
            reward: float,
            power: float,
            comfort_penalty: float,
            power_penalty: float,
            timestep: int,
            simulation_time: float) -> None:
        """Store relevant data to episode summary in progress.csv.

        Args:
            reward (float): Current reward achieved.
            power (float): Power consumption in current step (W).
            comfort_penalty (float): Temperature comfort penalty depending on reward function.
            power_penalty (float): Power consumption penalty depending on reward function.
            timestep (int): Current episode timestep in simulation.
            simulation_time (float): Total time elapsed in current episode (seconds).

        """
        if reward is not None:
            self.rewards.append(reward)
        if power is not None:
            self.powers.append(power)
        if comfort_penalty is not None:
            self.comfort_penalties.append(comfort_penalty)
        if power_penalty is not None:
            self.power_penalties.append(power_penalty)
        if comfort_penalty != 0:
            self.comfort_violation_timesteps += 1
        self.total_timesteps = timestep
        self.total_time_elapsed = simulation_time

    def _reset_logger(self) -> None:
        """Reset relevant data to next episode summary in progress.csv.
        """
        self.steps_data = [self.monitor_header.split(',')]
        self.steps_data_normalized = [self.monitor_header.split(',')]
        self.rewards = []
        self.powers = []
        self. comfort_penalties = []
        self.power_penalties = []
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.comfort_violation_timesteps = 0

    def activate_flag(self) -> None:
        """Activate Sinergym CSV logger
        """
        self.flag = True

    def deactivate_flag(self) -> None:
        """Deactivate Sinergym CSV logger
        """
        self.flag = False
