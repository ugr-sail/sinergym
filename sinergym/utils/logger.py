"""Sinergym Loggers"""
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

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


class TerminalLogger():
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


class BaseLogger(ABC):
    """Abstract Logger for agent interaction with environment. Save all interactions and episode summary in Dataframes as attributes.

    Attributes:
        data (List[List[Any]]): List to store step data.
        data_normalized (List[List[Any]]): List to store normalized step data.
        summary_data (Dict[str,Any]): Dictionary to store episode summary data including rewards, penalties, power demands, and time information.
    """

    def __init__(self):
        """Logger constructor."""

        # Steps and episode data initialization
        self.data = []
        self.data_normalized = []
        self.summary_data = {
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

    def log_data(
        self,
        obs: List[Any],
        action: Union[int, np.ndarray, List[Any]],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """Log step information and store it in data attribute.

        Args:
            obs(List[Any]): Observation from step.
            action(Union[int, np.ndarray, List[Any]]): Action done in step.
            terminated(bool): terminated flag in step.
            truncated(bool): truncated flag in step.
            info(Dict[str, Any]): Extra info collected in step.
        """

        self.data.append(
            self._create_row_content(
                obs, action, terminated, truncated, info))
        # Store data information for summary
        self._store_information_summary(info)

    def log_normalized_data(
        self,
        obs: List[Any],
        action: Union[int, np.ndarray, List[Any]],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> None:
        """Log step information and store it in data attribute.

        Args:
            obs(List[Any]): Observation from step.
            action(Union[int, np.ndarray, List[Any]]): Action done in step.
            reward(Optional[float]): Reward returned in step.
            terminated(bool): terminated flag in step.
            truncated(bool): truncated flag in step.
            info(Optional[Dict[str, Any]]): Extra info collected in step.
        """

        self.data_normalized.append(
            self._create_row_content(
                obs, action, terminated, truncated, info))

    def return_episode_data(self, episode: int) -> None:
        """Return episode information and all data collected.

        Args:
            episode (int): Current simulation episode number.

        Returns:
            Tuple: Progress data (episode summary), monitor data(steps data), monitor data normalized.

        """
        progress_data = self._create_row_summary_content(episode)
        monitor_data = self.data
        monitor_data_normalized = self.data_normalized

        return progress_data, monitor_data, monitor_data_normalized

    def reset_logger(self) -> None:
        """Reset relevant data to next episode summary.
        """
        self.data = []
        self.data_normalized = []
        for key, value in self.summary_data.items():
            if isinstance(value, list):
                self.summary_data[key] = []
            else:
                self.summary_data[key] = 0

    @abstractmethod
    def _create_row_content(
        self,
        obs: List[Any],
        action: Union[int, np.ndarray, List[Any]],
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any]
    ) -> List:
        """Assemble the array data to log in the new row

        Args:
            obs(List[Any]): Observation from step.
            action(Union[int, np.ndarray, List[Any]]): Action done in step.
            terminated(bool): terminated flag in step.
            truncated(bool): truncated flag in step.
            info(Optional[Dict[str, Any]]): Extra info collected in step.

        Returns:
            List: Row content created in order to being logged.
        """
        pass

    @abstractmethod
    def _store_information_summary(
        self,
        info: Dict[str, Any]
    ) -> None:
        """Store relevant data to episode summary.

        Args:
            info(Optional[Dict[str, Any]]): Extra info collected in step.
        """
        pass

    @abstractmethod
    def _create_row_summary_content(self, episode: int) -> List:
        """Create the row content for the episode summary.

        Args:
            episode (int): Current simulation episode number.

        Returns:
            List: Row content created in order to being logged.
        """
        pass


class Logger(BaseLogger):
    """Logger for agent interaction with environment. Save all interactions and episode summary in Dataframes as attributes.
    """

    def __init__(
            self):
        """Logger constructor."""

        super().__init__()
        # Note: Update self.summary_data with episode data if it is different

    def _create_row_content(
            self,
            obs: List[Any],
            action: Union[int, np.ndarray, List[Any]],
            terminated: bool,
            truncated: bool,
            info: Dict[str, Any]) -> List:
        """Assemble the array data to log in the new row

        Args:
            obs(List[Any]): Observation from step.
            action(Union[int, np.ndarray, List[Any]]): Action done in step.
            terminated(bool): terminated flag in step.
            truncated(bool): truncated flag in step.
            info(Optional[Dict[str, Any]]): Extra info collected in step.

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

    def _store_information_summary(
            self,
            info: Dict[str, Any]) -> None:
        """Store relevant data to episode summary.

        Args:
            info(Optional[Dict[str, Any]]): Extra info collected in step.
        """
        # In reset (timestep=1), some keys are not available in info
        if info['timestep'] > 1:

            self.summary_data['rewards'].append(info['reward'])
            self.summary_data['reward_energy_terms'].append(
                info['energy_term'])
            self.summary_data['reward_comfort_terms'].append(
                info['comfort_term'])
            self.summary_data['abs_energy_penalties'].append(
                info['abs_energy_penalty'])
            self.summary_data['abs_comfort_penalties'].append(
                info['abs_comfort_penalty'])
            self.summary_data['total_power_demands'].append(
                info['total_power_demand'])
            self.summary_data['total_temperature_violations'].append(
                info['total_temperature_violation'])
            if info['comfort_term'] < 0:
                self.summary_data['comfort_violation_timesteps'] += 1
            self.summary_data['total_time_elapsed'] = info['time_elapsed(hours)']
            self.summary_data['total_timesteps'] = info['timestep']

    def _create_row_summary_content(self, episode: int) -> List:
        """Create the row content for the episode summary.

        Args:
            episode (int): Current simulation episode number.

        Returns:
            List: Row content created in order to being logged.
        """
        try:
            comfort_violation = (
                self.summary_data['comfort_violation_timesteps'] /
                self.summary_data['total_timesteps'] *
                100)
        except ZeroDivisionError:
            comfort_violation = np.nan

        return [episode,
                np.sum(self.summary_data['rewards']),
                np.mean(self.summary_data['rewards']),
                np.std(self.summary_data['rewards']),
                np.sum(self.summary_data['reward_energy_terms']),
                np.mean(self.summary_data['reward_energy_terms']),
                np.sum(self.summary_data['reward_comfort_terms']),
                np.mean(self.summary_data['reward_comfort_terms']),
                np.sum(self.summary_data['abs_energy_penalties']),
                np.mean(self.summary_data['abs_energy_penalties']),
                np.sum(self.summary_data['abs_comfort_penalties']),
                np.mean(self.summary_data['abs_comfort_penalties']),
                np.sum(self.summary_data['total_power_demands']),
                np.mean(self.summary_data['total_power_demands']),
                np.sum(self.summary_data['total_temperature_violations']),
                np.mean(self.summary_data['total_temperature_violations']),
                comfort_violation,
                self.summary_data['total_timesteps'],
                self.summary_data['total_time_elapsed']]


if not missing:
    class WandBOutputFormat(KVWriter):
        """
        Dumps key / value pairs onto WandB. This class is based on SB3 used in logger callback
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
