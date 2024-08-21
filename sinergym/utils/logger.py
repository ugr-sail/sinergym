"""Sinergym Loggers"""
import logging
import sys
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


class LoggerStorage():
    """Logger storage for agent interaction with environment. Save all interactions in list or list of lists as attributes.

    Attributes:
        observations (List[List[float]]): List to store observations.
        normalized_observations (List[List[float]]): List to store normalized observations (if exists).
        actions (List[Union[int, np.ndarray, List[float]]]): List to store agent actions.
        rewards (List[float]): List to store rewards.
        infos (List[Dict[str, Any]]): List to store info data.
        terminateds (List[bool]): List to store terminated flags.
        truncateds (List[bool]): List to store truncated flags.
    """

    def __init__(self):
        """Logger constructor."""

        # Interaction data initialization
        self.interactions = 0
        self.observations = []
        self.normalized_observations = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.terminateds = []
        self.truncateds = []
        self.custom_metrics = []

    def log_interaction(self,
                        obs: List[float],
                        action: Union[int, np.ndarray, List[float]],
                        reward: float,
                        info: Dict[str, Any],
                        terminated: bool,
                        truncated: bool,
                        custom_metrics: List[Any] = None) -> None:
        """Log interaction data.

        Args:
            obs (List[float]): Observation data.
            action (Union[int, np.ndarray, List[float]]): Action data.
            reward (float): Reward data.
            info (Dict[str, Any]): Info data.
            terminated (bool): Termination flag.
            truncated (bool): Truncation flag.
            custom_metrics (List[Any]): Custom metric data. Default is None.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        if custom_metrics is not None and len(custom_metrics) > 0:
            self.custom_metrics.append(custom_metrics)
        self.interactions += 1

    def log_norm_obs(self, norm_obs: List[float]) -> None:
        """Log normalized observation data.

        Args:
            norm_obs (List[float]): Normalized observation data.
        """
        self.normalized_observations.append(norm_obs)

    def reset_data(self) -> None:
        """Reset logger interactions data"""
        self.interactions = 0
        self.observations = []
        self.normalized_observations = []
        self.actions = []
        self.rewards = []
        self.infos = []
        self.terminateds = []
        self.truncateds = []
        self.custom_data = []


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
