"""Sinergym Loggers"""
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from sinergym.utils.constants import LOG_FORMAT


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


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that uses tqdm.write to avoid interfering with tqdm progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)  # Use tqdm.write to print the log message
            self.flush()
        except Exception:
            self.handleError(record)


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
        # consoleHandler = logging.StreamHandler(stream=sys.stdout)
        consoleHandler = TqdmLoggingHandler(stream=sys.stdout)
        consoleHandler.setFormatter(formatter)
        logger.addHandler(consoleHandler)
        logger.setLevel(level)
        logger.propagate = False
        return logger


class SimpleLogger():
    """Sinergym terminal logger for simulation executions.
    """

    def getLogger(
        self
    ):
        """Return Sinergym logger for the progress output in terminal.

            Args:
                name (str): logger name
                level (str): logger level
                formatter (Callable): logger formatter class

            Returns:
                logging.logger

            """
        logger = logging.getLogger('Printer')
        logger.setLevel(logging.INFO)
        # consoleHandler = logging.StreamHandler(stream=sys.stdout)
        simple_handler = TqdmLoggingHandler(stream=sys.stdout)
        simple_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(simple_handler)
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
        """Initialize LoggerStorage."""
        self.reset_data()

    def log_interaction(self,
                        obs: Union[List[float], np.ndarray],
                        action: Union[int, np.ndarray, List[float]],
                        reward: float,
                        info: Dict[str, Any],
                        terminated: bool,
                        truncated: bool,
                        custom_metrics: Optional[List[Any]] = None) -> None:
        """Log interaction data.

        Args:
            obs (Union[List[float], np.ndarray]): Observation data.
            action (Union[int, np.ndarray, List[float]]): Action data.
            reward (float): Reward data.
            info (Dict[str, Any]): Info data.
            terminated (bool): Termination flag.
            truncated (bool): Truncation flag.
            custom_metrics (List[Any]): Custom metric data. Default is None.
        """
        # Convert inputs to consistent formats
        obs = obs.tolist() if isinstance(obs, np.ndarray) else obs
        action = action.tolist() if isinstance(
            action, np.ndarray) else [action] if isinstance(
            action, (int, np.int64)) else action

        # Store data
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.infos.append(info)
        self.terminateds.append(terminated)
        self.truncateds.append(truncated)
        if custom_metrics:
            self.custom_metrics.append(custom_metrics)

        self.interactions += 1

    def log_norm_obs(self, norm_obs: Union[List[float], np.ndarray]) -> None:
        """Log normalized observation data.

        Args:
            norm_obs (Union[List[float], np.ndarray]): Normalized observation data.
        """
        self.normalized_observations.append(
            norm_obs.tolist() if isinstance(
                norm_obs, np.ndarray) else norm_obs)

    def log_obs(self, obs: Union[List[float], np.ndarray]) -> None:
        """Log observation data.

        Args:
            obs (Union[List[float], np.ndarray]): Observation data.
        """
        self.observations.append(
            obs.tolist() if isinstance(
                obs, np.ndarray) else obs)

    def log_info(self, info: Dict[str, Any]) -> None:
        """Log info data.

        Args:
            info (Dict[str, Any]): Info data.
        """
        self.infos.append(info)

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
        self.custom_metrics = []


try:
    import wandb
    from stable_baselines3.common.logger import KVWriter

    class WandBOutputFormat(KVWriter):  # pragma: no cover
        """
        Dumps key / value pairs onto WandB. This class is based on SB3 used in logger callback
        """

        def __init__(self):
            # Define X-Axis for SB3 metrics
            wandb.define_metric('time/*',
                                step_metric='time/total_timesteps')
            wandb.define_metric('train/*',
                                step_metric='time/total_timesteps')
            wandb.define_metric('rollout/*',
                                step_metric='time/total_timesteps')

        def write(
            self,
            key_values: Dict[str, Any],
            key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
            step: int = 0,
        ) -> None:

            # We store all metrics in a diuctionary to do a single log call
            metrics_to_log = {}

            for (key, value), (_, excluded) in zip(
                sorted(key_values.items()), sorted(key_excluded.items())
            ):

                if excluded is not None and "wandb" in excluded:
                    continue

                if isinstance(value, np.ScalarType):
                    if not isinstance(value, str):
                        # Store the metric
                        metrics_to_log[key] = value

            # Ensure 'time/total_timesteps' is included in the log
            # if 'time/total_timesteps' not in metrics_to_log:
            #     metrics_to_log['time/total_timesteps'] = step

            # Log all metrics
            wandb.log(metrics_to_log)
except ImportError:
    class WandBOutputFormat():  # pragma: no cover
        """WandBOutputFormat class for logging in WandB from SB3 logger.
        """

        def __init__(self):
            print(
                'WandB or SB3 is not installed. Please install it to use WandBOutputFormat.')
