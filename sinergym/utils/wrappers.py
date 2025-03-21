"""Implementation of custom Gym environments."""

import csv
import os
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from epw.weather import Weather
from gymnasium import Env
from gymnasium.wrappers.utils import RunningMeanStd

from sinergym.utils.common import is_wrapped, ornstein_uhlenbeck_process
from sinergym.utils.constants import LOG_WRAPPERS_LEVEL, YEAR
from sinergym.utils.logger import LoggerStorage, TerminalLogger
from sinergym.utils.rewards import EnergyCostLinearReward

# ---------------------------------------------------------------------------- #
#                             Observation wrappers                             #
# ---------------------------------------------------------------------------- #


class DatetimeWrapper(gym.ObservationWrapper):
    """Wrapper to transform datetime variables into a more useful representation:
       - 'day_of_month' is replaced with 'is_weekend' (1 if weekend, 0 otherwise).
       - 'hour' is replaced with its sine and cosine encoding.
       - 'month' is replaced with its sine and cosine encoding.
       The observation space is updated automatically.
    """

    logger = TerminalLogger().getLogger(name='WRAPPER DatetimeWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env):
        super().__init__(env)

        # Obtain observation variables from environment
        obs_vars = self.get_wrapper_attr('observation_variables')

        # Check datetime variables are defined in environment
        required_vars = {'month', 'day_of_month', 'hour'}
        if not required_vars.issubset(obs_vars):
            self.logger.error(
                "month, day_of_month, and hour must be defined in the environment's observation space.")
            raise ValueError

        # Update observation space
        obs_space = self.get_wrapper_attr('observation_space')
        self.observation_space = gym.spaces.Box(
            low=obs_space.low[0],
            high=obs_space.high[0],
            # +2 porque agregamos senos y cosenos
            shape=(obs_space.shape[0] + 2,),
            dtype=obs_space.dtype
        )

        # Update observation variables with new datetime variables
        new_obs_vars = deepcopy(obs_vars)
        new_obs_vars[new_obs_vars.index('day_of_month')] = 'is_weekend'
        hour_idx = new_obs_vars.index('hour')
        new_obs_vars[hour_idx] = 'hour_cos'
        new_obs_vars.insert(hour_idx + 1, 'hour_sin')
        month_idx = new_obs_vars.index('month')
        new_obs_vars[month_idx] = 'month_cos'
        new_obs_vars.insert(month_idx + 1, 'month_sin')

        self.observation_variables = new_obs_vars
        self.logger.info('Wrapper initialized.')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Transforms the observation to replace time variables with encoded representations.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Transformed observation.
        """
        obs_dict = dict(
            zip(self.env.get_wrapper_attr('observation_variables'), obs))

        # Obtain year if present
        year = obs_dict.get('year', YEAR)

        # Create datetime object
        dt = datetime(
            year, int(
                obs_dict['month']), int(
                obs_dict['day_of_month']), int(
                obs_dict['hour']))

        # Build new observation of transformed datetime variables
        new_obs = {key: obs_dict[key] if key in obs_dict else None for key in self.get_wrapper_attr(
            'observation_variables')}
        new_obs.update({
            'is_weekend': 1.0 if dt.weekday() >= 5 else 0.0,
            'hour_cos': np.cos(2 * np.pi * obs_dict['hour'] / 24),
            'hour_sin': np.sin(2 * np.pi * obs_dict['hour'] / 24),
            'month_cos': np.cos(2 * np.pi * (obs_dict['month'] - 1) / 12),
            'month_sin': np.sin(2 * np.pi * (obs_dict['month'] - 1) / 12),
        })

        return np.fromiter(new_obs.values(), dtype=np.float32)

# ---------------------------------------------------------------------------- #


class PreviousObservationWrapper(gym.ObservationWrapper):
    """Wrapper to add observation values from previous timestep to
    current environment observation"""

    logger = TerminalLogger().getLogger(
        name='WRAPPER PreviousObservationWrapper',
        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 previous_variables: List[str]):
        super().__init__(env)

        # Obtain observation variables from environment and check if previous
        # variables are defined
        obs_vars = self.get_wrapper_attr('observation_variables')
        if not set(previous_variables).issubset(obs_vars):
            missing_vars = set(previous_variables) - set(obs_vars)
            self.logger.error(f'Missing observation variables: {missing_vars}')
            raise ValueError

        # Update observation variables
        self.previous_variables = previous_variables
        self.observation_variables = obs_vars + \
            [var + '_previous' for var in previous_variables]

        # Update observation space
        obs_space = self.get_wrapper_attr('observation_space')
        self.observation_space = gym.spaces.Box(
            low=obs_space.low[0],
            high=obs_space.high[0],
            shape=(obs_space.shape[0] + len(previous_variables),),
            dtype=obs_space.dtype
        )

        # Initialize previous observation with zeros
        self.previous_observation = np.zeros(
            len(previous_variables), dtype=np.float32)

        self.logger.info('Wrapper initialized.')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add previous observation to the current one

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: observation with
        """
        # Concatenate current obs with previous observation variables
        new_obs = np.concatenate((obs, self.previous_observation))

        # Update previous observation to current observation
        obs_vars = self.get_wrapper_attr('observation_variables')
        self.previous_observation = np.array(
            [obs[obs_vars.index(var)] for var in self.previous_variables], dtype=np.float32)

        return new_obs

# ---------------------------------------------------------------------------- #


class MultiObsWrapper(gym.Wrapper):

    logger = TerminalLogger().getLogger(name='WRAPPER MultiObsWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(
            self,
            env: Env,
            n: int = 5,
            flatten: bool = True) -> None:
        """Stack of observations.

        Args:
            env (Env): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super().__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = deque([], maxlen=n)
        shape = self.get_wrapper_attr('observation_space').shape
        new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=new_shape,
            dtype=self.env.get_wrapper_attr('observation_space').dtype)

        self.logger.info('Wrapper initialized.')

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            np.ndarray: Stacked previous observations.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.n):
            self.history.append(obs)
        return self._get_obs(), info

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (np.ndarray): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated episode and dict with extra information.
        """

        observation, reward, terminated, truncated, info = self.env.step(
            action)
        self.history.append(observation)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get observation history.

        Returns:
            np.array: Array of previous observations.
        """
        if self.get_wrapper_attr('ind_flat'):
            return np.array(self.history, dtype=np.float32).reshape(-1,)
        else:
            return np.array(self.history, dtype=np.float32)

# ---------------------------------------------------------------------------- #


class NormalizeObservation(gym.Wrapper):

    logger = TerminalLogger().getLogger(name='WRAPPER NormalizeObservation',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 automatic_update: bool = True,
                 epsilon: float = 1e-8,
                 mean: Union[list, np.float64, str] = None,
                 var: Union[list, np.float64, str] = None):
        """Initializes the NormalizationWrapper. Mean and var values can be None and being updated during interaction with environment.

        Args:
            env (Env): The environment to apply the wrapper.
            automatic_update (bool, optional): Whether or not to update the mean and variance values automatically. Defaults to True.
            epsilon (float, optional): A stability parameter used when scaling the observations. Defaults to 1e-8.
            mean (list, np.float64, str, optional): The mean value used for normalization. It can be a mean.txt path too. Defaults to None.
            var (list, np.float64, str, optional): The variance value used for normalization. It can be a var.txt path too. Defaults to None.
        """
        super().__init__(env)

        # Attributes
        self.automatic_update = automatic_update
        self.epsilon = epsilon
        self.num_envs = 1
        self.is_vector_env = False
        self.unwrapped_observation = None

        # Initialize normalization calibration
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

        # Set mean and variance
        self.obs_rms.mean = self._process_metric(
            mean, 'mean') or self.obs_rms.mean
        self.obs_rms.var = self._process_metric(var, 'var') or self.obs_rms.var

        self.logger.info('Wrapper initialized.')

    def step(self, action: np.ndarray) -> Tuple[
            np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps through the environment and normalizes the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save original obs in class attribute
        self.unwrapped_observation = deepcopy(obs)

        # Normalize observation and return
        return self.normalize(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""

        episode = self.get_wrapper_attr('episode')

        # Update normalization calibration if it is required
        if episode > 0:
            self._save_normalization_calibration()

        obs, info = self.env.reset(**kwargs)
        self.unwrapped_observation = deepcopy(obs)

        return self.normalize(obs), info

    def close(self):
        """save normalization calibration and close the environment."""
        # Update normalization calibration if it is required
        if self.get_wrapper_attr('episode') > 0:
            self._save_normalization_calibration()
        self.env.close()

# ----------------------- Wrapper extra functionality ----------------------- #
    def _process_metric(self, metric, metric_name) -> Union[np.ndarray, None]:
        """Validates, loads, and converts mean/variance metrics."""
        if metric is None:
            return None

        if isinstance(metric, str):  # If it's a file path
            if os.path.exists(metric):
                return np.loadtxt(metric, dtype=np.float64)
            self.logger.error(f"{metric_name}.txt file not found: {metric}")
            raise FileNotFoundError

        # Convert list to np.ndarray if needed
        metric = np.asarray(metric, dtype=np.float64)

        if metric.shape[0] != self.observation_space.shape[0]:
            self.logger.error(
                f"{metric_name} shape mismatch: expected {
                    self.observation_space.shape[0]}, got {
                    metric.shape[0]}")
            raise ValueError

        return metric

    def _save_normalization_calibration(self):
        """Saves the normalization calibration data in the output folder as txt files.
        """
        episode_path = self.get_wrapper_attr('episode_path')
        workspace_path = self.get_wrapper_attr('workspace_path')

        np.savetxt(os.path.join(episode_path, 'mean.txt'), self.mean)
        np.savetxt(os.path.join(episode_path, 'var.txt'), self.var)
        np.savetxt(os.path.join(workspace_path, 'mean.txt'), self.mean)
        np.savetxt(os.path.join(workspace_path, 'var.txt'), self.var)

        self.logger.info('Normalization calibration saved.')

    def deactivate_update(self):
        """
        Deactivates the automatic update of the normalization wrapper.
        After calling this method, the normalization wrapper will not update its calibration automatically.
        """
        self.automatic_update = False

    def activate_update(self):
        """
        Activates the automatic update of the normalization wrapper.
        After calling this method, the normalization wrapper will update its calibration automatically.
        """
        self.automatic_update = True

    @property
    def mean(self) -> np.float64:
        """Returns the mean value of the observations."""
        return self.obs_rms.mean

    @property
    def var(self) -> np.float64:
        """Returns the variance value of the observations."""
        return self.obs_rms.var

    def set_mean(self, mean: Union[List[float], np.ndarray, str]):
        """Sets the mean value of the observations."""
        mean = self._process_metric(mean, 'mean')
        self.obs_rms.mean = deepcopy(mean)

    def set_var(self, var: Union[List[float], np.ndarray, str]):
        """Sets the variance value of the observations."""
        var = self._process_metric(var, 'var')
        self.obs_rms.var = deepcopy(var)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalizes the observation using the running mean and variance of the observations.
        If automatic_update is enabled, the running mean and variance will be updated too."""
        if self.automatic_update:
            self.obs_rms.update(obs)

        std = np.sqrt(self.obs_rms.var + self.epsilon)
        return (obs - self.obs_rms.mean) / std


class WeatherForecastingWrapper(gym.Wrapper):

    logger = TerminalLogger().getLogger(
        name='WRAPPER WeatherForecastingWrapper',
        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 n: int = 5,
                 delta: int = 1,
                 columns: List[str] = ['Dry Bulb Temperature',
                                       'Relative Humidity',
                                       'Wind Direction',
                                       'Wind Speed',
                                       'Direct Normal Radiation',
                                       'Diffuse Horizontal Radiation'],
                 forecast_variability: Optional[Dict[str,
                                                     Tuple[float,
                                                           float,
                                                           float]]] = None):
        """Adds weather forecast information to the current observation.

        Args:
            env (Env): Original Gym environment.
            n (int, optional): Number of observations to be added. Default to 5.
            delta (int, optional): Time interval between observations. Defaults to 1.
            columns (List[str], optional): List of the names of the meteorological variables that will make up the weather forecast observation.
            forecast_variability (Dict[str, Tuple[float, float, float]], optional): Dictionary with the variation for each column in the weather data. Defaults to None.
            The key is the column name and the value is a tuple with the sigma, mean and tau for OU process. If not provided, it assumes no variability.
        Raises:
            ValueError: If any key in `forecast_variability` is not present in the `columns` list.
        """
        if forecast_variability is not None:
            for variable in forecast_variability.keys():
                if variable not in columns:
                    raise ValueError(
                        f"The variable '{variable}' in forecast_variability is not in columns.")

        super().__init__(env)
        self.n = n
        self.delta = delta
        self.columns = columns
        self.forecast_variability = forecast_variability
        new_observation_variables = []
        for i in range(1, n + 1):
            for column in columns:
                new_observation_variables.append(
                    'forecast_' + str(i) + '_' + column)
        self.observation_variables = self.env.get_wrapper_attr(
            'observation_variables') + new_observation_variables
        new_shape = (self.get_wrapper_attr(
            'observation_space').shape[0] + (len(columns) * n),)
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=new_shape,
            dtype=self.env.get_wrapper_attr('observation_space').dtype)
        self.forecast_data = None
        self.logger.info('Wrapper initialized.')

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Tuple with next observation, and dict with information about the environment.
        """
        self.set_forecast_data()

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.observation(obs, info)

        return obs, info

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (np.ndarray): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated
            episode and dict with Information about the environment.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs, info)

        return obs, reward, terminated, truncated, info

    def set_forecast_data(self):
        """Set the weather data used to build de state observation. If forecast_variability is not None,
           it applies Ornstein-Uhlenbeck process to the data.
        """
        data = Weather()
        data.read(self.get_wrapper_attr('weather_path'))
        self.forecast_data = data.dataframe.loc[:, [
            'Month', 'Day', 'Hour'] + self.columns]

        if self.forecast_variability is not None:
            self.forecast_data = ornstein_uhlenbeck_process(
                data=self.forecast_data, variability_config=self.forecast_variability)

    def observation(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Build the state observation by adding weather forecast information.

        Args:
            obs (np.ndarray): Original observation.
            info (Dict[str, Any]): Information about the environment.
        Returns:
            np.ndarray: Transformed observation.
        """
        # Search for the index corresponding to the time of the current
        # observation.
        filter = (
            (self.forecast_data['Month'].to_numpy() == info['month']) &
            (self.forecast_data['Day'].to_numpy() == info['day']) &
            (self.forecast_data['Hour'].to_numpy() == info['hour'] + 1)
        )
        i = np.where(filter)[0][0]

        # Create a list of indexes corresponding to the weather forecasts to be
        # added
        indexes = np.arange(
            i + self.delta,
            i + self.delta * self.n + 1,
            self.delta)
        indexes = indexes[indexes < len(self.forecast_data)]

        # Exceptional case 1: no weather forecast remains. In this case we fill in by repeating
        # the information from the weather forecast observation of current time
        # until the required size is reached.
        if len(indexes) == 0:
            indexes = [i]

        # Obtain weather forecast observations
        selected_rows = self.forecast_data.iloc[indexes,
                                                :][self.columns].values

        # Exceptional case 2: If there are not enough weather forecasts, repeat the last weather forecast observation
        # until the required size is reached.
        if len(selected_rows) < self.n:
            needed_rows = self.n - len(selected_rows)
            # Ensure adecuate shape
            last_row = selected_rows[-1:]
            selected_rows = np.vstack(
                [selected_rows, np.repeat(last_row, needed_rows, axis=0)])

        # Flatten the selected rows
        obs = np.concatenate((obs, selected_rows.ravel()))

        return obs


class EnergyCostWrapper(gym.Wrapper):
    logger = TerminalLogger().getLogger(name='WRAPPER EnergyCostWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 energy_cost_data_path: str,
                 reward_kwargs: Optional[Dict[str,
                                              Any]] = {'temperature_variables': ['air_temperature'],
                                                       'energy_variables': ['HVAC_electricity_demand_rate'],
                                                       'energy_cost_variables': ['energy_cost'],
                                                       'range_comfort_winter': [20.0,
                                                                                23.5],
                                                       'range_comfort_summer': [23.0,
                                                                                26.0],
                                                       'temperature_weight': 0.4,
                                                       'energy_weight': 0.4,
                                                       'lambda_energy': 1e-4,
                                                       'lambda_temperature': 1.0,
                                                       'lambda_energy_cost': 1.0},
                 energy_cost_variability: Optional[Tuple[float,
                                                         float,
                                                         float]] = None):
        """
        Adds energy cost information to the current observation.

        Args:
            env (Env): Original Gym environment.
            energy_cost_data_path (str): Pathfile from which the energy cost data is obtained.
            energy_cost_variability (Tuple[float,float,float], optional): variation for energy cost data for OU process (sigma, mu and tau).
            reward_kwargs (Dict[str, Any], optional): Parameters for customizing the reward function.

        """
        allowed_keys = {
            'temperature_variables',
            'energy_variables',
            'energy_cost_variables',
            'range_comfort_winter',
            'range_comfort_summer',
            'temperature_weight',
            'energy_weight',
            'lambda_energy',
            'lambda_temperature',
            'lambda_energy_cost'
        }

        if reward_kwargs:
            for key in reward_kwargs.keys():
                if key not in allowed_keys:
                    raise ValueError(
                        f"The key '{key}' in reward_kwargs is not recognized.")

        super().__init__(env)
        self.energy_cost_variability = {
            'value': energy_cost_variability} if energy_cost_variability is not None else None
        self.energy_cost_data_path = energy_cost_data_path
        self.observation_variables = self.env.get_wrapper_attr(
            'observation_variables') + ['energy_cost']
        new_shape = self.env.get_wrapper_attr('observation_space').shape[0] + 1
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=(
                new_shape,
            ),
            dtype=self.env.get_wrapper_attr('observation_space').dtype)
        self.energy_cost_data = None
        self.reward_fn = EnergyCostLinearReward(**reward_kwargs)
        self.logger.info('Wrapper initialized.')

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Resets the environment.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Tuple with next observation, and dict with information about the environment.
        """
        self.set_energy_cost_data()

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.observation(obs, info)

        return obs, info

    def step(self, action: np.ndarray

             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Tuple with next observation, reward, bool for terminated
            episode and dict with Information about the environment.
        """

        obs, _, terminated, truncated, info = self.env.step(action)
        new_obs = self.observation(obs, info)

        obs_dict = dict(zip(self.get_wrapper_attr('observation_variables'), np.concatenate(
            (new_obs[:len(self.get_wrapper_attr('observation_variables')) - 1], [new_obs[-1]]))))

        # Recalculation of reward with new info
        new_reward, new_terms = self.reward_fn(obs_dict)
        info = {
            key: info[key] for key in list(
                info.keys())[
                :list(
                    info.keys()).index('reward') +
                1]}

        info.update({'reward': new_reward})
        info.update(new_terms)

        return new_obs, new_reward, terminated, truncated, info

    def set_energy_cost_data(self):
        """Sets the cost of energy data used to construct the state observation.
        """

        df = pd.read_csv(self.energy_cost_data_path, sep=';')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['datetime'] += pd.DateOffset(hours=1)

        df['Month'] = df['datetime'].dt.month
        df['Day'] = df['datetime'].dt.day
        df['Hour'] = df['datetime'].dt.hour

        self.energy_cost_data = df[['Month', 'Day', 'Hour', 'value']]

        if self.energy_cost_variability:
            self.energy_cost_data = ornstein_uhlenbeck_process(
                data=self.energy_cost_data, variability_config=self.energy_cost_variability)

    def observation(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Build the state observation by adding energy cost information.

        Args:
            obs (np.ndarray): Original observation.
            info (Dict[str, Any]): Information about the environment.
        Returns:
            np.ndarray: Transformed observation.
        """
        # Search for the index corresponding to the time of the current
        # observation.
        filter = (
            (self.energy_cost_data['Month'].to_numpy() == info['month']) &
            (self.energy_cost_data['Day'].to_numpy() == info['day']) &
            (self.energy_cost_data['Hour'].to_numpy() == info['hour'])
        )
        i = np.where(filter)[0][0]

        # Obtain energy cost observation
        selected_row = self.energy_cost_data.loc[i, ['value']].values

        # Flatten the selected rows
        obs = np.concatenate((obs, selected_row.ravel()))

        return obs


class DeltaTempWrapper(gym.ObservationWrapper):
    """Wrapper to add delta temperature information to the current observation. If setpoint variables
    has only one element, it will be considered as a unique setpoint for all temperature variables.
    IMPORTANT: temperature variables and setpoint of each zone must be defined in the same order."""

    logger = TerminalLogger().getLogger(name='WRAPPER DeltaTempWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 temperature_variables: List[str],
                 setpoint_variables: List[str]):
        """
        Args:
            env (Env): Original Gym environment.
            temperature_variables (List[str]): List of temperature variables.
            setpoint_variables (List[str]): List of setpoint variables. If the length is 1, it will be considered as a unique setpoint for all temperature variables.
        """
        super().__init__(env)

        # Check variables definition
        if len(setpoint_variables) != 1 and len(
                temperature_variables) != len(setpoint_variables):
            self.logger.error(
                'Setpoint variables must have one element length or the same length than temperature variables.'
                f'Current setpoint variables length: {setpoint_variables}')
            raise ValueError

        # Check all temperature and setpoint variables are in environment
        # observation variables
        if any(variable not in self.get_wrapper_attr('observation_variables')
                for variable in temperature_variables):
            self.logger.error(
                'Some temperature variables are not defined in observation space.')
            raise ValueError
        if any(variable not in self.get_wrapper_attr('observation_variables')
                for variable in setpoint_variables):
            self.logger.error(
                'Some setpoint variables are not defined in observation space.')
            raise ValueError

        # Define wrappers attributes
        self.delta_temperatures = temperature_variables
        self.delta_setpoints = setpoint_variables

        # Add delta temperature variables to observation variables
        new_observation_variables = deepcopy(
            self.get_wrapper_attr('observation_variables'))
        for temp_var in temperature_variables:
            new_observation_variables.append('delta_' + temp_var)
        self.observation_variables = new_observation_variables

        # Update observation space shape
        new_shape = self.env.get_wrapper_attr(
            'observation_space').shape[0] + len(temperature_variables)
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=(
                new_shape,
            ),
            dtype=self.env.get_wrapper_attr('observation_space').dtype)

        self.logger.info('Wrapper initialized.')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add delta temperature information to the current observation.
        """
        # Get obs dictionary
        obs_dict = dict(
            zip(self.env.get_wrapper_attr('observation_variables'), obs))

        # Get temperature values and setpoint(s) values
        temperatures = [obs_dict[variable]
                        for variable in self.delta_temperatures]
        setpoints = [obs_dict[variable] for variable in self.delta_setpoints]

        # Calculate delta values
        if len(setpoints) == 1:
            delta_temps = [temp - setpoints[0] for temp in temperatures]
        else:
            delta_temps = [temp - setpoint for temp, setpoint in zip(
                temperatures, setpoints)]

        # Update observation array appending delta values
        new_obs = np.concatenate((obs, delta_temps))

        return new_obs


# ---------------------------------------------------------------------------- #
#                                Action wrappers                               #
# ---------------------------------------------------------------------------- #


class IncrementalWrapper(gym.ActionWrapper):
    """A wrapper for an incremental values of desired action variables"""

    logger = TerminalLogger().getLogger(name='WRAPPER IncrementalWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(
        self,
        env: gym.Env,
        incremental_variables_definition: Dict[str, Tuple[float, float]],
        initial_values: List[float],
    ):
        """
        Args:
            env (gym.Env): Original Sinergym environment.
            incremental_variables_definition (Dict[str, Tuple[float, float]]): Dictionary defining incremental variables.
                                                                           Key: variable name, Value: Tuple with delta and step values.
                                                                           Delta: maximum range, Step: intermediate value jumps.
            initial_values (List[float]): Initial values for incremental variables. Length of this list and dictionary must match.
        """

        super().__init__(env)

        # Params
        self.current_values = np.array(initial_values, dtype=np.float32)

        # Check environment is valid
        if self.env.get_wrapper_attr('is_discrete'):
            self.logger.error(
                'Env wrapped by this wrapper must be continuous instead of discrete.')
            raise TypeError
        if any(variable not in self.env.get_wrapper_attr('action_variables')
               for variable in incremental_variables_definition.keys()):
            self.logger.error(
                'Some of the incremental variables specified does not exist as action variable in environment.')
            raise ValueError
        if len(initial_values) != len(incremental_variables_definition):
            self.logger.error(
                'Number of incremental variables does not match with initial values.')
            raise ValueError

        # All posible incremental variations
        self.values_definition = {}
        # Original action space variables
        action_space_low = deepcopy(
            self.env.get_wrapper_attr('action_space').low)
        action_space_high = deepcopy(
            self.env.get_wrapper_attr('action_space').high)
        # Calculating incremental variations and action space for each
        # incremental variable
        for variable, (delta_temp,
                       step_temp) in incremental_variables_definition.items():

            # Possible incrementations for each incremental variable.
            values = np.arange(
                step_temp,
                delta_temp +
                step_temp /
                10,
                step_temp)
            values = [v for v in [*-np.flip(values), 0, *values]]

            # Index of the action variable
            index = self.env.get_wrapper_attr(
                'action_variables').index(variable)

            self.values_definition[index] = values
            action_space_low[index] = min(values)
            action_space_high[index] = max(values)

        # New action space definition
        self.action_space = gym.spaces.Box(
            low=action_space_low,
            high=action_space_high,
            shape=self.env.get_wrapper_attr('action_space').shape,
            dtype=np.float32)

        self.logger.info(
            f'New incremental continuous action space: {self.action_space}')
        self.logger.info(
            f'Incremental variables configuration (variable: delta, step): {incremental_variables_definition}')
        self.logger.info('Wrapper initialized')

    def action(self, action):
        """Takes the continuous action and apply increment/decrement before to send to the next environment layer."""
        action_ = deepcopy(action)

        # Update current values with incremental values where required
        for i, (index, values) in enumerate(self.values_definition.items()):
            # Get increment value
            increment_value = action[index]
            # Round increment value to nearest value
            increment_value = min(
                values, key=lambda x: abs(
                    x - increment_value))
            # Update current_values
            self.current_values[i] += increment_value
            # Clip the value with original action space
            self.current_values[i] = max(
                self.env.get_wrapper_attr('action_space').low[index], min(
                    self.current_values[i], self.env.get_wrapper_attr('action_space').high[index]))

            action_[index] = self.current_values[i]

        return action_

# ---------------------------------------------------------------------------- #


class DiscreteIncrementalWrapper(gym.ActionWrapper):
    """A wrapper for an incremental setpoint discrete action space environment.
    WARNING: A environment with only temperature setpoints control must be used
    with this wrapper."""

    logger = TerminalLogger().getLogger(
        name='WRAPPER DiscreteIncrementalWrapper',
        level=LOG_WRAPPERS_LEVEL)

    def __init__(
        self,
        env: gym.Env,
        initial_values: List[float],
        delta_temp: float = 2.0,
        step_temp: float = 0.5,
    ):
        """
        Args:
            env: The original Sinergym env.
            action_names: Name of the action variables with the setpoint control you want to do incremental.
            initial_values: Initial values of the setpoints.
            delta_temp: Maximum temperature variation in the setpoints in one step.
            step_temp: Minimum temperature variation in the setpoints in one step.
        """

        super().__init__(env)

        # Params
        self.current_setpoints = np.array(initial_values, dtype=np.float32)

        # Check environment is valid
        if self.env.get_wrapper_attr('is_discrete'):
            self.logger.error(
                'Env wrapped by this wrapper must be continuous instead of discrete.')
            raise TypeError
        if len(self.get_wrapper_attr('current_setpoints')) != len(
                self.env.get_wrapper_attr('action_variables')):
            self.logger.error(
                'Number of variables is different from environment')
            raise ValueError

        # Define all posible setpoint variations
        values = np.arange(step_temp, delta_temp + step_temp / 10, step_temp)
        values = [v for v in [*values, *-values]]

        # Creating action_mapping function for the discrete environment
        self.mapping = {}
        do_nothing = np.array([0.0 for _ in range(len(self.env.get_wrapper_attr(
            'action_variables')))], dtype=np.float32)  # do nothing
        self.mapping[0] = do_nothing
        n = 1

        # Generate all posible actions
        for k in range(len(self.env.get_wrapper_attr('action_variables'))):
            for v in values:
                x = deepcopy(do_nothing)
                x[k] = v
                self.mapping[n] = np.array(x, dtype=np.float32)
                n += 1

        self.action_space = gym.spaces.Discrete(n)

        self.logger.info(f'New incremental action mapping: {n}')
        self.logger.info(f'{self.get_wrapper_attr('mapping')}')
        self.logger.info('Wrapper initialized')

    # Define action mapping method
    def action_mapping(self, action: int) -> np.ndarray:
        return self.mapping[action]

    def action(self, action: int) -> np.ndarray:
        """Takes the discrete action and transforms it to setpoints tuple."""
        action_ = deepcopy(action)
        action_ = self.get_wrapper_attr('action_mapping')(action_)
        # Update current setpoints values with incremental action
        self.current_setpoints = np.array([
            sum(i) for i in zip(
                self.get_wrapper_attr('current_setpoints'),
                action_)], dtype=np.float32)
        # clip setpoints returned
        self.current_setpoints = np.clip(
            self.get_wrapper_attr('current_setpoints'),
            self.env.get_wrapper_attr('action_space').low,
            self.env.get_wrapper_attr('action_space').high)

        return self.current_setpoints

    # Updating property
    @property  # pragma: no cover
    def is_discrete(self) -> bool:
        if isinstance(self.action_space, gym.spaces.Box):
            return False
        elif isinstance(self.action_space, gym.spaces.Discrete) or \
                isinstance(self.action_space, gym.spaces.MultiDiscrete) or \
                isinstance(self.action_space, gym.spaces.MultiBinary):
            return True
        else:
            self.logger.warning(
                'Action space is not continuous or discrete?')
            return False

# ---------------------------------------------------------------------------- #


class DiscretizeEnv(gym.ActionWrapper):
    """ Wrapper to discretize an action space.
    """

    logger = TerminalLogger().getLogger(name='WRAPPER DiscretizeEnv',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 discrete_space: Union[gym.spaces.Discrete,
                                       gym.spaces.MultiDiscrete,
                                       gym.spaces.MultiBinary],
                 action_mapping: Callable[[Union[int, List[int]]], np.ndarray]):
        """Wrapper for Discretize action space.

        Args:
            env (Env): Original environment.
            discrete_space (Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary]): Discrete Space.
            action_mapping (Callable[[Union[int,List[int]]], np.ndarray]): Function with action as argument, its output must match with original env action space, otherwise an error will be raised.
        """
        super().__init__(env)
        self.action_space = discrete_space
        self.action_mapping = action_mapping

        self.logger.info(
            f'New Discrete Space and mapping: {self.action_space}')
        self.logger.info(
            'Make sure that the action space is compatible and contained in the original environment.')
        self.logger.info('Wrapper initialized')

    def action(self, action: Union[int, List[int]]) -> np.ndarray:
        action_ = deepcopy(action)
        action_ = self.get_wrapper_attr(
            'action_mapping')(action_)
        return action_

    # Updating property
    @property  # pragma: no cover
    def is_discrete(self) -> bool:
        if isinstance(self.action_space, gym.spaces.Box):
            return False
        elif isinstance(self.action_space, gym.spaces.Discrete) or \
                isinstance(self.action_space, gym.spaces.MultiDiscrete) or \
                isinstance(self.action_space, gym.spaces.MultiBinary):
            return True
        else:
            self.logger.warning(
                'Action space is not continuous or discrete?')
            return False

# ---------------------------------------------------------------------------- #


class NormalizeAction(gym.ActionWrapper):
    """Wrapper to normalize action space.
    """

    logger = TerminalLogger().getLogger(name='WRAPPER NormalizeAction',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 normalize_range: Tuple[float, float] = (-1.0, 1.0)):
        """Wrapper to normalize action space in default continuous environment (not to combine with discrete environments). The action will be parsed to real action space before to send to the simulator (very useful ion DRL algorithms)

        Args:
            env (Env): Original environment.
            normalize_range (Tuple[float,float]): Range to normalize action variable values. Defaults to values between [-1.0,1.0].
        """
        super().__init__(env)

        # Ensure the action space is continuous
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.logger.critical(
                'The original environment must have a continuous action space.')
            raise TypeError

        self.real_space = deepcopy(env.action_space)
        lower_norm_value, upper_norm_value = normalize_range

        # Define the normalized action space
        action_dim = env.action_space.shape[0]
        self.normalized_space = gym.spaces.Box(
            low=np.full(action_dim, lower_norm_value, dtype=np.float32),
            high=np.full(action_dim, upper_norm_value, dtype=np.float32),
            dtype=np.float32
        )

        # Updated action space to normalized space
        self.action_space = self.normalized_space

        # Calculate the scale factor
        self.scale = (self.real_space.high - self.real_space.low) / \
            (self.normalized_space.high - self.normalized_space.low)

        self.logger.info(f'New normalized action space: {self.action_space}')
        self.logger.info('Wrapper initialized.')

    def reverting_action(self,
                         action: np.ndarray) -> np.ndarray:
        """ This method maps a normalized action in a real action space.

        Args:
            action (np.ndarray): Normalize action received in environment

        Returns:
            np.array: Action transformed in simulator real action space.
        """
        return self.real_space.low + \
            (action - self.normalized_space.low) * self.scale

    def action(self, action: np.ndarray) -> np.ndarray:
        return self.reverting_action(action)

# ---------------------------------------------------------------------------- #
#                                Reward Wrappers                               #
# ---------------------------------------------------------------------------- #


class MultiObjectiveReward(gym.Wrapper):

    logger = TerminalLogger().getLogger(name='WRAPPER MultiObjectiveReward',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self, env: Env, reward_terms: List[str]):
        """The environment will return a reward vector of each objective instead of a scalar value.

        Args:
            env (Env): Original Sinergym environment.
            reward_terms (List[str]): List of keys in reward terms which will be included in reward vector.
        """
        super().__init__(env)
        self.reward_terms = reward_terms

        self.logger.info('wrapper initialized.')

    def step(self, action: np.ndarray) -> Tuple[
            np.ndarray, List[float], bool, bool, Dict[str, Any]]:
        """Perform the action and environment return reward vector. If reward term is not in info reward_terms, it will be ignored.

        Args:
            action (np.ndarray): Action to be executed in environment.

        Returns:
            Tuple[ np.ndarray, List[float], bool, bool, Dict[str, Any]]: observation, vector reward, terminated, truncated and info.
        """
        # Execute normal reward
        obs, _, terminated, truncated, info = self.env.step(action)
        reward_vector = [value for key, value in info.items(
        ) if key in self.get_wrapper_attr('reward_terms')]
        return obs, reward_vector, terminated, truncated, info

# ---------------------------------------------------------------------------- #
#                                Others (Logger)                               #
# ---------------------------------------------------------------------------- #


class BaseLoggerWrapper(ABC, gym.Wrapper):

    def __init__(
        self,
        env: Env,
        storage_class: Callable = LoggerStorage
    ):
        """Base class for LoggerWrapper and its children classes.

        Args:
            env (Env): Original Sinergym environment.
            storage_class (Callable, optional): Storage class to be used. Defaults to Sinergym LoggerStorage class.
        """

        super().__init__(env)
        self.data_logger = storage_class()
        self.has_normalization = is_wrapped(self.env, NormalizeObservation)
        # Overwrite in case you want more metrics
        self.custom_variables: List[str] = []
        self.summary_metrics: List[str] = []

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment and the information logged."""
        # Reset logger data
        self.data_logger.reset_data()

        # Environment reset
        obs, info = self.env.reset(seed=seed, options=options)

        # Log reset observation
        if self.has_normalization:
            self.data_logger.log_norm_obs(obs)
            self.data_logger.log_obs(
                self.get_wrapper_attr('unwrapped_observation'))
        else:
            self.data_logger.log_obs(obs)

        self.data_logger.log_info(info)

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[
            np.ndarray, float, bool, bool, Dict[str, Any]]:

        # Environment step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Process custom_metrics
        custom_metrics = self.calculate_custom_metrics(obs,
                                                       action,
                                                       reward,
                                                       info,
                                                       terminated,
                                                       truncated)

        if self.has_normalization:
            self.data_logger.log_norm_obs(obs)

        log_data = {
            "obs": obs if not self.has_normalization else self.get_wrapper_attr('unwrapped_observation'),
            "action": action,
            "reward": reward,
            "info": info,
            "terminated": terminated,
            "truncated": truncated,
            "custom_metrics": custom_metrics}
        self.data_logger.log_interaction(**log_data)

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the environment and save normalization calibration."""
        # Reset logger data
        self.data_logger.reset_data()
        # Close the environment
        self.env.close()

    @abstractmethod  # pragma: no cover
    def calculate_custom_metrics(self,
                                 obs: np.ndarray,
                                 action: np.ndarray,
                                 reward: float,
                                 info: Dict[str, Any],
                                 terminated: bool,
                                 truncated: bool):
        """Calculate custom metrics from current interaction (or passed using self.data_logger attributes)

            Args:
                obs (np.ndarray): Observation from environment.
                action (np.ndarray): Action taken in environment.
                reward (float): Reward received from environment.
                info (Dict[str, Any]): Information from environment.
                terminated (bool): Flag to indicate if episode is terminated.
                truncated (bool): Flag to indicate if episode is truncated.
        """
        pass

    @abstractmethod  # pragma: no cover
    def get_episode_summary(self) -> Dict[str, float]:
        """Return data summary for the logger. This method should be implemented in the child classes.
           This method determines the data summary of episodes in Sinergym environments.

        Returns:
            Dict[str, float]: Data summary.
        """
        pass

# ---------------------------------------------------------------------------- #


class LoggerWrapper(BaseLoggerWrapper):

    logger = TerminalLogger().getLogger(name='WRAPPER LoggerWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(
        self,
        env: Env,
        storage_class: Callable = LoggerStorage
    ):
        """Wrapper to log data from environment interaction.

        Args:
            env (Env): Original Sinergym environment.
            storage_class (Callable, optional): Storage class to be used. Defaults to Sinergym LoggerStorage class.
        """
        super().__init__(env, storage_class)
        # Overwrite in case you want more metrics
        self.custom_variables = []
        # Overwite in case you have other summary metrics (same as
        # self.get_episode_summary return)
        self.summary_metrics = ['episode_num',
                                'mean_reward',
                                'std_reward',
                                'mean_reward_comfort_term',
                                'std_reward_comfort_term',
                                'mean_reward_energy_term',
                                'std_reward_energy_term',
                                'mean_comfort_penalty',
                                'std_comfort_penalty',
                                'mean_energy_penalty',
                                'std_energy_penalty',
                                'mean_temperature_violation',
                                'std_temperature_violation',
                                'mean_power_demand',
                                'std_power_demand',
                                'cumulative_power_demand',
                                'comfort_violation_time(%)',
                                'length(timesteps)',
                                'time_elapsed(hours)',
                                'terminated',
                                'truncated']
        self.logger.info('Wrapper initialized.')

    def calculate_custom_metrics(self,
                                 obs: np.ndarray,
                                 action: np.ndarray,
                                 reward: float,
                                 info: Dict[str, Any],
                                 terminated: bool,
                                 truncated: bool):
        return []

    def get_episode_summary(self) -> Dict[str, float]:
        # Get information from logger
        comfort_terms = [info['comfort_term']
                         for info in self.data_logger.infos[1:]]
        energy_terms = [info['energy_term']
                        for info in self.data_logger.infos[1:]]
        comfort_penalties = [info['comfort_penalty']
                             for info in self.data_logger.infos[1:]]
        energy_penalties = [info['energy_penalty']
                            for info in self.data_logger.infos[1:]]
        temperature_violations = [info['total_temperature_violation']
                                  for info in self.data_logger.infos[1:]]
        power_demands = [info['total_power_demand']
                         for info in self.data_logger.infos[1:]]
        try:
            comfort_violation_time = len(
                [value for value in temperature_violations if value > 0]) / self.get_wrapper_attr('timestep') * 100
        except ZeroDivisionError:
            comfort_violation_time = 0

        # Data summary
        data_summary = {
            'episode_num': self.get_wrapper_attr('episode'),
            'mean_reward': np.mean(
                self.data_logger.rewards),
            'std_reward': np.std(
                self.data_logger.rewards),
            'mean_reward_comfort_term': np.mean(comfort_terms),
            'std_reward_comfort_term': np.std(comfort_terms),
            'mean_reward_energy_term': np.mean(energy_terms),
            'std_reward_energy_term': np.std(energy_terms),
            'mean_comfort_penalty': np.mean(comfort_penalties),
            'std_comfort_penalty': np.std(comfort_penalties),
            'mean_energy_penalty': np.mean(energy_penalties),
            'std_energy_penalty': np.std(energy_penalties),
            'mean_temperature_violation': np.mean(temperature_violations),
            'std_temperature_violation': np.std(temperature_violations),
            'mean_power_demand': np.mean(power_demands),
            'std_power_demand': np.std(power_demands),
            'cumulative_power_demand': np.sum(power_demands),
            'comfort_violation_time(%)': comfort_violation_time,
            'length(timesteps)': self.get_wrapper_attr('timestep'),
            'time_elapsed(hours)': self.data_logger.infos[-1]['time_elapsed(hours)'],
            'terminated': self.data_logger.terminateds[-1],
            'truncated': self.data_logger.truncateds[-1],
        }
        return data_summary

# ---------------------------------------------------------------------------- #


class CSVLogger(gym.Wrapper):

    logger = TerminalLogger().getLogger(name='WRAPPER CSVLogger',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(
        self,
        env: Env,
        monitor: bool = True,
        info_excluded_keys: List[str] = ['reward',
                                         'action',
                                         'timestep',
                                         'month',
                                         'day',
                                         'hour',
                                         'time_elapsed(hours)',
                                         'reward_weight',
                                         'is_raining']
    ):
        """Logger to save logger data in CSV files while is running. It is required to be wrapped by a BaseLoggerWrapper child class previously.

        Args:
            env (Env): Original Gym environment in Sinergym.
            monitor (bool, optional): Flag to enable monitor data of all interactions. Defaults to True.
            info_excluded_keys (List[str], optional): List of keys in info dictionary to be excluded from CSV files. Defaults to ['reward', 'action', 'timestep', 'month', 'day', 'hour', 'time_elapsed(hours)', 'reward_weight', 'is_raining'].

        """
        super().__init__(env)

        self.info_excluded_keys = info_excluded_keys
        self.monitor = monitor

        # Check if it is wrapped by a BaseLoggerWrapper child class (required)
        if not is_wrapped(self.env, BaseLoggerWrapper):
            self.logger.error(
                'It is required to be wrapped by a BaseLoggerWrapper child class previously.')
            raise ValueError

        # Store paths to avoid redundant calls
        self.workspace_path = self.get_wrapper_attr('workspace_path')
        self.progress_file_path = os.path.join(
            self.workspace_path, 'progress.csv')
        self.weather_variability_config_path = os.path.join(
            self.workspace_path, 'weather_variability_config.csv')

        self.logger.info('Wrapper initialized.')

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment. Saving current logger episode summary and interaction in CSV files.

        Args:
        seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). If value is None, a seed will be chosen from some source of entropy. Defaults to None.
        options (Optional[Dict[str, Any]]): Additional information to specify how the environment is reset. Defaults to None.

        Returns:
        Tuple[np.ndarray, Dict[str, Any]]: Current observation and info context with additional information.
        """
        # If it is not the first episode
        if self.get_wrapper_attr('is_running'):
            # Log all episode information
            self.dump_log_files()
            self.logger.info(
                'End of episode detected, data updated in monitor and progress.csv.')

        return self.env.reset(seed=seed, options=options)

    def close(self) -> None:
        """Recording last episode summary and close env.
        """
        self.dump_log_files()
        self.logger.info(
            'Environment closed, data updated in monitor and progress.csv.')
        self.env.close()

    def dump_log_files(self) -> None:
        """Dump all logger data into CSV files.
        """

        episode_data = self.get_wrapper_attr('data_logger')

        if not episode_data.rewards:
            return

        # -------------------------------- Monitor.csv ------------------------------- #
        if self.monitor:

            self.monitor_path = os.path.join(
                self.get_wrapper_attr('episode_path'), 'monitor')
            os.makedirs(self.monitor_path, exist_ok=True)

            # Observations
            observation_variables = self.get_wrapper_attr(
                'observation_variables')
            self._save_csv(
                'observations.csv',
                observation_variables,
                episode_data.observations)

            # Normalized Observations
            if episode_data.normalized_observations:
                self._save_csv(
                    'normalized_observations.csv',
                    observation_variables,
                    episode_data.normalized_observations)

            # Rewards
            self._save_csv(
                'rewards.csv', ['reward'], [
                    [r] for r in episode_data.rewards])

            # Infos (excluding specified keys)
            filtered_infos = [
                [v for k, v in info.items() if k not in self.info_excluded_keys]
                for info in episode_data.infos[1:]  # Skip first (reset) row
            ]
            if filtered_infos:
                info_header = [
                    k for k in episode_data.infos[-1].keys() if k not in self.info_excluded_keys]
                # Including reset info step
                self._save_csv(
                    'infos.csv', info_header, [
                        [None] * len(info_header)] + filtered_infos)

            # Agent Actions
            action_variables = self.get_wrapper_attr('action_variables')
            self._save_csv(
                'agent_actions.csv', action_variables, [
                    [a] if not isinstance(
                        a, list) else a for a in episode_data.actions])

            # Simulated Actions
            simulated_actions = [[*info['action']] if isinstance(
                info['action'], list) else [info['action']] for info in episode_data.infos[1:]]
            self._save_csv(
                'simulated_actions.csv',
                action_variables,
                simulated_actions)

            # Custom Metrics (if available)
            if episode_data.custom_metrics:
                custom_variables = self.get_wrapper_attr('custom_variables')
                self._save_csv(
                    'custom_metrics.csv',
                    custom_variables,
                    episode_data.custom_metrics)

        # ------------------------------- Progress.csv ------------------------------- #
        episode_summary = self.get_wrapper_attr('get_episode_summary')()
        is_first_episode = self.get_wrapper_attr('episode') == 1

        with open(self.progress_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            if is_first_episode:
                writer.writerow(episode_summary.keys())
            writer.writerow(episode_summary.values())

        # ---------------------- Weather bariability config csv ---------------------- #
        modeling = self.get_wrapper_attr('model')

        if modeling.weather_variability_config:
            with open(self.weather_variability_config_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                if is_first_episode:
                    header = ['episode_num'] + [f"{var}_{param}"
                                                for var in modeling.weather_variability_config
                                                for param in ['sigma', 'mu', 'tau']]
                    writer.writerow(header)

                values = [
                    self.get_wrapper_attr('episode')] + [
                    val for params in modeling.weather_variability_config.values() for val in params]
                writer.writerow(values)

    def _save_csv(self, filename, header, rows):
        """Utility function to save CSV files safely."""
        try:
            with open(os.path.join(self.monitor_path, filename), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        except Exception as e:
            self.logger.error(f'Error writing {filename}: {e}')


try:
    import wandb

    class WandBLogger(gym.Wrapper):  # pragma: no cover

        logger = TerminalLogger().getLogger(name='WRAPPER WandBLogger',
                                            level=LOG_WRAPPERS_LEVEL)

        def __init__(self,
                     env: Env,
                     entity: Optional[str] = None,
                     project_name: Optional[str] = None,
                     run_name: Optional[str] = None,
                     group: Optional[str] = None,
                     job_type: Optional[str] = None,
                     tags: Optional[List[str]] = None,
                     episode_percentage: float = 0.9,
                     save_code: bool = False,
                     dump_frequency: int = 1000,
                     artifact_save: bool = True,
                     artifact_type: str = 'output',
                     excluded_info_keys: List[str] = ['reward',
                                                      'action',
                                                      'timestep',
                                                      'month',
                                                      'day',
                                                      'hour',
                                                      'time_elapsed(hours)',
                                                      'reward_weight',
                                                      'is_raining'],
                     excluded_episode_summary_keys: List[str] = ['terminated',
                                                                 'truncated']):
            """Wrapper to log data in WandB platform. It is required to be wrapped by a BaseLoggerWrapper child class previously.

            Args:
                env (Env): Original Sinergym environment.
                entity (Optional[str]): The entity to which the project belongs. If you are using a previous wandb run, it is not necessary to specify it. Defaults to None.
                project_name (Optional[str]): The project name. If you are using a previous wandb run, it is not necessary to specify it. Defaults to None.
                run_name (Optional[str]): The name of the run. Defaults to None (Sinergym env name + wandb unique identifier).
                group (Optional[str]): The name of the group to which the run belongs. Defaults to None.
                job_type (Optional[str]): The type of job. Defaults to None.
                tags (Optional[List[str]]): List of tags for the run. Defaults to None.
                episode_percentage (float): Percentage of episode which must be completed to log episode summary. Defaults to 0.9.
                save_code (bool): Whether to save the code in the run. Defaults to False.
                dump_frequency (int): Frequency to dump log in platform. Defaults to 1000.
                artifact_save (bool): Whether to save artifacts in WandB. Defaults to True.
                artifact_type (str): Type of artifact to save. Defaults to 'output'.
                excluded_info_keys (List[str]): List of keys to exclude from info dictionary. Defaults to ['reward', 'action', 'timestep', 'month', 'day', 'hour', 'time_elapsed(hours)', 'reward_weight', 'is_raining'].
                excluded_episode_summary_keys (List[str]): List of keys to exclude from episode summary. Defaults to ['terminated', 'truncated'].
            """
            super().__init__(env)

            # Check if logger is active (required)
            if not is_wrapped(self, BaseLoggerWrapper):
                self.logger.error(
                    'It is required to be wrapped by a BaseLoggerWrapper child class previously.')
                raise ValueError

            # Define wandb run name if is not specified
            run_name = run_name or f'{
                self.env.get_wrapper_attr('name')}_{
                wandb.util.generate_id()}'

            # Init WandB session
            # If there is no active run and entity and project has been specified,
            # initialize a new one using the parameters
            if not wandb.run and (entity and project_name):
                self.wandb_run = wandb.init(entity=entity,
                                            project=project_name,
                                            name=run_name,
                                            group=group,
                                            job_type=job_type,
                                            tags=tags,
                                            save_code=save_code,
                                            reinit=False)
            # If there is an active run
            elif wandb.run:
                # Use the active run
                self.wandb_run = wandb.run
            else:
                self.logger.error(
                    'Error initializing WandB run, if project and entity are not specified, it should be a previous active wandb run, but it has not been found.')
                raise RuntimeError

            # Flag to Wandb finish with env close
            self.wandb_finish = True

            # Define X-Axis for episode summaries
            self.wandb_run.define_metric(
                'episode_summaries/*',
                step_metric='episode_summaries/episode_num')

            # Attributes
            self.dump_frequency = dump_frequency
            self.artifact_save = artifact_save
            self.artifact_type = artifact_type
            self.episode_percentage = episode_percentage
            self.wandb_id = self.wandb_run.id
            self.excluded_info_keys = excluded_info_keys
            self.excluded_episode_summary_keys = excluded_episode_summary_keys
            self.global_timestep = 0

            self.logger.info('Wrapper initialized.')

        def step(self, action: np.ndarray
                 ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            """Sends action to the environment. Logging new interaction information in WandB platform.

            Args:
                action (np.ndarray): Action selected by the agent.

            Returns:
                Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
            """
            self.global_timestep += 1
            # Execute step ion order to get new observation and reward back
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Log step information if frequency is correct
            if self.global_timestep % self.dump_frequency == 0:
                self.logger.debug(
                    f'Dump frequency reached ({
                        self.global_timestep}), logging to WandB.')
                self.wandb_log()

            return obs, reward, terminated, truncated, info

        def reset(self,
                  seed: Optional[int] = None,
                  options: Optional[Dict[str,
                                         Any]] = None) -> Tuple[np.ndarray,
                                                                Dict[str,
                                                                     Any]]:
            """Reset the environment. Recording episode summary in WandB platform if it is not the first episode.

            Args:
                seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). if value is None, a seed will be chosen from some source of entropy. Defaults to None.
                options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

            Returns:
                Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
            """
            # It isn't the first episode simulation, so we can logger last
            # episode
            if self.get_wrapper_attr('is_running'):
                # Log all episode information
                if self.get_wrapper_attr(
                        'timestep') > self.episode_percentage * self.get_wrapper_attr('timestep_per_episode'):
                    self.wandb_log_summary()
                else:
                    self.logger.warning(
                        f'Episode ignored for log summary in WandB Platform, it has not be completed in at least {
                            self.episode_percentage * 100}%.')
                self.logger.info(
                    'End of episode detected, dumping summary metrics in WandB Platform.')

            # Then, reset environment
            obs, info = self.env.reset(seed=seed, options=options)

            return obs, info

        def close(self) -> None:
            """Recording last episode summary and close env.

            Args:
                wandb_finish (bool): Whether to finish WandB run. Defaults to True.
            """

            # Log last episode summary
            # Log all episode information
            if self.get_wrapper_attr('timestep') > self.episode_percentage * \
                    self.get_wrapper_attr('timestep_per_episode'):
                self.wandb_log_summary()
            else:
                self.logger.warning(
                    'Episode ignored for log summary in WandB Platform, it has not be completed in at least {self.episode_percentage * 100}%.')
            self.logger.info(
                'Environment closed, dumping summary metrics in WandB Platform.')

            # Finish WandB run
            if self.wandb_finish:
                # Save artifact
                if self.artifact_save:
                    self.save_artifact()
                self.wandb_run.finish()

            # Then, close env
            self.env.close()

        def wandb_log(self) -> None:
            """Log last step information in WandB platform.
            """

            # Interaction registration such as obs, action, reward...
            # (organized in a nested dictionary)
            log_dict = {}
            data_logger = self.get_wrapper_attr('data_logger')

            # OBSERVATION
            observation_variables = self.get_wrapper_attr(
                'observation_variables')
            log_dict['Observations'] = dict(
                zip(observation_variables, data_logger.observations[-1]))
            if is_wrapped(self, NormalizeObservation):
                log_dict['Normalized_observations'] = dict(
                    zip(observation_variables, data_logger.normalized_observations[-1]))

            # ACTION
            action_variables = self.get_wrapper_attr('action_variables')
            # Original action sent
            log_dict['Agent_actions'] = dict(
                zip(action_variables, data_logger.actions[-1]))
            # Action values performed in simulation
            log_dict['Simulation_actions'] = dict(
                zip(action_variables, data_logger.infos[-1]['action']))

            # REWARD
            log_dict['Reward'] = {'reward': data_logger.rewards[-1]}

            # INFO
            log_dict['Info'] = {
                key: float(value) for key,
                value in data_logger.infos[-1].items() if key not in self.excluded_info_keys}

            # CUSTOM METRICS
            if self.get_wrapper_attr('custom_variables'):
                log_dict['Variables_custom'] = dict(zip(self.get_wrapper_attr(
                    'custom_variables'), data_logger.custom_metrics[-1]))

            # Log in WandB
            self._log_data(log_dict)

        def wandb_log_summary(self) -> None:
            """Log episode summary in WandB platform.
            """
            if self.get_wrapper_attr('data_logger').rewards:
                # Get information from logger of LoggerWrapper
                episode_summary = self.get_wrapper_attr(
                    'get_episode_summary')()
                # Deleting excluded keys
                episode_summary = {key: value for key, value in episode_summary.items(
                ) if key not in self.get_wrapper_attr('excluded_episode_summary_keys')}
                # Log summary data in WandB
                self._log_data({'episode_summaries': episode_summary})

        def save_artifact(self) -> None:
            """Save sinergym output as artifact in WandB platform.
            """
            artifact = wandb.Artifact(
                name=self.wandb_run.name,
                type=self.artifact_type)
            artifact.add_dir(
                local_path=self.get_wrapper_attr('workspace_path'),
                name='Sinergym_output/')
            self.wandb_run.log_artifact(artifact)

        def set_wandb_finish(self, wandb_finish: bool) -> None:
            """Set if WandB run must be finished when environment is closed.

            Args:
                wandb_finish (bool): Whether to finish WandB run.
            """
            self.wandb_finish = wandb_finish

        def _log_data(self, data: Dict[str, Any]) -> None:
            """Log data in WandB platform. Nesting the dictionary correctly in different sections.

            Args:
                data (Dict[str, Any]): Dictionary with data to be logged.
            """

            for key, value in data.items():
                if isinstance(value, dict):
                    self.wandb_run.log(
                        {f'{key}/{k}': v for k, v in value.items()}, step=self.global_timestep)
                else:
                    self.wandb_run.log({key: value}, step=self.global_timestep)
except ImportError:
    class WandBLogger():  # pragma: no cover
        logger = TerminalLogger().getLogger(name='WRAPPER WandBLogger',
                                            level=LOG_WRAPPERS_LEVEL)
        """Wrapper to log data in WandB platform. It is required to be wrapped by a BaseLoggerWrapper child class previously.
        """

        def __init__(self, env: Env):
            self.logger.warning(
                'WandB is not installed. Please install it to use WandBLogger.')


# ---------------------------------------------------------------------------- #


class ReduceObservationWrapper(gym.Wrapper):

    logger = TerminalLogger().getLogger(
        name='WRAPPER ReduceObservationWrapper',
        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 obs_reduction: List[str]):
        """Wrapper to reduce the observation space of the environment. These variables removed from
        the space are included in the info dictionary. This way they are recordable but not used in DRL process.

        Args:
            env (Env): Original environment.
            obs_reduction (List[str]): List of observation variables to be removed.
        """
        super().__init__(env)

        # Check if the variables to be removed are in the observation space
        original_obs_vars = self.env.get_wrapper_attr('observation_variables')
        missing_vars = [
            var for var in obs_reduction if var not in original_obs_vars]
        if missing_vars:
            self.logger.error(
                f'Some observation variables to be removed are not defined: {missing_vars}')
            raise ValueError

        # Calculate index of variables to keep
        self.keep_index = np.array([i for i, var in enumerate(
            original_obs_vars) if var not in obs_reduction])

        # Update observation variables
        self.observation_variables = [
            var for var in original_obs_vars if var not in obs_reduction]
        self.removed_observation_variables = obs_reduction

        # Update observation space
        original_obs_space = self.env.get_wrapper_attr('observation_space')
        self.observation_space = gym.spaces.Box(
            low=original_obs_space.low[0],
            high=original_obs_space.high[0],
            shape=(len(self.observation_variables),),
            dtype=original_obs_space.dtype
        )

        self.logger.info('Wrapper initialized.')

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Separating removed variables from observation values and adding it to info dict.

        Args:
            action (np.ndarray): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Filter removed variables from observation using precalculated index
        reduced_obs = obs[self.keep_index]

        return reduced_obs, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Sends action to the environment. Separating removed variables from observation values and adding it to info dict"""
        obs, info = self.env.reset(seed=seed, options=options)

        # Filter removed variables from observation using precalculated index
        reduced_obs = obs[self.keep_index]

        return reduced_obs, info

# ---------------------------------------------------------------------------- #
#                      Real-time building context wrappers                     #
# ---------------------------------------------------------------------------- #


class VariabilityContextWrapper(gym.Wrapper):

    logger = TerminalLogger().getLogger(
        name='WRAPPER VariabilityContextWrapper',
        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env,
                 context_space: gym.spaces.Box,
                 delta_value: float = 1.0,
                 step_frequency_range: Tuple[int, int] = (96, 96 * 7),
                 ):
        """Wrapper that modifies the environment's context variables at random intervals.

        Args:
            env (gym.Env): The environment to wrap.
            context_space (gym.spaces.Box): The space defining valid context variable values.
            delta_value (float): Maximum absolute change applied to context variables at each update.
            step_frequency_range (Tuple[int, int]): Range for the number of steps before each update.
        """
        super().__init__(env)

        # Validations
        if not isinstance(context_space, gym.spaces.Box):
            self.logger.error(
                'context_space must be an instance of gym.spaces.Box.')
            raise TypeError

        context_variables = self.get_wrapper_attr('context_variables')
        if context_space.shape[0] != len(context_variables):
            self.logger.error(
                f'Context space shape ({context_space.shape[0]}) is not coherent with '
                f'environment context variables ({len(context_variables)}).')
            raise ValueError

        if delta_value <= 0:
            self.logger.error(
                f'Delta value must be > 0, but received {delta_value}.')
            raise ValueError

        if not (isinstance(step_frequency_range, tuple)
                and len(step_frequency_range) == 2):
            self.logger.error(
                f'Invalid step_frequency_range: {step_frequency_range}. Must be a tuple (min, max).')
            raise ValueError

        if step_frequency_range[0] <= 0 or step_frequency_range[0] >= step_frequency_range[1]:
            self.logger.error(
                f'Invalid step_frequency_range: {step_frequency_range}.')
            raise ValueError

        # Initialization
        self.context_space = context_space
        self.delta_context = (-delta_value, delta_value)
        self.step_frequency_range = step_frequency_range

        initial_context = self.get_wrapper_attr('initial_context')
        if initial_context is not None:
            self.current_context = np.array(
                initial_context, np.float32)
        else:
            self.current_context = np.random.uniform(
                self.context_space.low,
                self.context_space.high,
                size=self.context_space.shape[0]).astype(np.float32)

        self.next_context_values, self.next_step_update = self._generate_context_values()

        self.logger.info('Wrapper initialized.')

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes an action and updates the environment's context if needed.

        Args:
            action (np.ndarray): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """

        # Discount frequency
        self.next_step_update -= 1

        if self.next_step_update == 0:
            # Update context
            self.get_wrapper_attr('update_context')(self.next_context_values)
            self.current_context = self.next_context_values
            self.logger.info(
                f'Context updated with values: {
                    self.next_context_values}')
            # Calculate next update
            self.next_context_values, self.next_step_update = self._generate_context_values()

        obs, reward, terminated, truncated, info = self.env.step(action)

        return obs, reward, terminated, truncated, info

    def _generate_context_values(self) -> Tuple[np.ndarray, int]:
        """Generates new context values and determines the next update step.

        Returns:
            Tuple[np.ndarray, int]:
                - The new context values after applying random deltas.
                - The number of steps until the next update.
        """
        # Generate random delta context values
        delta_context_values = np.random.uniform(
            self.delta_context[0],
            self.delta_context[1],
            size=self.current_context.shape[0]).astype(np.float32)
        # Apply delta values and clip with context space
        next_context_values = np.clip(
            self.current_context +
            delta_context_values,
            self.context_space.low,
            self.context_space.high).astype(np.float32)

        # Generate random step frequency
        next_step_update = np.random.randint(
            self.step_frequency_range[0], self.step_frequency_range[1])

        return next_context_values, next_step_update

# ---------------------------------------------------------------------------- #
#                         Specific environment wrappers                        #
# ---------------------------------------------------------------------------- #


class OfficeGridStorageSmoothingActionConstraintsWrapper(
        gym.ActionWrapper):  # pragma: no cover
    def __init__(self, env):
        if env.get_wrapper_attr('building_path').split(
                '/')[-1] != 'OfficeGridStorageSmoothing.epJSON':
            raise ValueError(
                'OfficeGridStorageSmoothingActionConstraintsWrapper: This wrapper is not valid for this environment.')
        super().__init__(env)

    def action(self, act: np.ndarray) -> np.ndarray:
        """Due to Charge rate and Discharge rate can't be more than 0.0 simultaneously (in OfficeGridStorageSmoothing.epJSON),
           this wrapper clips one of them to 0.0 when both have a value upper than 0.0 (randomly).

        Args:
            act (np.ndarray): Action to be clipped

        Returns:
            np.ndarray: Action Clipped
        """
        if self.get_wrapper_attr('flag_discrete'):
            null_value = 0.0
        else:
            # -1.0 is 0.0 when action space transformation to simulator action space.
            null_value = -1.0
        if act[2] > null_value and act[3] > null_value:
            random_rate_index = np.random.randint(2, 4)
            act[random_rate_index] = null_value
        return act
