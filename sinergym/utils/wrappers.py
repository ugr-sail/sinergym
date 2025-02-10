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
import wandb
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
    """Wrapper to substitute day value by is_weekend flag, and hour and month by sin and cos values.
       Observation space is updated automatically."""

    logger = TerminalLogger().getLogger(name='WRAPPER DatetimeWrapper',
                                        level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: Env):
        super(DatetimeWrapper, self).__init__(env)

        # Check datetime variables are defined in environment
        try:
            assert all(time_variable in self.get_wrapper_attr('observation_variables')
                       for time_variable in ['month', 'day_of_month', 'hour'])
        except AssertionError as err:
            self.logger.error(
                'month, day_of_month and hour must be defined in observation space in environment previously.')
            raise err

        # Update new shape
        new_shape = self.env.get_wrapper_attr('observation_space').shape[0] + 2
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=(
                new_shape,
            ),
            dtype=self.env.get_wrapper_attr('observation_space').dtype)
        # Update observation variables
        new_observation_variables = deepcopy(
            self.get_wrapper_attr('observation_variables'))

        day_index = new_observation_variables.index('day_of_month')
        new_observation_variables[day_index] = 'is_weekend'
        hour_index = new_observation_variables.index('hour')
        new_observation_variables[hour_index] = 'hour_cos'
        new_observation_variables.insert(hour_index + 1, 'hour_sin')
        month_index = new_observation_variables.index('month')
        new_observation_variables[month_index] = 'month_cos'
        new_observation_variables.insert(month_index + 1, 'month_sin')

        self.observation_variables = new_observation_variables

        self.logger.info('Wrapper initialized.')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies calculation in is_weekend flag, and sen and cos in hour and month

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Transformed observation.
        """
        # Get obs_dict with observation variables from original env
        obs_dict = dict(
            zip(self.env.get_wrapper_attr('observation_variables'), obs))

        # New obs dict with same values than obs_dict but with new fields with
        # None
        new_obs = dict.fromkeys(self.get_wrapper_attr('observation_variables'))
        for key, value in obs_dict.items():
            if key in new_obs.keys():
                new_obs[key] = value
        dt = datetime(
            int(obs_dict['year']) if obs_dict.get('year', False) else YEAR,
            int(obs_dict['month']),
            int(obs_dict['day_of_month']),
            int(obs_dict['hour']))
        # Update obs
        new_obs['is_weekend'] = 1.0 if dt.isoweekday() in [6, 7] else 0.0
        new_obs['hour_cos'] = np.cos(2 * np.pi * obs_dict['hour'] / 24)
        new_obs['hour_sin'] = np.sin(2 * np.pi * obs_dict['hour'] / 24)
        new_obs['month_cos'] = np.cos(2 * np.pi * (obs_dict['month'] - 1) / 12)
        new_obs['month_sin'] = np.sin(2 * np.pi * (obs_dict['month'] - 1) / 12)

        return np.array(list(new_obs.values()))

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
        super(PreviousObservationWrapper, self).__init__(env)
        # Check and apply previous variables to observation space and variables
        # names
        self.previous_variables = previous_variables
        new_observation_variables = deepcopy(
            self.get_wrapper_attr('observation_variables'))
        for obs_var in previous_variables:
            assert obs_var in self.get_wrapper_attr(
                'observation_variables'), '{} variable is not defined in observation space, revise the name.'.format(obs_var)
            new_observation_variables.append(obs_var + '_previous')
        # Update observation variables
        self.observation_variables = new_observation_variables
        # Update new shape
        new_shape = self.env.get_wrapper_attr(
            'observation_space').shape[0] + len(previous_variables)
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=(
                new_shape,
            ),
            dtype=self.env.get_wrapper_attr('observation_space').dtype)

        # previous observation initialization
        self.previous_observation = np.zeros(
            shape=len(previous_variables), dtype=np.float32)

        self.logger.info('Wrapper initialized.')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add previous observation to the current one

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: observation with
        """
        # Concatenate current obs with previous observation variables
        new_obs = np.concatenate(
            (obs, self.get_wrapper_attr('previous_observation')))
        # Update previous observation to current observation
        self.previous_observation = []
        for variable in self.previous_variables:
            index = self.env.get_wrapper_attr(
                'observation_variables').index(variable)
            self.previous_observation.append(obs[index])

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
        super(MultiObsWrapper, self).__init__(env)
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

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

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
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)

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
        gym.Wrapper.__init__(self, env)

        # Check mean and var format if it is defined
        mean = self._check_and_update_metric(mean, 'mean')
        var = self._check_and_update_metric(var, 'var')

        self.num_envs = 1
        self.is_vector_env = False
        self.automatic_update = automatic_update

        self.unwrapped_observation = None
        # Initialize normalization calibration
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.obs_rms.mean = mean if mean is not None else self.obs_rms.mean
        self.obs_rms.var = var if var is not None else self.obs_rms.var
        self.epsilon = epsilon

        self.logger.info('Wrapper initialized.')

    def step(self, action: Union[int, np.ndarray]) -> Tuple[
            np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Steps through the environment and normalizes the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save original obs in class attribute
        self.unwrapped_observation = deepcopy(obs)

        # Normalize observation and return
        return self.normalize(np.array([obs]))[
            0], reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""

        # Update normalization calibration if it is required
        if self.get_wrapper_attr('episode') > 0:
            self._save_normalization_calibration()

        obs, info = self.env.reset(**kwargs)

        # Save original obs in class attribute
        self.unwrapped_observation = deepcopy(obs)

        return self.normalize(np.array([obs]))[0], info

    def close(self):
        """save normalization calibration and close the environment."""
        # Update normalization calibration if it is required
        if self.get_wrapper_attr('episode') > 0:
            self._save_normalization_calibration()

        self.env.close()

# ----------------------- Wrapper extra functionality ----------------------- #

    def _check_and_update_metric(self, metric, metric_name):
        if metric is not None:
            # Check type and conversions
            if isinstance(metric, str):
                try:
                    metric = np.loadtxt(metric)
                except FileNotFoundError as err:
                    self.logger.error(
                        '{}.txt file not found. Please, check the path.'.format(metric_name))
                    raise err
            elif isinstance(metric, list) or isinstance(metric, np.ndarray):
                metric = np.float64(metric)
            else:
                self.logger.error(
                    '{} values must be a list, a numpy array or a path to a txt file.'.format(metric_name))
                raise ValueError

            # Check dimension of mean and var
            try:
                assert len(metric) == self.observation_space.shape[0]
            except AssertionError as err:
                self.logger.error(
                    '{} values must have the same shape than environment observation space.'.format(metric_name))
                raise err

        return metric

    def _save_normalization_calibration(self):
        """Saves the normalization calibration data in the output folder as txt files.
        """
        self.logger.info(
            'Saving normalization calibration data.')
        # Save in txt in episode output folder
        np.savetxt(fname=self.get_wrapper_attr(
            'episode_path') + '/mean.txt', X=self.mean)
        np.savetxt(fname=self.get_wrapper_attr(
            'episode_path') + '/var.txt', X=self.var)
        # Overwrite output root folder mean and var as latest calibration
        np.savetxt(fname=self.get_wrapper_attr(
            'workspace_path') + '/mean.txt', X=self.mean)
        np.savetxt(fname=self.get_wrapper_attr(
            'workspace_path') + '/var.txt', X=self.var)

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

    def set_mean(self, mean: Union[list, np.float64, str]):
        """Sets the mean value of the observations."""
        mean = self._check_and_update_metric(mean, 'mean')
        self.obs_rms.mean = deepcopy(mean)

    def set_var(self, var: Union[list, np.float64, str]):
        """Sets the variance value of the observations."""
        var = self._check_and_update_metric(var, 'var')
        self.obs_rms.var = deepcopy(var)

    def normalize(self, obs):
        """Normalizes the observation using the running mean and variance of the observations.
        If automatic_update is enabled, the running mean and variance will be updated too."""
        if self.automatic_update:
            self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / \
            np.sqrt(self.obs_rms.var + self.epsilon)


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

        super(WeatherForecastingWrapper, self).__init__(env)
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

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Performs the action in the new environment.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

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

        super(EnergyCostWrapper, self).__init__(env)
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

    def step(self, action: Union[int, np.ndarray]

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
        super(DeltaTempWrapper, self).__init__(env)

        # Check variables definition
        assert len(setpoint_variables) == 1 or len(setpoint_variables) == len(
            temperature_variables), 'Setpoint variables must have one element length or the same length than temperature variables.'

        # Check all temperature and setpoint variables are in environment
        # observation variables
        assert all([variable in self.get_wrapper_attr('observation_variables')
                    for variable in temperature_variables]), 'Some temperature variables are not defined in observation space.'
        assert all([variable in self.get_wrapper_attr('observation_variables')
                    for variable in setpoint_variables]), 'Some setpoint variables are not defined in observation space.'

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
        try:
            assert not self.env.get_wrapper_attr('is_discrete')
        except AssertionError as err:
            self.logger.error(
                'Env wrapped by this wrapper must be continuous.')
            raise err
        try:
            assert all([variable in self.env.get_wrapper_attr('action_variables')
                       for variable in list(incremental_variables_definition.keys())])
        except AssertionError as err:
            self.logger.error(
                'Some of the incremental variables specified does not exist as action variable in environment.')
            raise err
        try:
            assert len(initial_values) == len(
                incremental_variables_definition)
        except AssertionError as err:
            self.logger.error(
                'Number of incremental variables does not match with initial values')
            raise err

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
            'New incremental continuous action space: {}'.format(
                self.action_space))
        self.logger.info(
            'Incremental variables configuration (variable: delta, step): {}'.format(
                incremental_variables_definition))
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
        try:
            assert not self.env.get_wrapper_attr('is_discrete')
        except AssertionError as err:
            self.logger.error(
                'Env wrapped by this wrapper must be continuous.')
            raise err
        try:
            assert len(
                self.get_wrapper_attr('current_setpoints')) == len(
                self.env.get_wrapper_attr('action_variables'))
        except AssertionError as err:
            self.logger.error(
                'Number of variables is different from environment')
            raise err

        # Define all posible setpoint variations
        values = np.arange(step_temp, delta_temp + step_temp / 10, step_temp)
        values = [v for v in [*values, *-values]]

        # Creating action_mapping function for the discrete environment
        self.mapping = {}
        do_nothing = [0.0 for _ in range(
            len(self.env.get_wrapper_attr('action_variables')))]  # do nothing
        self.mapping[0] = do_nothing
        n = 1

        # Generate all posible actions
        for k in range(len(self.env.get_wrapper_attr('action_variables'))):
            for v in values:
                x = deepcopy(do_nothing)
                x[k] = v
                self.mapping[n] = x
                n += 1

        self.action_space = gym.spaces.Discrete(n)

        self.logger.info('New incremental action mapping: {}'.format(n))
        self.logger.info('{}'.format(self.get_wrapper_attr('mapping')))
        self.logger.info('Wrapper initialized')

    # Define action mapping method
    def action_mapping(self, action: int) -> np.ndarray:
        return np.array(self.mapping[action], dtype=np.float32)

    def action(self, action):
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
                 action_mapping: Callable[[Union[int,
                                                 List[int]]],
                                          Union[float,
                                                List[float]]]):
        """Wrapper for Discretize action space.

        Args:
            env (Env): Original environment.
            discrete_space (Union[gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary]): Discrete Space.
            action_mapping (Callable[[Union[int, List[int]]], Union[float, List[float]]]): Function with action as argument, its output must match with original env action space, otherwise an error will be raised.
        """
        super().__init__(env)
        self.action_space = discrete_space
        self.action_mapping = action_mapping

        self.logger.info(
            'New Discrete Space and mapping: {}'.format(
                self.action_space))
        self.logger.info(
            'Make sure that the action space is compatible and contained in the original environment.')
        self.logger.info('Wrapper initialized')

    def action(self, action: Union[int, List[int]]) -> np.ndarray:
        action_ = deepcopy(action)
        action_ = np.array(self.get_wrapper_attr(
            'action_mapping')(action_), dtype=np.float32)
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

        # Checks
        try:
            assert not self.get_wrapper_attr('is_discrete')
        except AssertionError as err:
            self.logger.critical(
                'The original environment must be continuous')
            raise err

        # Define real space for simulator
        self.real_space = deepcopy(self.action_space)
        # Define normalize space
        lower_norm_value, upper_norm_value = normalize_range
        self.normalized_space = gym.spaces.Box(
            low=np.array(
                np.repeat(
                    lower_norm_value,
                    env.get_wrapper_attr('action_space').shape[0]),
                dtype=np.float32),
            high=np.array(
                np.repeat(
                    upper_norm_value,
                    env.get_wrapper_attr('action_space').shape[0]),
                dtype=np.float32),
            dtype=env.get_wrapper_attr('action_space').dtype)
        # Updated action space to normalized space
        self.action_space = self.normalized_space

        self.logger.info(
            'New normalized action Space: {}'.format(
                self.action_space))
        self.logger.info('Wrapper initialized')

    def reverting_action(self,
                         action: Any):
        """ This method maps a normalized action in a real action space.

        Args:
            action (Any): Normalize action received in environment

        Returns:
            np.array: Action transformed in simulator real action space.
        """

        # Convert action to the original action space
        action_ = (action - self.normalized_space.low) * (self.real_space.high - self.real_space.low) / \
            (self.normalized_space.high - self.normalized_space.low) + self.real_space.low

        return action_

    def action(self, action: Any):
        return self.get_wrapper_attr('reverting_action')(action)

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
        super(MultiObjectiveReward, self).__init__(env)
        self.reward_terms = reward_terms

        self.logger.info('wrapper initialized.')

    def step(self, action: Union[int, np.ndarray]) -> Tuple[
            np.ndarray, List[float], bool, bool, Dict[str, Any]]:
        """Perform the action and environment return reward vector. If reward term is not in info reward_terms, it will be ignored.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

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

        super(BaseLoggerWrapper, self).__init__(env)
        self.data_logger = storage_class()
        # Overwrite in case you want more metrics
        self.custom_variables = []
        self.summary_metrics = []

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
        if is_wrapped(self.env, NormalizeObservation):
            self.data_logger.log_norm_obs(obs)
            self.data_logger.log_obs(
                self.get_wrapper_attr('unwrapped_observation'))
        else:
            self.data_logger.log_obs(obs)

        self.data_logger.log_info(info)

        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[
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

        # Log interaction information of step
        if is_wrapped(self.env, NormalizeObservation):
            self.data_logger.log_norm_obs(obs)
            self.data_logger.log_interaction(
                obs=self.get_wrapper_attr('unwrapped_observation'),
                action=action,
                reward=reward,
                info=info,
                terminated=terminated,
                truncated=truncated,
                custom_metrics=custom_metrics)
        else:
            self.data_logger.log_interaction(
                obs=obs,
                action=action,
                reward=reward,
                info=info,
                terminated=terminated,
                truncated=truncated,
                custom_metrics=custom_metrics)

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
                                 action: Union[int, np.ndarray],
                                 reward: float,
                                 info: Dict[str, Any],
                                 terminated: bool,
                                 truncated: bool):
        """Calculate custom metrics from current interaction (or passed using self.data_logger attributes)

            Args:
                obs (np.ndarray): Observation from environment.
                action (Union[int, np.ndarray]): Action taken in environment.
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
        super(LoggerWrapper, self).__init__(env, storage_class)
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
                                'mean_abs_comfort_penalty',
                                'std_abs_comfort_penalty',
                                'mean_abs_energy_penalty',
                                'std_abs_energy_penalty',
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
                                 action: Union[int, np.ndarray],
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
        abs_comfort_penalties = [info['abs_comfort_penalty']
                                 for info in self.data_logger.infos[1:]]
        abs_energy_penalties = [info['abs_energy_penalty']
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
            'mean_abs_comfort_penalty': np.mean(abs_comfort_penalties),
            'std_abs_comfort_penalty': np.std(abs_comfort_penalties),
            'mean_abs_energy_penalty': np.mean(abs_energy_penalties),
            'std_abs_energy_penalty': np.std(abs_energy_penalties),
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
            info_excluded_keys (List[str], optional): List of keys in info dictionary to be excluded from CSV files. Defaults to ['reward', 'action', 'timestep', 'month', 'day', 'hour', 'time_elapsed(hours)', 'reward_weight', 'is_raining'].

        """
        super(CSVLogger, self).__init__(env)
        self.info_excluded_keys = info_excluded_keys

        try:
            assert is_wrapped(self.env, BaseLoggerWrapper)
        except AssertionError as err:
            self.logger.error(
                'It is required to be wrapped by a BaseLoggerWrapper child class previously.')
            raise err

        self.progress_file_path = self.get_wrapper_attr(
            'workspace_path') + '/progress.csv'

        self.weather_variability_config_path = self.get_wrapper_attr(
            'workspace_path') + '/weather_variability_config.csv'

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
        # reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        return obs, info

    def close(self) -> None:
        """Recording last episode summary and close env.
        """

        # Log csv files
        self.dump_log_files()
        self.logger.info(
            'Environment closed, data updated in monitor and progress.csv.')

        # Close env, updating the logger information
        self.env.close()

    def dump_log_files(self) -> None:
        """Dump all logger data in CSV files.
        """
        monitor_path = self.get_wrapper_attr('episode_path') + '/monitor'
        os.makedirs(monitor_path, exist_ok=True)
        episode_data = self.get_wrapper_attr('data_logger')

        if len(episode_data.rewards) > 0:

            # Observations
            with open(monitor_path + '/observations.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.get_wrapper_attr('observation_variables'))
                writer.writerows(episode_data.observations)

            # Normalized Observations
            if len(episode_data.normalized_observations) > 0:
                with open(monitor_path + '/normalized_observations.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        self.get_wrapper_attr('observation_variables'))
                    writer.writerows(episode_data.normalized_observations)

            # Rewards
            with open(monitor_path + '/rewards.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['reward'])
                for reward in episode_data.rewards:
                    writer.writerow([reward])

            # Infos (except excluded keys)
            with open(monitor_path + '/infos.csv', 'w') as f:
                writer = csv.writer(f)
                column_names = [key for key in episode_data.infos[-1].keys(
                ) if key not in self.get_wrapper_attr('info_excluded_keys')]
                # Skip reset row
                rows = [[value for key, value in info.items() if key not in self.get_wrapper_attr(
                    'info_excluded_keys')] for info in episode_data.infos[1:]]
                writer.writerow(column_names)
                # write null row for reset
                writer.writerow([None for _ in range(len(column_names))])
                writer.writerows(rows)

            # Agent Actions
            with open(monitor_path + '/agent_actions.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.get_wrapper_attr('action_variables'))
                if isinstance(episode_data.actions[0], list):
                    writer.writerows(episode_data.actions)
                else:
                    for action in episode_data.actions:
                        writer.writerow([action])

            # Simulated actions
            with open(monitor_path + '/simulated_actions.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.get_wrapper_attr('action_variables'))
                # reset_action = [None for _ in range(
                #    len(self.get_wrapper_attr('action_variables')))]
                simulated_actions = [list(info['action'])
                                     for info in episode_data.infos[1:]]
                if isinstance(simulated_actions[0], list):
                    writer.writerows(simulated_actions)
                else:
                    for action in simulated_actions:
                        writer.writerow([action])

            # Custom metrics
            if len(episode_data.custom_metrics) > 0:
                with open(monitor_path + '/custom_metrics.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.get_wrapper_attr('custom_variables'))
                    writer.writerows(episode_data.custom_metrics)

            # Update progress.csv
            episode_summary = self.get_wrapper_attr('get_episode_summary')()

            with open(self.get_wrapper_attr('progress_file_path'), 'a+') as f:
                writer = csv.writer(f)
                # If first episode, write header
                if self.get_wrapper_attr('episode') == 1:
                    writer.writerow(list(episode_summary.keys()))
                writer.writerow(list(episode_summary.values()))

            # Update weather_variability_config if exists
            modeling = self.get_wrapper_attr('model')
            config_path = self.get_wrapper_attr(
                'weather_variability_config_path')

            if modeling.weather_variability_config is not None:
                with open(config_path, 'a+') as f:
                    writer = csv.writer(f)

                    # If first episode, write header
                    if self.get_wrapper_attr('episode') == 1:
                        header = ['episode_num'] + [
                            f"{var_name}_{var_param}"
                            for var_name in list(modeling.weather_variability_config.keys())
                            for var_param in ['sigma', 'mu', 'tau']
                        ]
                        writer.writerow(header)

                    # Write OU params for each weather variable
                    var_values = list()
                    var_values = [
                        self.get_wrapper_attr('episode')] + [
                        value for params in modeling.weather_variability_config.values() for value in params]
                    writer.writerow(var_values)


# ---------------------------------------------------------------------------- #

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
        super(WandBLogger, self).__init__(env)

        # Check if logger is active
        try:
            assert is_wrapped(self, BaseLoggerWrapper)
        except AssertionError as err:
            self.logger.error(
                'It is required to be wrapped by a BaseLoggerWrapper child class previously.')
            raise err

        # Define wandb run name if is not specified
        run_name = run_name if run_name is not None else self.env.get_wrapper_attr(
            'name') + '_' + wandb.util.generate_id()

        # Init WandB session
        # If there is no active run and entity and project has been specified,
        # initialize a new one using the parameters
        if wandb.run is None and (
                entity is not None and project_name is not None):
            self.wandb_run = wandb.init(entity=entity,
                                        project=project_name,
                                        name=run_name,
                                        group=group,
                                        job_type=job_type,
                                        tags=tags,
                                        save_code=save_code,
                                        reinit=False)
        # If there is an active run
        elif wandb.run is not None:
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

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Logging new interaction information in WandB platform.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """
        self.global_timestep += 1
        # Execute step ion order to get new observation and reward back
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Log step information if frequency is correct
        if self.global_timestep % self.dump_frequency == 0:
            self.logger.debug(
                'Dump Frequency reached in timestep {}, dumping data in WandB.'.format(
                    self.global_timestep))
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
        # It isn't the first episode simulation, so we can logger last episode
        if self.get_wrapper_attr('is_running'):
            # Log all episode information
            if self.get_wrapper_attr(
                    'timestep') > self.episode_percentage * self.get_wrapper_attr('timestep_per_episode'):
                self.wandb_log_summary()
            else:
                self.logger.warning(
                    'Episode ignored for log summary in WandB Platform, it has not be completed in at least {}%.'.format(
                        self.episode_percentage * 100))
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
                'Episode ignored for log summary in WandB Platform, it has not be completed in at least {}%.'.format(
                    self.episode_percentage * 100))
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
        if is_wrapped(self, NormalizeObservation):
            log_dict['Normalized_observations'] = dict(zip(self.get_wrapper_attr(
                'observation_variables'), data_logger.normalized_observations[-1]))
            log_dict['Observations'] = dict(zip(self.get_wrapper_attr(
                'observation_variables'), data_logger.observations[-1]))
        else:
            log_dict['Observations'] = dict(zip(self.get_wrapper_attr(
                'observation_variables'), data_logger.observations[-1]))

        # ACTION
        # Original action sent
        log_dict['Agent_actions'] = dict(
            zip(self.get_wrapper_attr('action_variables'), data_logger.actions[-1]))
        # Action values performed in simulation
        log_dict['Simulation_actions'] = dict(zip(self.get_wrapper_attr(
            'action_variables'), data_logger.infos[-1]['action']))

        # REWARD
        log_dict['Reward'] = {'reward': data_logger.rewards[-1]}

        # INFO
        log_dict['Info'] = {
            key: float(value) for key,
            value in data_logger.infos[-1].items() if key not in self.excluded_info_keys}

        # CUSTOM METRICS
        if len(self.get_wrapper_attr('custom_variables')) > 0:
            custom_metrics = dict(zip(self.get_wrapper_attr(
                'custom_variables'), data_logger.custom_metrics[-1]))
            log_dict['Variables_custom'] = custom_metrics

        # Log in WandB
        self._log_data(log_dict)

    def wandb_log_summary(self) -> None:
        """Log episode summary in WandB platform.
        """
        if len(self.get_wrapper_attr('data_logger').rewards) > 0:
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
                for k, v in value.items():
                    self.wandb_run.log({f'{key}/{k}': v},
                                       step=self.global_timestep)
            else:
                self.wandb_run.log({key: value},
                                   step=self.global_timestep)


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
        try:
            assert all(
                var in self.get_wrapper_attr('observation_variables')
                for var in obs_reduction)
        except AssertionError as err:
            self.logger.error(
                'Some observation variable to be removed is not defined in the original observation space.')
            raise err

        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=self.env.get_wrapper_attr('observation_space').low[0],
            high=self.env.get_wrapper_attr('observation_space').high[0],
            shape=(
                self.env.get_wrapper_attr('observation_space').shape[0] -
                len(obs_reduction),
            ),
            dtype=self.env.get_wrapper_attr('observation_space').dtype)

        # Separate removed variables from observation variables
        self.observation_variables = list(
            filter(
                lambda x: x not in obs_reduction, deepcopy(
                    self.get_wrapper_attr('observation_variables'))))
        self.removed_observation_variables = obs_reduction

        self.logger.info('Wrapper initialized.')

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Separating removed variables from observation values and adding it to info dict.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Processig obs to delete removed variables and add them to info
        obs_dict = dict(
            zip(self.env.get_wrapper_attr('observation_variables'), obs))
        reduced_obs_dict = {
            key: obs_dict[key] for key in self.get_wrapper_attr('observation_variables')}

        return np.array(list(reduced_obs_dict.values())
                        ), reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Sends action to the environment. Separating removed variables from observation values and adding it to info dict"""
        obs, info = self.env.reset(seed=seed, options=options)

        # Processig obs to delete removed variables and add them to info
        obs_dict = dict(
            zip(self.env.get_wrapper_attr('observation_variables'), obs))
        reduced_obs_dict = {
            key: obs_dict[key] for key in self.get_wrapper_attr('observation_variables')}

        return np.array(list(reduced_obs_dict.values())), info

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
            env (Env): The environment to wrap.
            context_space (gym.spaces.Box): The space defining valid context variable values.
            delta_value (float): Maximum absolute change applied to context variables at each update.
            step_frequency_range (Tuple[int, int], optional): Range for the number of steps before each update. Defaults to (96, 96 * 7).
        """
        super().__init__(env)

        # Definition checks
        try:
            assert context_space.shape[0] == len(
                self.get_wrapper_attr('context_variables'))
        except AssertionError as err:
            self.logger.error(
                'Context space shape is not coherent with environment context variables.')
            raise err

        try:
            assert delta_value > 0
        except AssertionError as err:
            self.logger.error(
                'Delta temperature must be greater than 0.')
            raise err

        try:
            assert step_frequency_range[0] > 0 and step_frequency_range[0] < step_frequency_range[1]
        except AssertionError as err:
            self.logger.error(
                'Step frequency range invalid.')
            raise err

        # Initialization
        self.context_space = context_space
        self.delta_context = (-delta_value, delta_value)
        self.step_frequency_range = step_frequency_range

        if self.get_wrapper_attr('initial_context') is not None:
            self.current_context = np.array(
                self.get_wrapper_attr('initial_context'), np.float32)
        else:
            self.current_context = np.random.uniform(
                self.context_space.low,
                self.context_space.high,
                size=self.context_space.shape[0]).astype(np.float32)

        self.next_context_values, self.next_step_update = self._generate_context_values()

        self.logger.info('Wrapper initialized.')

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Complete...

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """

        # Discount frequency
        self.next_step_update -= 1
        if self.next_step_update == 0:
            # Update context
            self.get_wrapper_attr('update_context')(self.next_context_values)
            self.current_context = self.next_context_values
            self.logger.info('Context updated with values: {}'.format(
                self.next_context_values))
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
        assert env.get_wrapper_attr('building_path').split(
            '/')[-1] == 'OfficeGridStorageSmoothing.epJSON', 'OfficeGridStorageSmoothingActionConstraintsWrapper: This wrapper is not valid for this environment.'
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
