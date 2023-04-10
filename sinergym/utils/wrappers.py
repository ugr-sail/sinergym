"""Implementation of custom Gym environments."""

import random
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from sinergym.utils.common import is_wrapped
from sinergym.utils.logger import CSVLogger


class MultiObjectiveReward(gym.Wrapper):

    def __init__(self, env: Any, reward_terms: List[str]):
        """The environment will return a reward vector of each objective instead of a scalar value.

        Args:
            env (Any): Original Sinergym environment.
            reward_terms (List[str]): List of keys in reward terms which will be included in reward vector.
        """
        super(MultiObjectiveReward, self).__init__(env)
        self.reward_terms = reward_terms

    def step(self, action: Union[int, np.ndarray]) -> Tuple[
            np.ndarray, List[float], bool, bool, Dict[str, Any]]:
        """Perform the action and environment return reward vector.

        Args:
            action (Union[int, np.ndarray]): Action to be executed in environment.

        Returns:
            Tuple[ np.ndarray, List[float], bool, bool, Dict[str, Any]]: observation, vector reward, terminated, truncated and info.
        """
        # Execute normal reward
        obs, _, terminated, truncated, info = self.env.step(action)
        reward_vector = [
            value for key,
            value in info.items() if key in self.reward_terms]
        assert len(reward_vector) == len(
            self.reward_terms), 'Some reward term is unknown'
        return obs, reward_vector, terminated, truncated, info


class NormalizeObservation(gym.ObservationWrapper):

    def __init__(self,
                 env: Any,
                 ranges: Dict[str, Sequence[Any]]):
        """Observations normalized to range [0, 1].

        Args:
            env (Any): Original Sinergym environment.
            ranges (Dict[str, Sequence[Any]]): Observation variables ranges to apply normalization (rely on environment).
        """
        super(NormalizeObservation, self).__init__(env)
        self.unwrapped_observation = None
        self.ranges = ranges

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies normalization to observation.

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Normalized observation.
        """
        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()

        # NOTE: If you want to record day, month and hour, you should add that
        # variables as keys
        for i, variable in enumerate(self.env.variables['observation']):
            # normalization (handle DivisionbyZero Error)
            if (self.ranges[variable][1] -
                    self.ranges[variable][0] == 0):
                obs[i] = max(
                    self.ranges[variable][0], min(
                        obs[i], self.ranges[variable][1]))
            else:
                obs[i] = (obs[i] - self.ranges[variable][0]) / \
                    (self.ranges[variable][1] - self.ranges[variable][0])

            # If value is out
            if np.isnan(obs[i]):
                obs[i] = 0
            elif obs[i] > 1:
                obs[i] = 1
            elif obs[i] < 0:
                obs[i] = 0
        # Return obs values in the SAME ORDER than obs argument.
        return np.array(obs)

    def get_unwrapped_obs(self) -> Optional[np.ndarray]:
        """Get last environment observation without normalization.

        Returns:
            Optional[np.ndarray]: Last original observation. If it is the first observation, this value is None.
        """
        return self.unwrapped_observation


class MultiObsWrapper(gym.Wrapper):

    def __init__(self, env: Any, n: int = 5, flatten: bool = True) -> None:
        """Stack of observations.

        Args:
            env (Any): Original Gym environment.
            n (int, optional): Number of observations to be stacked. Defaults to 5.
            flatten (bool, optional): Whether or not flat the observation vector. Defaults to True.
        """
        super(MultiObsWrapper, self).__init__(env)
        self.n = n
        self.ind_flat = flatten
        self.history = deque([], maxlen=n)
        shape = env.observation_space.shape
        new_shape = (shape[0] * n,) if flatten else ((n,) + shape)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=new_shape, dtype=np.float32)

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
        if self.ind_flat:
            return np.array(self.history).reshape(-1,)
        else:
            return np.array(self.history)


class LoggerWrapper(gym.Wrapper):

    def __init__(
        self,
        env: Any,
        logger_class: Callable = CSVLogger,
        monitor_header: Optional[List[str]] = None,
        progress_header: Optional[List[str]] = None,
        flag: bool = True,
    ):
        """CSVLogger to log interactions with environment.

        Args:
            env (Any): Original Gym environment.
            logger_class (CSVLogger): CSV Logger class to use to log all information.
            monitor_header: Header for monitor.csv in each episode. Default is None (default format).
            progress_header: Header for progress.csv in whole simulation. Default is None (default format).
            flag (bool, optional): State of logger (activate or deactivate). Defaults to True.
        """
        gym.Wrapper.__init__(self, env)
        # Headers for csv logger
        monitor_header_list = monitor_header if monitor_header is not None else ['timestep'] + env.variables['observation'] + env.variables['action'] + [
            'time (seconds)', 'reward', 'power_penalty', 'comfort_penalty', 'abs_comfort', 'terminated']
        self.monitor_header = ''
        for element_header in monitor_header_list:
            self.monitor_header += element_header + ','
        self.monitor_header = self.monitor_header[:-1]
        progress_header_list = progress_header if progress_header is not None else [
            'episode_num',
            'cumulative_reward',
            'mean_reward',
            'cumulative_power_consumption',
            'mean_power_consumption',
            'cumulative_comfort_penalty',
            'mean_comfort_penalty',
            'cumulative_power_penalty',
            'mean_power_penalty',
            'comfort_violation (%)',
            'mean_comfort_violation',
            'std_comfort_violation',
            'cumulative_comfort_violation',
            'length(timesteps)',
            'time_elapsed(seconds)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.logger = logger_class(
            monitor_header=self.monitor_header,
            progress_header=self.progress_header,
            log_progress_file=env.simulator._env_working_dir_parent +
            '/progress.csv',
            flag=flag)

    def step(self, action: Union[int, np.ndarray]
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Sends action to the environment. Logging new information in monitor.csv.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not, Whether episode has been truncated or not, and a dictionary with extra information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # We added some extra values (month,day,hour) manually in env, so we
        # need to delete them.
        if is_wrapped(self, NormalizeObservation):
            # Record action and new observation in simulator's csv
            self.logger.log_step_normalize(
                obs=obs,
                action=info['action'],
                terminated=terminated,
                info=info)
            # Record original observation too
            self.logger.log_step(
                obs=self.env.get_unwrapped_obs(),
                action=info['action'],
                terminated=terminated,
                info=info)
        else:
            # Only record observation without normalization
            self.logger.log_step(
                obs=obs,
                action=info['action'],
                terminated=terminated,
                info=info)

        return obs, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict[str,
                                     Any]] = None) -> Tuple[np.ndarray,
                                                            Dict[str,
                                                                 Any]]:
        """Reset the environment. Recording episode summary in logger

        Args:
            seed (Optional[int]): The seed that is used to initialize the environment's episode (np_random). if value is None, a seed will be chosen from some source of entropy. Defaults to None.
            options (Optional[Dict[str, Any]]):Additional information to specify how the environment is reset. Defaults to None.

        Returns:
            Tuple[np.ndarray,Dict[str,Any]]: Current observation and info context with additional information.
        """
        # It isn't first episode simulation, so we can logger last episode
        if self.env.simulator._episode_existed:
            self.env.simulator.logger_main.debug(
                'End of episode, recording summary (progress.csv) if logger is active')
            self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Create monitor.csv for information of this episode
        self.env.simulator.logger_main.debug(
            'Creating monitor.csv for current episode (episode ' + str(
                self.env.simulator._epi_num) + ') if logger is active')
        self.logger.set_log_file(
            self.env.simulator._eplus_working_dir + '/monitor.csv')

        if is_wrapped(self, NormalizeObservation):
            # Store initial state of simulation (normalized)
            self.logger.log_step_normalize(obs=obs, action=[None for _ in range(
                len(self.env.variables['action']))], terminated=False, info=info)
            # And store original obs
            self.logger.log_step(obs=self.env.get_unwrapped_obs(),
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 terminated=False,
                                 info=info)
        else:
            # Only store original step
            self.logger.log_step(obs=obs,
                                 action=[None for _ in range(
                                     len(self.env.variables['action']))],
                                 terminated=False,
                                 info=info)

        return obs, info

    def close(self) -> None:
        """Recording last episode summary and close env.
        """
        # Record last episode summary before end simulation
        self.env.simulator.logger_main.debug(
            'End of episode, recording summary (progress.csv) if logger is active')
        self.logger.log_episode(episode=self.env.simulator._epi_num)

        # Then, close env
        self.env.close()

    def activate_logger(self) -> None:
        """Activate logger if its flag False.
        """
        self.logger.activate_flag()

    def deactivate_logger(self) -> None:
        """Deactivate logger if its flag True.
        """
        self.logger.deactivate_flag()


class DatetimeWrapper(gym.ObservationWrapper):
    """Wrapper to substitute day value by is_weekend flag, and hour and month by sin and cos values.
       Observation space is updated automatically."""

    def __init__(self,
                 env: Any):
        super(DatetimeWrapper, self).__init__(env)
        # Update new shape
        new_shape = env.observation_space.shape[0] + 2
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32)
        # Update observation variables
        day_index = self.variables['observation'].index('day')
        self.variables['observation'][day_index] = 'is_weekend'
        hour_index = self.variables['observation'].index('hour')
        self.variables['observation'][hour_index] = 'hour_cos'
        self.variables['observation'].insert(hour_index + 1, 'hour_sin')
        month_index = self.variables['observation'].index('month')
        self.variables['observation'][month_index] = 'month_cos'
        self.variables['observation'].insert(month_index + 1, 'month_sin')

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies calculation in is_weekend flag, and sen and cos in hour and month

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: Transformed observation.
        """
        obs_dict = dict(zip(self.original_obs, obs))
        # New obs dict with same values than obs_dict but with new fields with
        # None
        new_obs = dict.fromkeys(self.variables['observation'])
        for key, value in obs_dict.items():
            if key in new_obs.keys():
                new_obs[key] = value
        dt = datetime(
            int(obs_dict['year']),
            int(obs_dict['month']),
            int(obs_dict['day']),
            int(obs_dict['hour']))
        # Update obs
        new_obs['is_weekend'] = 1.0 if dt.isoweekday() in [6, 7] else 0.0
        new_obs['hour_cos'] = np.cos(obs_dict['hour'])
        new_obs['hour_sin'] = np.sin(obs_dict['hour'])
        new_obs['month_cos'] = np.cos(obs_dict['month'])
        new_obs['month_sin'] = np.sin(obs_dict['month'])

        return np.array(list(new_obs.values()))


class PreviousObservationWrapper(gym.ObservationWrapper):
    """Wrapper to add observation values from previous timestep to
    current environment observation"""

    def __init__(self,
                 env: Any,
                 previous_variables: List[str]):
        super(PreviousObservationWrapper, self).__init__(env)
        # Check and apply previous variables to observation space and variables
        # names
        self.original_variable_index = []
        for obs_var in previous_variables:
            assert obs_var in self.variables['observation'], '{} variable is not defined in observation space, revise the name.'.format(
                obs_var)
            self.original_variable_index.append(
                self.variables['observation'].index(obs_var))
            self.variables['observation'].append(obs_var + '_previous')
        # Update new shape
        new_shape = env.observation_space.shape[0] + len(previous_variables)
        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32)

        # previous observation initialization
        self.previous_observation = np.zeros(
            shape=len(previous_variables), dtype=np.float32)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Add previous observation to the current one

        Args:
            obs (np.ndarray): Original observation.

        Returns:
            np.ndarray: observation with
        """

        new_obs = np.concatenate((obs, self.previous_observation))
        # Aquí tengo que seleccionar las variables que se correponden con lo
        # que son
        self.previous_observation = obs[self.original_variable_index]

        return new_obs


class DiscreteIncrementalWrapper(gym.ActionWrapper):
    """A wrapper for an incremental setpoint discrete action space environment.
    WARNING: A environment with only temperature setpoints control must be used
    with this wrapper."""

    def __init__(
        self,
        env: gym.Env,
        max_values: List[float],
        min_values: List[float],
        delta_temp: float = 2.0,
        step_temp: float = 0.5,
    ):
        """
        Args:
            env: The original Sinergym env.
            action_names: Name of the action variables with the setpoint control you want to do incremental.
            initial_values: Initial values of the setpoints. One list per zone: [[heating zone 1, cooling zone 1], [heating zone 2, cooling zone 2], ...]
            heating_range: Acceptable values for the heating setpoint.
            cooling_range: Acceptable values for the cooling setpoint.
            delta_temp: Maximum temperature variation in the setpoints in one step.
            step_temp: Minimum temperature variation in the setpoints in one step.
        """

        super().__init__(env)

        # Params
        self.env = env
        self.current_setpoints = []
        self.max_values = max_values
        self.min_values = min_values

        # calculate initial values for setpoints
        for external_schedule in self.env.simulator._config.building.ExternalInterface_Schedule:
            self.current_setpoints.append(external_schedule.initial_value)

        # Check environment is valid
        assert len(
            self.current_setpoints) == len(
            self.env.variables['action']), 'IncrementalWrapper: Number of variables is different from environment'
        assert len(
            self.current_setpoints) == len(
            self.max_values), 'IncrementalWrapper: max_values specified is incorrect for the number of action variables'
        assert len(
            self.current_setpoints) == len(
            self.min_values), 'IncrementalWrapper: min_values specified is incorrect for the number of action variables'
        assert self.env.flag_discrete, 'IncrementalWrapper: Environment wrapped must be discrete'

        # Define all posible setpoint variations
        values = np.arange(step_temp, delta_temp + step_temp / 10, step_temp)
        values = [v for v in [*values, *-values]]

        # Reset default environment action_mapping
        self.action_mapping = {}
        do_nothing = [0.0 for _ in range(
            len(self.env.variables['action']))]  # do nothing
        self.action_mapping[0] = do_nothing
        n = 1

        # Generate all posible actions
        for k in range(len(self.env.variables['action'])):
            for v in values:
                x = do_nothing.copy()
                x[k] = v
                self.action_mapping[n] = x
                n += 1

        self.action_space = gym.spaces.Discrete(n)
        print(f'New incremental action mapping: {n}')
        print(self.action_mapping)

    def action(self, action):
        """Takes the discrete action and transforms it to setpoints tuple."""
        action_ = self.action_mapping[action]
        # Update current setpoints values with incremental action
        self.current_setpoints = [
            sum(i) for i in zip(
                self.current_setpoints,
                action_)]

        setpoints = np.clip(
            np.array(self.current_setpoints),
            self.min_values,
            self.max_values
        )
        return list(setpoints)

    # ---------------------- Specific environment wrappers ---------------------#


class OfficeGridStorageSmoothingActionConstraintsWrapper(
        gym.ActionWrapper):  # pragma: no cover
    def __init__(self, env):
        assert env.idf_path.split(
            '/')[-1] == 'OfficeGridStorageSmoothing.idf', 'OfficeGridStorageSmoothingActionConstraintsWrapper: This wrapper is not valid for this environment.'
        super().__init__(env)

    def action(self, act: np.ndarray) -> np.ndarray:
        """Due to Charge rate and Discharge rate can't be more than 0.0 simultaneously (in OfficeGridStorageSmoothing.idf),
           this wrapper clips one of the to 0.0 when both have a value upper than 0.0 (randomly).

        Args:
            act (np.ndarray): Action to be clipped

        Returns:
            np.ndarray: Action Clipped
        """
        if self.flag_discrete:
            null_value = 0.0
        else:
            # -1.0 is 0.0 when action space transformation to simulator action space.
            null_value = -1.0
        if act[2] > null_value and act[3] > null_value:
            random_rate_index = random.randint(2, 3)
            act[random_rate_index] = null_value
        return act
