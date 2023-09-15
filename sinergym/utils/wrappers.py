"""Implementation of custom Gym environments."""

import random
from collections import deque
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd

from sinergym.envs.eplus_env import EplusEnv
from sinergym.utils.common import is_wrapped
from sinergym.utils.constants import LOG_WRAPPERS_LEVEL, YEAR
from sinergym.utils.logger import CSVLogger, Logger


class MultiObjectiveReward(gym.Wrapper):

    logger = Logger().getLogger(name='WRAPPER MultiObjectiveReward',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(self, env: EplusEnv, reward_terms: List[str]):
        """The environment will return a reward vector of each objective instead of a scalar value.

        Args:
            env (EplusEnv): Original Sinergym environment.
            reward_terms (List[str]): List of keys in reward terms which will be included in reward vector.
        """
        super(MultiObjectiveReward, self).__init__(env)
        self.reward_terms = reward_terms

        self.logger.info('wrapper initialized.')

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
        reward_vector = [value for key, value in info.items(
        ) if key in self.get_wrapper_attr('reward_terms')]
        try:
            assert len(reward_vector) == len(
                self.get_wrapper_attr('reward_terms'))
        except AssertionError as err:
            self.logger.error('Some reward term is unknown')
            raise err
        return obs, reward_vector, terminated, truncated, info


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self,
                 env: EplusEnv,
                 epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
             env (Env): The environment to apply the wrapper
             epsilon (float): A stability parameter that is used when scaling the observations. Defaults to 1e-8
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.num_envs = 1
        self.is_vector_env = False

        self.unwrapped_observation = None
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()

        # Normalize observation and return
        return self.normalize(np.array([obs]))[
            0], reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        # Save original obs in class attribute
        self.unwrapped_observation = obs.copy()

        return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalizes the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / \
            np.sqrt(self.obs_rms.var + self.epsilon)


class MultiObsWrapper(gym.Wrapper):

    logger = Logger().getLogger(name='WRAPPER MultiObsWrapper',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(
            self,
            env: EplusEnv,
            n: int = 5,
            flatten: bool = True) -> None:
        """Stack of observations.

        Args:
            env (EplusEnv): Original Gym environment.
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
            low=-5e6, high=5e6, shape=new_shape, dtype=np.float32)

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


class LoggerWrapper(gym.Wrapper):

    logger = Logger().getLogger(name='WRAPPER LoggerWrapper',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(
        self,
        env: EplusEnv,
        logger_class: Callable = CSVLogger,
        monitor_header: Optional[List[str]] = None,
        progress_header: Optional[List[str]] = None,
        flag: bool = True,
    ):
        """CSVLogger to log interactions with environment.

        Args:
            env (EplusEnv): Original Gym environment in Sinergym.
            logger_class (CSVLogger): CSV Logger class to use to log all information.
            monitor_header: Header for monitor.csv in each episode. Default is None (default format).
            progress_header: Header for progress.csv in whole simulation. Default is None (default format).
            flag (bool, optional): State of logger (activate or deactivate). Defaults to True.
        """
        super(LoggerWrapper, self).__init__(env)
        # Headers for csv logger
        monitor_header_list = monitor_header if monitor_header is not None else ['timestep'] + self.get_wrapper_attr('observation_variables') + self.get_wrapper_attr(
            'action_variables') + ['time (hours)', 'reward', 'energy_penalty', 'comfort_penalty', 'absolute_temperature_violation', 'absolute_energy', 'terminated']
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
            'cumulative_energy_penalty',
            'mean_energy_penalty',
            'comfort_violation (%)',
            'cumulative_temperature_violation',
            'mean_temperature_violation',
            'std_temperature_violation',
            'length (timesteps)',
            'time_elapsed (hours)']
        self.progress_header = ''
        for element_header in progress_header_list:
            self.progress_header += element_header + ','
        self.progress_header = self.progress_header[:-1]

        # Create simulation logger, by default is active (flag=True)
        self.file_logger = logger_class(
            monitor_header=self.get_wrapper_attr('monitor_header'),
            progress_header=self.get_wrapper_attr('progress_header'),
            log_progress_file=self.get_wrapper_attr('workspace_path') +
            '/progress.csv',
            flag=flag)

        self.logger.info('Wrapper initialized.')

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
            self.file_logger.log_step_normalize(
                obs=obs,
                action=info['action'],
                terminated=terminated,
                info=info)
            # Record original observation too
            self.file_logger.log_step(
                obs=self.env.get_wrapper_attr('unwrapped_observation'),
                action=info['action'],
                terminated=terminated,
                info=info)
        else:
            # Only record observation without normalization
            self.file_logger.log_step(
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
        if self.get_wrapper_attr('is_running'):
            self.logger.info(
                'End of episode detected, recording summary (progress.csv) if logger is active')
            self.file_logger.log_episode(
                episode=self.env.get_wrapper_attr('episode'))

        # Then, reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Create monitor.csv for information of this episode
        self.logger.info(
            'Creating monitor.csv for current episode (episode ' + str(
                self.get_wrapper_attr('episode')) + ') if logger is active')
        self.file_logger.set_log_file(
            self.get_wrapper_attr('model').episode_path + '/monitor.csv')

        if is_wrapped(self, NormalizeObservation):
            # Store initial state of simulation (normalized)
            self.file_logger.log_step_normalize(obs=obs, action=[None for _ in range(len(
                self.env.get_wrapper_attr('action_variables')))], terminated=False, info=info)
            # And store original obs
            self.file_logger.log_step(obs=self.env.get_wrapper_attr('unwrapped_observation'),
                                      action=[None for _ in range(
                                          len(self.get_wrapper_attr('action_variables')))],
                                      terminated=False,
                                      info=info)
        else:
            # Only store original step
            self.file_logger.log_step(obs=obs,
                                      action=[None for _ in range(
                                          len(self.get_wrapper_attr('action_variables')))],
                                      terminated=False,
                                      info=info)

        return obs, info

    def close(self) -> None:
        """Recording last episode summary and close env.
        """
        # Record last episode summary before end simulation
        self.logger.info(
            'End of episode, recording summary (progress.csv) if logger is active')
        self.file_logger.log_episode(
            episode=self.env.get_wrapper_attr('episode'))

        # Then, close env
        self.env.close()

    def activate_logger(self) -> None:
        """Activate logger if its flag False.
        """
        self.file_logger.activate_flag()

    def deactivate_logger(self) -> None:
        """Deactivate logger if its flag True.
        """
        self.file_logger.deactivate_flag()


class DatetimeWrapper(gym.ObservationWrapper):
    """Wrapper to substitute day value by is_weekend flag, and hour and month by sin and cos values.
       Observation space is updated automatically."""

    logger = Logger().getLogger(name='WRAPPER DatetimeWrapper',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: EplusEnv):
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
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32)
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


class PreviousObservationWrapper(gym.ObservationWrapper):
    """Wrapper to add observation values from previous timestep to
    current environment observation"""

    logger = Logger().getLogger(name='WRAPPER PreviousObservationWrapper',
                                level=LOG_WRAPPERS_LEVEL)

    def __init__(self,
                 env: EplusEnv,
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
            low=-5e6, high=5e6, shape=(new_shape,), dtype=np.float32)

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


class DiscreteIncrementalWrapper(gym.ActionWrapper):
    """A wrapper for an incremental setpoint discrete action space environment.
    WARNING: A environment with only temperature setpoints control must be used
    with this wrapper."""

    logger = Logger().getLogger(name='WRAPPER DiscreteIncrementalWrapper',
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
        self.current_setpoints = initial_values

        # Check environment is valid
        try:
            assert not self.env.get_wrapper_attr('flag_discrete')
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

        # Reset default environment action_mapping and enable discrete
        # environment flag
        self.flag_discrete = True
        self.action_mapping = {}
        do_nothing = [0.0 for _ in range(
            len(self.env.get_wrapper_attr('action_variables')))]  # do nothing
        self.action_mapping[0] = do_nothing
        n = 1

        # Generate all posible actions
        for k in range(len(self.env.get_wrapper_attr('action_variables'))):
            for v in values:
                x = do_nothing.copy()
                x[k] = v
                self.action_mapping[n] = x
                n += 1

        self.action_space = gym.spaces.Discrete(n)
        self.logger.info('New incremental action mapping: {}'.format(n))
        self.logger.info('{}'.format(self.get_wrapper_attr('action_mapping')))
        self.logger.info('Wrapper initialized')

    def action(self, action):
        """Takes the discrete action and transforms it to setpoints tuple."""
        action_ = self.get_wrapper_attr('action_mapping')[action]
        # Update current setpoints values with incremental action
        self.current_setpoints = [
            sum(i) for i in zip(
                self.get_wrapper_attr('current_setpoints'),
                action_)]
        # clip setpoints returned
        self.current_setpoints = np.clip(
            np.array(self.get_wrapper_attr('current_setpoints')),
            self.env.get_wrapper_attr('real_space').low,
            self.env.get_wrapper_attr('real_space').high
        )

        # if normalization flag is active, this wrapper should normalize
        # before.
        if self.env.get_wrapper_attr('flag_normalization'):
            norm_values = self.env.get_wrapper_attr(
                'normalized_space').high - self.env.get_wrapper_attr('normalized_space').low
            setpoints_normalized = norm_values * ((self.get_wrapper_attr('current_setpoints') - self.env.get_wrapper_attr('real_space').low) / (
                self.env.get_wrapper_attr('real_space').high - self.env.get_wrapper_attr('real_space').low)) + self.env.get_wrapper_attr('normalized_space').low
            return list(setpoints_normalized)

        return list(self.current_setpoints)

    # ---------------------- Specific environment wrappers ---------------------#


class OfficeGridStorageSmoothingActionConstraintsWrapper(
        gym.ActionWrapper):  # pragma: no cover
    def __init__(self, env):
        assert env.building_path.split(
            '/')[-1] == 'OfficeGridStorageSmoothing.epJSON', 'OfficeGridStorageSmoothingActionConstraintsWrapper: This wrapper is not valid for this environment.'
        super().__init__(env)

    def action(self, act: np.ndarray) -> np.ndarray:
        """Due to Charge rate and Discharge rate can't be more than 0.0 simultaneously (in OfficeGridStorageSmoothing.epJSON),
           this wrapper clips one of the to 0.0 when both have a value upper than 0.0 (randomly).

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
            random_rate_index = random.randint(2, 3)
            act[random_rate_index] = null_value
        return act
