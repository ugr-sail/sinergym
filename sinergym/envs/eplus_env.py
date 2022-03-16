"""
Gym environment for simulation with EnergyPlus.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pkg_resources

from sinergym.simulators import EnergyPlus
from sinergym.utils.common import (parse_observation_action_space,
                                   parse_variables, setpoints_transform)
from sinergym.utils.rewards import ExpReward, LinearReward


class EplusEnv(gym.Env):
    """
    Environment with EnergyPlus simulator.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        idf_file,
        weather_file,
        variables_file,
        spaces_file,
        env_name='eplus-env-v1',
        discrete_actions=True,
        weather_variability=None,
        reward=LinearReward,
        reward_kwargs=dict(),
        config_params: dict = None
    ):
        """Environment with EnergyPlus simulator.

        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            variables_file (str): Variables defined in environment to be observation and action (see sinergym/data/variables/ for examples).
            spaces_file (str): Action and observation space defined in a xml (see sinergym/data/variables/ for examples).
            env_name (str, optional): Env name used for working directory generation. Defaults to 'eplus-env-v1'.
            discrete_actions (bool, optional): Whether the actions are discrete (True) or continuous (False). Defaults to True.
            weather_variability (tuple, optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Reward instance): Reward function instance used for agent feedback. Defaults to LinearReward.
            rewarg_kwargs (dict, optional): Parameters to be passed to the reward function.
            config_params (dict, optional): Parameters to configure the simulation.
        """
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = pkg_resources.resource_filename(
            'sinergym', 'data/')

        self.idf_path = os.path.join(self.pkg_data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(
            self.pkg_data_path, 'weather', weather_file)
        self.variables_path = os.path.join(
            self.pkg_data_path, 'variables', variables_file)
        self.spaces_path = os.path.join(
            self.pkg_data_path, 'variables', spaces_file)

        self.simulator = EnergyPlus(
            env_name=env_name,
            eplus_path=eplus_path,
            bcvtb_path=bcvtb_path,
            idf_path=self.idf_path,
            weather_path=self.weather_path,
            variable_path=self.variables_path,
            config_params=config_params
        )

        # parse variables (observation and action) from cfg file
        self.variables = parse_variables(self.variables_path)

        # Random noise to apply for weather series
        self.weather_variability = weather_variability

        # parse observation and action spaces from spaces_path
        space = parse_observation_action_space(self.spaces_path)
        observation_def = space['observation']
        discrete_action_def = space['discrete_action']
        continuous_action_def = space['continuous_action']

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=observation_def[0],
            high=observation_def[1],
            shape=observation_def[2],
            dtype=observation_def[3])

        # Action space
        self.flag_discrete = discrete_actions

        # Discrete
        if self.flag_discrete:
            self.action_mapping = discrete_action_def
            self.action_space = gym.spaces.Discrete(len(discrete_action_def))
        # Continuous
        else:
            # Defining action values setpoints (one per value)
            self.action_setpoints = []
            for i in range(len(self.variables['action'])):
                # action_variable --> [low,up]
                self.action_setpoints.append([
                    continuous_action_def[0][i], continuous_action_def[1][i]])

            self.action_space = gym.spaces.Box(
                # continuous_action_def[2] --> shape
                low=np.repeat(-1, continuous_action_def[2][0]),
                high=np.repeat(1, continuous_action_def[2][0]),
                dtype=continuous_action_def[3]
            )

        # Reward class
        self.reward_fn = reward(self, **reward_kwargs)
        self.obs_dict = None

    def step(self, action):
        """Sends action to the environment.

        Args:
            action (Union[int, float, np.integer, np.ndarray, List[Any], Tuple[Any]]): Action selected by the agent.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: Observation for next timestep, reward obtained, Whether the episode has ended or not and a dictionary with extra information
        """

        action_ = self._get_action(action)

        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        time_info, obs, done = self.simulator.step(action_)
        # Create dictionary with observation
        self.obs_dict = dict(zip(self.variables['observation'], obs))
        # Add current timestep information
        self.obs_dict['day'] = time_info[0]
        self.obs_dict['month'] = time_info[1]
        self.obs_dict['hour'] = time_info[2]

        # Calculate reward
        reward, terms = self.reward_fn()

        # Extra info
        info = {
            'timestep': int(
                time_info[3] / self.simulator._eplus_run_stepsize),
            'time_elapsed': int(time_info[3]),
            'day': self.obs_dict['day'],
            'month': self.obs_dict['month'],
            'hour': self.obs_dict['hour'],
            'total_power': terms['total_energy'],
            'total_power_no_units': terms['reward_energy'],
            'comfort_penalty': terms['reward_comfort'],
            'temperatures': terms['temperatures'],
            'out_temperature': self.obs_dict['Site Outdoor Air Drybulb Temperature (Environment)'],
            'action_': action_}

        return np.array(list(self.obs_dict.values()),
                        dtype=np.float32), reward, done, info

    def reset(self) -> np.ndarray:
        """Reset the environment.

        Returns:
            np.ndarray: Current observation.
        """
        # Change to next episode
        time_info, obs, _ = self.simulator.reset(self.weather_variability)
        self.obs_dict = dict(zip(self.variables['observation'], obs))

        self.obs_dict['day'] = time_info[0]
        self.obs_dict['month'] = time_info[1]
        self.obs_dict['hour'] = time_info[2]

        return np.array(list(self.obs_dict.values()), dtype=np.float32)

    def close(self) -> None:
        """End simulation."""

        self.simulator.end_env()

    def _get_action(self, action):
        """Transform the action for sending it to the simulator."""

        # Get action depending on flag_discrete
        if self.flag_discrete:
            # Index for action_mapping
            if np.issubdtype(type(action), np.integer):
                if isinstance(action, int):
                    setpoints = self.action_mapping[action]
                else:
                    setpoints = self.action_mapping[np.asscalar(action)]
            # Manual action
            elif isinstance(action, tuple) or isinstance(action, list):
                # stable-baselines DQN bug prevention
                if len(action) == 1:
                    setpoints = self.action_mapping[np.asscalar(action)]
                else:
                    setpoints = action
            elif isinstance(action, np.ndarray):
                setpoints = self.action_mapping[np.asscalar(action)]
            else:
                print("ERROR: ", type(action))
            action_ = list(setpoints)
        else:
            # transform action to setpoints simulation
            action_ = setpoints_transform(
                action, self.action_space, self.action_setpoints)

        return action_