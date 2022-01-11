"""Gym environment for simulation with EnergyPlus.

Funcionalities:
    - Both discrete and continuous action spaces
    - Add variability into the weather series
    - Reward is computed with absolute difference to comfort range
    - Raw observations, defined in the variables.cfg file
"""

import gym
import os
import opyplus
import pkg_resources
import numpy as np

from sinergym.utils.config import Config
from sinergym.utils.common import get_current_time_info, parse_variables, create_variable_weather, parse_observation_action_space, setpoints_transform
from sinergym.simulators import EnergyPlus
from sinergym.utils.rewards import ExpReward, LinearReward
from pprint import pprint


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
        reward=LinearReward(),
        config_params: dict = None
    ):
        """Environment with EnergyPlus simulator.


        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            variables_file (str): Variables defined in environment to be observation and action (see sinergym/data/variables/ for examples).
            spaces_file (str): Action and observation space defined in a xml (see sinergym/data/variables/ for examples).
            env_name: Env name used for working directory generation.
            discrete_actions (bool, optional): Whether the actions are discrete (True) or continuous (False). Defaults to True.
            weather_variability (tuple, optional): Tuple with sigma, mu and tao of the Ornstein-Uhlenbeck process to be applied to weather data. Defaults to None.
            reward (Reward instance): Reward function instance used for agent feedback. Defaults to LinearReward.
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
        self.cls_reward = reward

    def step(self, action):
        """Sends action to the environment.

        Args:
            action (int or np.array): Action selected by the agent.

        Returns:
            np.array: Observation for next timestep.
            float: Reward obtained.
            bool: Whether the episode has ended or not.
            dict: A dictionary with extra information.
        """

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

        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        time_info, obs, done = self.simulator.step(action_)
        # Create dictionary with observation
        obs_dict = dict(zip(self.variables['observation'], obs))
        # Add current timestep information
        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        # Calculate reward

        # Calculate temperature mean for all building zones
        temp_values = [value for key, value in obs_dict.items(
        ) if key.startswith('Zone Air Temperature')]

        power = obs_dict['Facility Total HVAC Electricity Demand Rate (Whole Building)']
        reward, terms = self.cls_reward.calculate(
            power, temp_values, time_info[1], time_info[0])

        # Extra info
        info = {
            'timestep': int(
                time_info[3] / self.simulator._eplus_run_stepsize),
            'time_elapsed': int(time_info[3]),
            'day': obs_dict['day'],
            'month': obs_dict['month'],
            'hour': obs_dict['hour'],
            'total_power': power,
            'total_power_no_units': terms['reward_energy'],
            'comfort_penalty': terms['reward_comfort'],
            'temperatures': temp_values,
            'out_temperature': obs_dict['Site Outdoor Air Drybulb Temperature (Environment)'],
            'action_': action_}

        return np.array(list(obs_dict.values()),
                        dtype=np.float32), reward, done, info

    def reset(self):
        """Reset the environment.

        Returns:
            np.array: Current observation.
        """
        # Change to next episode
        time_info, obs, done = self.simulator.reset(self.weather_variability)
        obs_dict = dict(zip(self.variables['observation'], obs))

        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        return np.array(list(obs_dict.values()), dtype=np.float32)

    def render(self, mode='human'):
        """Environment rendering."""
        pass

    def close(self):
        """End simulation."""

        self.simulator.end_env()
