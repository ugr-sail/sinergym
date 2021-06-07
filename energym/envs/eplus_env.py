"""Gym environment for simulation with EnergyPlus.

Funcionalities:
    - Both discrete and continuous action spaces
    - Add variability into the weather series
    - Reward is computed with absolute difference to comfort range
    - Raw observations, defined in the variables.cfg file
"""


import gym
import os
import pkg_resources
import numpy as np

from opyplus import Epm, WeatherData

from ..utils.common import get_current_time_info, parse_variables, create_variable_weather, parse_observation_action_space, CSVLogger
from ..simulators import EnergyPlus
from ..utils.rewards import SimpleReward


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
        weather_variability=None
    ):
        """Environment with EnergyPlus simulator.


        Args:
            idf_file (str): Name of the IDF file with the building definition.
            weather_file (str): Name of the EPW file for weather conditions.
            discrete_actions (bool, optional): Whether the actions are discrete (True) or continuous (False). Defaults to True.
            weather_variability (tuple, optional): Tuple with the mean and standard desviation of the Gaussian noise to be applied to weather data. Defaults to None.
        """

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = pkg_resources.resource_filename(
            'energym', 'data/')

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
            variable_path=self.variables_path
        )

        # Utils for getting time info, weather and variable names
        self.epm = Epm.from_idf(self.idf_path)
        self.variables = parse_variables(self.variables_path)
        self.weather_data = WeatherData.from_epw(self.weather_path)

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
        if self.flag_discrete:
            self.action_mapping = discrete_action_def
            self.action_space = gym.spaces.Discrete(len(discrete_action_def))
        else:
            self.action_space = gym.spaces.Box(
                low=np.array(continuous_action_def[0]),
                high=np.array(continuous_action_def[1]),
                dtype=continuous_action_def[3]
            )

        # Reward class
        self.cls_reward = SimpleReward()
        # Episode rewards
        self.ep_rewards = []

        # Headers for csv loggers
        monitor_header_list = ['timestep,month,day,hour']+self.variables['observation'] + \
            self.variables['action']+['time (seconds)', 'reward',
                                      'power_penalty', 'comfort_penalty', 'done']
        self.monitor_header = ''
        for element_header in monitor_header_list:
            self.monitor_header += element_header+','
        self.monitor_header = self.monitor_header[:-1]
        self.progress_header = 'episode_num,cumulative_reward,mean_reward,cumulative_power_consumption,mean_power_consumption,cumulative_comfort_penalty,mean_comfort_penalty,cumulative_power_penalty,mean_power_penalty,comfort_violation (%),length(timesteps),time_elapsed(seconds)'

        # Create simulation logger, by default is active (flag=True)
        self.logger = CSVLogger(monitor_header=self.monitor_header, progress_header=self.progress_header,
                                log_progress_file=self.simulator._env_working_dir_parent+'/progress.csv', flag=True)

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
                if type(action) == int:
                    setpoints = self.action_mapping[action]
                else:
                    setpoints = self.action_mapping[np.asscalar(action)]
            # Manual action
            elif type(action) == tuple or type(action) == list:
                # stable-baselines DQN bug prevention
                if len(action) == 1:
                    setpoints = self.action_mapping[np.asscalar(action)]
                else:
                    setpoints = action
            else:
                print("ERROR: ", type(action))
            action_ = list(setpoints)
        else:
            action_ = list(action)

        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        t, obs, done = self.simulator.step(action_)
        # Create dictionary with observation
        obs_dict = dict(zip(self.variables['observation'], obs))
        # Add current timestep information
        time_info = get_current_time_info(self.epm, t)
        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        # Calculate reward

        # Calculate temperature mean for all building zones
        temp_values = [value for key, value in obs_dict.items(
        ) if key.startswith('Zone Air Temperature')]

        power = obs_dict['Facility Total HVAC Electric Demand Power (Whole Building)']
        reward, terms = self.cls_reward.calculate(
            power, temp_values, time_info[1], time_info[0])

        # Add reward to episode list
        self.ep_rewards.append(reward)

        # Extra info
        info = {
            'timestep': int(t/self.simulator._eplus_run_stepsize),
            'time_elapsed': int(t),
            'day': obs_dict['day'],
            'month': obs_dict['month'],
            'hour': obs_dict['hour'],
            'total_power': power,
            'total_power_no_units': terms['reward_energy'],
            'comfort_penalty': terms['reward_comfort'],
            'temperatures': temp_values,
            'out_temperature': obs_dict['Site Outdoor Air Drybulb Temperature (Environment)']
        }

        # Record action and new observation in simulator's csv
        self.logger.log_step(timestep=info['timestep'],
                             date=[info['month'],
                                   info['day'], info['hour']],
                             observation=obs,
                             action=action_,
                             simulation_time=info['time_elapsed'],
                             reward=reward,
                             total_power_no_units=info['total_power_no_units'],
                             comfort_penalty=info['comfort_penalty'],
                             power=info['total_power'],
                             done=done)

        return np.array(list(obs_dict.values())), reward, done, info

    def reset(self):
        """Reset the environment.

        Returns:
            np.array: Current observation.
        """
        # Create new random weather file
        new_weather = create_variable_weather(
            self.weather_data, self.weather_path, variation=self.weather_variability)

        # It isn't first episode simulation, so we can logger last episode
        if self.simulator._episode_existed:
            self.simulator.logger_main.debug(
                'End of episode, recording summary (progress.csv) if logger is active')
            self.logger.log_episode(episode=self.simulator._epi_num)

        # Change to next episode
        t, obs, done = self.simulator.reset(new_weather)

        # Reset episode rewards
        self.ep_rewards = []

        obs_dict = dict(zip(self.variables['observation'], obs))

        time_info = get_current_time_info(self.epm, t)
        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        # Create monitor.csv for information of this episode
        self.simulator.logger_main.debug(
            'Creating monitor.csv for current episode (episode '+str(self.simulator._epi_num)+') if logger is active')
        self.logger.set_log_file(
            self.simulator._eplus_working_dir+'/monitor.csv')
        # Store initial state of simulation
        self.logger.log_step(timestep=0,
                             date=[obs_dict['month'],
                                   obs_dict['day'], obs_dict['hour']],
                             observation=obs,
                             action=[None for _ in range(
                                 len(self.variables['action']))],
                             simulation_time=0,
                             reward=None,
                             total_power_no_units=None,
                             comfort_penalty=None,
                             power=None,
                             done=done)

        return np.array(list(obs_dict.values()))

    def render(self, mode='human'):
        """Environment rendering."""
        pass

    def close(self):
        """End simulation."""
        # Record last episode summary before end simulation
        self.simulator.logger_main.debug(
            'End of episode, recording summary (progress.csv) if logger is active')
        self.logger.log_episode(episode=self.simulator._epi_num)

        self.simulator.end_env()

    ## EXTRA GYM METHODS ##

    def activate_logger(self):
        self.logger.activate_flag()

    def deactivate_logger(self):
        self.logger.deactivate_flag()
