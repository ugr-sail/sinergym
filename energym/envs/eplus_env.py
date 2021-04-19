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

from ..utils.common import get_current_time_info, parse_variables, create_variable_weather
from ..simulators import EnergyPlus
from ..utils.rewards import SimpleReward


class EplusEnv(gym.Env):
    """
    Environment with EnergyPlus simulator.

    Observation:
        Type: Box(16)
        Num    Variable name                                             Min            Max
        0       Site Outdoor Air Drybulb Temperature                     -5e6            5e6
        1       Site Outdoor Air Relative Humidity                       -5e6            5e6
        2       Site Wind Speed                                          -5e6            5e6
        3       Site Wind Direction                                      -5e6            5e6
        4       Site Diffuse Solar Radiation Rate per Area               -5e6            5e6
        5       Site Direct Solar Radiation Rate per Area                -5e6            5e6
        6       Zone Thermostat Heating Setpoint Temperature             -5e6            5e6
        7       Zone Thermostat Cooling Setpoint Temperature             -5e6            5e6
        8       Zone Air Temperature                                     -5e6            5e6
        9       Zone Thermal Comfort Mean Radiant Temperature            -5e6            5e6
        10      Zone Air Relative Humidity                               -5e6            5e6
        11      Zone Thermal Comfort Clothing Value                      -5e6            5e6
        12      Zone Thermal Comfort Fanger Model PPD                    -5e6            5e6
        13      Zone People Occupant Count                               -5e6            5e6
        14      People Air Temperature                                   -5e6            5e6
        15      Facility Total HVAC Electric Demand Power                -5e6            5e6
        16      Current day                                               1              31
        17      Current month                                             1              12
        18      Current hour                                              0              23

        ...
    
    Actions:
        Type: Discrete(10)
        Num    Action
        0       Heating setpoint = 15, Cooling setpoint = 30
        1       Heating setpoint = 16, Cooling setpoint = 29
        2       Heating setpoint = 17, Cooling setpoint = 28
        3       Heating setpoint = 18, Cooling setpoint = 27
        4       Heating setpoint = 19, Cooling setpoint = 26
        5       Heating setpoint = 20, Cooling setpoint = 25
        6       Heating setpoint = 21, Cooling setpoint = 24
        7       Heating setpoint = 22, Cooling setpoint = 23
        8       Heating setpoint = 22, Cooling setpoint = 22
        9       Heating setpoint = 21, Cooling setpoint = 21

        Type: Box(2)
        Num    Variable name                         Min            Max
        0      Heating setpoint                     15.0           22.5
        1      Cooling setpoint                     22.5           30.0
    """

    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        idf_file,
        weather_file,
        env_name='eplus-env-v1',
        discrete_actions = True,
        weather_variability = None
    ):
        """
        Class constructor.

        Parameters
        ----------
        idf_file : str
            Name of the IDF file with the building definition.
        weather_file : str
            Name of the EPW file for weather conditions.
        discrete_actions : bool
            Whether the actions are discrete (True) or continuous (False).
        """

        variables_file = 'variables.cfg'

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.pkg_data_path = pkg_resources.resource_filename('energym', 'data/')

        self.idf_path = os.path.join(self.pkg_data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(self.pkg_data_path, 'weather', weather_file)
        self.variables_path = os.path.join(self.pkg_data_path, 'variables', variables_file)

        self.simulator = EnergyPlus(
            env_name = env_name,
            eplus_path = eplus_path,
            bcvtb_path = bcvtb_path,
            idf_path = self.idf_path,
            weather_path = self.weather_path,
            variable_path = self.variables_path
        )

        # Utils for getting time info, weather and variable names
        self.epm = Epm.from_idf(self.idf_path)
        self.variables = parse_variables(self.variables_path)
        self.weather_data = WeatherData.from_epw(self.weather_path)

        # Random noise to apply for weather series
        self.weather_variability = weather_variability

        # Observation space
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=(19,), dtype=np.float32)
        
        # Action space
        self.flag_discrete = discrete_actions
        if self.flag_discrete:
            self.action_mapping = {
                0: (15, 30), 
                1: (16, 29), 
                2: (17, 28), 
                3: (18, 27), 
                4: (19, 26), 
                5: (20, 25), 
                6: (21, 24), 
                7: (22, 23), 
                8: (22, 22),
                9: (21, 21)
            }
            self.action_space = gym.spaces.Discrete(10)
        else:
            self.action_space = gym.spaces.Box(
                low = np.array([15.0, 22.5]), 
                high = np.array([22.5, 30.0]), 
                shape = (2,), dtype = np.float32
            )

        # Reward class
        self.cls_reward = SimpleReward()

    def step(self, action):
        """
        Sends action to the environment.

        Parameters
        ----------
        action : int or np.array
            Action selected by the agent

        Returns
        -------
        np.array
            Observation for next timestep
        reward : float
            Reward obtained
        done : bool
            Whether the episode has ended or not
        info : dict
            A dictionary with extra information
        """
        
        # Get action depending on flag_discrete
        if self.flag_discrete:
            setpoints = self.action_mapping[action]
            action_ = [setpoints[0], setpoints[1]]
        else:
            action_ = list(action)
        
        # Send action to the simulator
        self.simulator.logger_main.debug(action_)
        t, obs, done = self.simulator.step(action_)
        # Create dictionary with observation
        obs_dict = dict(zip(self.variables, obs))
        # Add current timestep information
        time_info = get_current_time_info(self.epm, t)
        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        # Calculate reward
        temp = obs_dict['Zone Air Temperature']
        power = obs_dict['Facility Total HVAC Electric Demand Power']
        reward, terms = self.cls_reward.calculate(power, temp, time_info[1], time_info[0])
        
        # Extra info
        info = {
            'timestep': t,
            'day' : obs_dict['day'],
            'month' : obs_dict['month'],
            'hour' : obs_dict['hour'],
            'total_power': power,
            'total_power_no_units': terms['reward_energy'],
            'comfort_penalty': terms['reward_comfort'],
            'temperature': temp,
            'out_temperature': obs_dict['Site Outdoor Air Drybulb Temperature']
        }
        return np.array(list(obs_dict.values())), reward, done, info

    def reset(self):
        """"""
        # Create new random weather file
        new_weather = create_variable_weather(self.weather_data, self.weather_path, variation = self.weather_variability)
        
        t, obs, done = self.simulator.reset(new_weather)
        
        obs_dict = dict(zip(self.variables, obs))
        
        time_info = get_current_time_info(self.epm, t)
        obs_dict['day'] = time_info[0]
        obs_dict['month'] = time_info[1]
        obs_dict['hour'] = time_info[2]

        return np.array(list(obs_dict.values()))

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.simulator.end_env()
