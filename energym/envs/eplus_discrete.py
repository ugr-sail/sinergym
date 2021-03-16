"""Gym environment with discrete action space and raw observations."""

import gym
import os
import pkg_resources
import numpy as np
from opyplus import Epm

from ..utils import get_current_time_info, parse_variables
from ..simulators import EnergyPlus


class EplusDiscrete(gym.Env):
    """
    Discrete environment with EnergyPlus simulator.

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
    """


    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        idf_file,
        weather_file,
        comfort_range = (20, 22)
    ):
        """
        Class constructor

        Parameters
        ----------
        idf_file : str
            Name of the IDF file with the building definition.
        weather_file : str
            Name of the EPW file for weather conditions.
        comfort_range : tuple
            Temperature bounds (low, high) for calculating reward.
        """

        variables_file = 'variables.cfg'

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        data_path = pkg_resources.resource_filename('energym', 'data/')

        self.idf_path = os.path.join(data_path, 'buildings', idf_file)
        self.weather_path = os.path.join(data_path, 'weather', weather_file)
        self.variables_path = os.path.join(data_path, 'variables', variables_file)

        self.comfort_range = comfort_range

        self.simulator = EnergyPlus(
            env_name = 'eplus-discrete-v1',
            eplus_path = eplus_path,
            bcvtb_path = bcvtb_path,
            idf_path = self.idf_path,
            weather_path = self.weather_path,
            variable_path = self.variables_path
        )

        # Utils for getting time info and variable names
        self.epm = Epm.from_idf(self.idf_path)
        self.variables = parse_variables(self.variables_path)

        # Observation space
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=(19,), dtype=np.float32)
        
        # Action space
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

    def step(self, action):
        """
        Sends action to the environment.

        Parameters
        ----------
        action : int
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
        
        # Map action into setpoint
        setpoints = self.action_mapping[action]
        action_ = [setpoints[0], setpoints[1]]
        
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
        reward, energy_term, comfort_penalty = self._get_reward(temp, power)
        
        # Extra info
        info = {
            'timestep': t,
            'day' : obs_dict['day'],
            'month' : obs_dict['month'],
            'hour' : obs_dict['hour'],
            'total_power': power,
            'total_power_no_units': energy_term,
            'comfort_penalty': comfort_penalty
        }
        return np.array(obs_dict.values), reward, done, info

    def reset(self):
        t, obs, done = self.simulator.reset()
        return np.array(obs)

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.simulator.end_env()

    def _get_reward(self, temperature, power, beta = 1e-4):
        """
        Method for calculating the reward.

        reward = - beta * power - comfort_penalty

        The comfort penalty is just the difference between the current temperature
        and the bounds to the comfort range. If temperature between comfort_range,
        then comfort_penalty = 0.

        Parameters
        ----------
        temperature : float
            Current interior temperature
        power : float
            Current power consumption
        beta : float
            Parameter for normalizing and remove units from power

        Returns
        -------
        reward : float
            Total reward for this timestep
        power_no_units : float
            Power multiplied by beta
        comfort_penalty : float
            The comfort penalty
        """
        comfort_penalty = 0.0
        if temperature < self.comfort_range[0]:
            comfort_penalty -= self.comfort_range[0] - temperature
        if temperature < self.comfort_range[1]:
            comfort_penalty -= temperature - self.comfort_range[1]
        power_no_units = beta * power
        reward = - power_no_units - comfort_penalty
        return reward, power_no_units, comfort_penalty