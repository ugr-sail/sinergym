"""Gym environment with discrete action space and raw observations."""

import gym
import os
import pkg_resources
import numpy as np
from ..simulators import EnergyPlus

class EplusDiscrete(gym.Env):
    """Discrete environment with EnergyPlus simulator."""
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        idf_file,
        weather_file,
        comfort_range = (20, 22)
    ):
        """Class constructor

        Parameters
        ----------
        idf_file : str
            Name of the IDF file with the building definition.
        weather_file : str
            Name of the EPW file for weather conditions.
        comfort_range : tuple
            Temperature bounds (low, high) for calculating reward.
        """

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.comfort_range = comfort_range
        data_path = pkg_resources.resource_filename('energym', 'data/')

        self.simulator = EnergyPlus(
            eplus_path = eplus_path,
            weather_path = os.path.join(data_path, 'weather', weather_file),
            bcvtb_path = bcvtb_path,
            variable_path = os.path.join(data_path, 'variables/variables.cfg'),
            idf_path = os.path.join(data_path, 'buildings', idf_file),
            env_name = 'eplus-discrete-v1'
        )
        # Observation space
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=(16,), dtype=np.float32)
        # Action space
        # 9 possible actions - [Heating SetPoint, Contant Cooling Setpoint = 25]
        self.action_mapping = {
            0: 15, 1: 16, 2: 17, 3: 18, 4: 19, 5: 20, 6: 21, 7: 22, 8: 23, 9: 24
        }
        self.action_space = gym.spaces.Discrete(10)

    def step(self, action):
        """Sends action to the environment.

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
        t1 = self.action_mapping[action]
        a = [t1, 25.0]
        # Send action to de simulator
        self.simulator.logger_main.debug(a)
        t, obs, done = self.simulator.step(a)
        # Calculate reward
        temp = obs[8]
        power = obs[-1]
        reward, energy_term, comfort_penalty = self._get_reward(temp, power)
        # Extra info
        info = {
            'timestep': t,
            'total_power': power,
            'total_power_no_units': energy_term,
            'comfort_penalty': comfort_penalty
        }
        return np.array(obs), reward, done, info

    def reset(self):
        t, obs, done = self.simulator.reset()
        return np.array(obs)

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.simulator.end_env()

    def _get_reward(self, temperature, power, beta = 1e-4):
        """Method for calculating the reward.

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