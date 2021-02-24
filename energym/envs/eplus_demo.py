import gym
import os
import pkg_resources
import numpy as np
from ..simulators import EnergyPlus

class EplusDemo(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, comfort_range = (20, 22)):
        """
        Observation:
            Type: Box(16)
            Num    Observation               Min            Max
            0      var0                     -5e6            5e6
            1      var1                     -5e6            5e6
            ...
            15     var15                    -5e6            5e6
        
        Actions:
            Type: Box(2)
            Num    Action
            0      Heating setpoint
            1      Cooling setpoint
        """

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        data_path = pkg_resources.resource_filename('energym', 'data/')

        self.simulator = EnergyPlus(
            eplus_path = eplus_path,
            weather_path = os.path.join(data_path, 'weather/USA_Pittsburg_TMY3.epw'),
            bcvtb_path = bcvtb_path,
            variable_path = os.path.join(data_path, 'variables/variables.cfg'),
            idf_path = os.path.join(data_path, 'buildings/5ZoneAutoDXVAV.idf'),
            env_name = 'eplus-demo-v1'
        )

        self.comfort_range = comfort_range
        
        self.min_heat = 17
        self.max_heat = 22
        self.min_cool = 23
        self.max_cool = 28

        self.low = np.array([self.min_heat, self.min_cool], dtype=np.float32)
        self.high = np.array([self.max_heat, self.max_cool], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-5e6, high=5e6, shape=(16,), dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            self.low, self.high, dtype=np.float32
        )

    def step(self, action):
 
        a = [action[0], action[1]]
        
        # Send action to simulator
        self.simulator.logger_main.debug('[Action] Heat = ', str(action[0]), ' Cool = ', str(action[1]))
        t, obs, done = self.simulator.step(a)
        temp = obs[9]
        power = obs[-1]
        reward = self._get_reward(temp, power)
        
        return np.array(obs), reward, done, {}

    def reset(self):
        t, obs, done = self.simulator.reset()
        return np.array(obs)

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.simulator.end_env()

    def _get_reward(self, temperature, power, beta = 1e-4):
        """
        reward = -beta * power - comfort_penalty
        """
        
        comfort_penalty = 0.0
        if temperature < self.comfort_range[0]:
            comfort_penalty -= self.comfort_range[0] - temperature
        if temperature < self.comfort_range[1]:
            comfort_penalty -= temperature - self.comfort_range[1]
        reward = - beta * power - comfort_penalty
        return reward