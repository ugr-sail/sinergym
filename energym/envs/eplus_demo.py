import gym
import os
import pkg_resources
import numpy as np
from ..simulators import EnergyPlus

class EplusDemo(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, comfort_range = (20, 22)):
        """
        """

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.comfort_range = comfort_range
        data_path = pkg_resources.resource_filename('energym', 'data/')

        self.simulator = EnergyPlus(
            eplus_path = eplus_path,
            weather_path = os.path.join(data_path, 'weather/USA_Pittsburg_TMY3.epw'),
            bcvtb_path = bcvtb_path,
            variable_path = os.path.join(data_path, 'variables/variables.cfg'),
            idf_path = os.path.join(data_path, 'buildings/5ZoneAutoDXVAV.idf'),
            env_name = 'eplus-demo-v1'
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
        """"""
        
        t1 = self.action_mapping[action]
        a = [t1, 25.0]
        # Send action to de simulator
        self.simulator.logger_main.debug(a)
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
        """"""
        comfort_penalty = 0.0
        if temperature < self.comfort_range[0]:
            comfort_penalty -= self.comfort_range[0] - temperature
        if temperature < self.comfort_range[1]:
            comfort_penalty -= temperature - self.comfort_range[1]
        reward = - beta * power - comfort_penalty
        return reward
