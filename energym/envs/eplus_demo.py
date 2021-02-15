import gym
import os
import numpy as np
from ..simulators import EnergyPlus

class EplusDemo(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, action_type = 'discrete', comfort_range = (20, 22)):
        """
        """

        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']
        self.comfort_range = comfort_range

        self.simulator = EnergyPlus(
            eplus_path = eplus_path,
            weather_path = '/home/jjimenez/energym/energym/data/weather/USA_Pittsburg_TMY3.epw',
            bcvtb_path = bcvtb_path,
            variable_path = '/home/jjimenez/energym/energym/data/variables/variables.cfg',
            idf_path = '/home/jjimenez/energym/energym/data/buildings/5ZoneAutoDXVAV.idf',
            env_name = 'eplus-demo-v1'
        )
        # Observation space
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=(16,), dtype=np.float32)
        # Action space
        self.action_type = action_type
        if action_type == 'discrete':
            # 9 possible actions - [Heating SetPoint, Cooling Setpoint]
            self.action_mapping = {
                0: [21, 19], 1: [21, 20], 2: [21, 21],
                3: [22, 19], 4: [22, 20], 5: [22, 21],
                6: [23, 19], 7: [23, 20], 8: [23, 21] 
            }
            self.action_space = gym.spaces.Discrete(9)
        elif action_type == 'multidiscrete':
            # 25 possible actions - [Heating SetPoint, Cooling Setpoint]
            self.action_mapping = {
                (0,0): [25, 17], (0,1): [25, 18], (0,2): [25, 19], (0,3): [25, 20], (0,4): [25, 21],
                (1,0): [24, 17], (1,1): [24, 18], (1,2): [24, 19], (1,3): [24, 20], (1,4): [24, 21],
                (2,0): [23, 17], (2,1): [23, 18], (2,2): [23, 19], (2,3): [23, 20], (2,4): [23, 21],
                (3,0): [22, 17], (3,1): [22, 18], (3,2): [22, 19], (3,3): [22, 20], (3,4): [22, 21],
                (4,0): [21, 17], (4,1): [21, 18], (4,2): [21, 19], (4,3): [21, 20], (4,4): [21, 21],
            }
            self.action_space = gym.spaces.MultiDiscrete((5, 5))
        else:
            raise NotImplementedError('Action type not supported.')

    def step(self, action):
        """"""

        if self.action_type == 'multidiscrete':
            a = self.action_mapping[tuple(action)]
        else:
            a = self.action_mapping[action]
        print(a)
        # Send action to de simulator
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

    def _get_reward(self, temperature, power, beta = 0.001):
        """"""
        reward = 0.0
        if temperature < self.comfort_range[0]:
            reward -= self.comfort_range[0] - temperature
        if temperature < self.comfort_range[1]:
            reward -= temperature - self.comfort_range[1]
        reward -= beta * power
        return reward