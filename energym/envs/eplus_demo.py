import gym
import os
from ..simulators import EnergyPlus

class EplusDemo(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        eplus_path = os.environ['EPLUS_PATH']
        bcvtb_path = os.environ['BCVTB_PATH']

        self.simulator = EnergyPlus(
            eplus_path = eplus_path,
            weather_path = '/home/jjimenez/energym/energym/data/weather/USA_Pittsburg_TMY3.epw',
            bcvtb_path = bcvtb_path,
            variable_path = '/home/jjimenez/energym/energym/data/variables/variables.cfg',
            idf_path = '/home/jjimenez/energym/energym/data/buildings/5ZoneAutoDXVAV.idf',
            env_name = 'eplus-demo-v1'
        )

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-5e6, high=5e6, shape=(15,), dtype=np.float32)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass