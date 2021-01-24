import gym
from ..simulators import EnergyPlus

class EplusDemo(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.simulator = EnergyPlus()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass