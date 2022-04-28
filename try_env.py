import gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import LoggerWrapper

extra_conf = {
    'action_space': {
        'ThermostatSetpoint:DualSetpoint': [{
            'name': 'Space1-DualSetP-RL',
            'heating_name': 'Space1-HtgSetP-RL',
            'cooling_name': 'Space1-ClgSetP-RL',
            'zones': ['SPACE1-1']
        }]
    }
}
env = gym.make('Eplus-demo-v1')
env.action_space = gym.spaces.Box(low=0, high=0, shape=(0,))
env = LoggerWrapper(env)
for i in range(1):
    obs = env.reset()
    rewards = []
    done = False
    current_month = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        rewards.append(reward)
        if info['month'] != current_month:  # display results every month
            current_month = info['month']
            print('Reward: ', sum(rewards), info)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()
