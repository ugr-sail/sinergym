import gym
import numpy as np

import sinergym

#include of the environment, this uses the .idf and .cfg files information
env = gym.make('Eplus-demo-v1')

for i in range(1): #the number of episodes we whant to loop for, in this case only 1
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
