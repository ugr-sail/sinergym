import gym
import energym
import numpy as np

env = gym.make('Eplus-discrete-v1')
obs = env.reset()
for i in range(9):
    obs = env.reset()
    rewards = []
    done = False
    k = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        rewards.append(reward)
        k += 1
    print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
env.close()