import gym
import energym
import numpy as np

env = gym.make('Eplus-discrete-v1')
obs = env.reset()
for i in range(10):
    obs = env.reset()
    rewards = []
    done = False
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        rewards.append(reward)
    print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
env.close()