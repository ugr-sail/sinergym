import gym
import energym
import numpy as np

env = gym.make('Eplus-demo-v1')
for i in range(1):
    obs = env.reset()
    rewards = []
    done = False
    k = 0
    while not done:
        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        rewards.append(reward)
        k += 1
        if k % 5000 == 0:
            print(k, reward, info)
    print('Episode ', i, 'Mean reward: ', np.mean(rewards), 'Cumulative reward: ', sum(rewards))
env.close()