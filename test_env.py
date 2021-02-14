import gym
import energym
import numpy as np

env = gym.make('Eplus-discrete-v1')
obs = env.reset()
for i in range(11):
    obs = env.reset()
    powers = [obs[-1]]
    done = False
    while not done:
        obs, reward, done, info = env.step(2)
        powers.append(obs[-1])
    print(np.mean(powers), np.min(powers), np.max(powers), np.std(powers))
env.close()