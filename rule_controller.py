import gym
import numpy as np

from sinergym.utils.controllers import RuleBasedController

env = gym.make('Eplus-5Zone-mixed-continuous-v1')

# create rule-controlled agent
agent = RuleBasedController(env)

for i in range(1):
    obs = env.reset()
    rewards = []
    done = False
    current_month = 0
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
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
