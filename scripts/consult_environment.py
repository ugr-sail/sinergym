import gymnasium as gym

import sinergym

# Get the list of available environments
print(sinergym.__version__)
print(sinergym.__ids__)

# Make and consult some of the environments
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
print(env.info())
