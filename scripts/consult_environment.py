import gymnasium as gym

import sinergym

# Get the list of available environments
print(sinergym.__version__)
print(sinergym.ids())

# Make and consult environment
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
print(env.get_wrapper_attr('to_str')())
