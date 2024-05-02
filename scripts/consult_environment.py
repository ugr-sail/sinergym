import gymnasium as gym

import sinergym
from sinergym.utils.common import get_ids

# Get the list of available environments
sinergym_environment_ids = get_ids()
print(sinergym_environment_ids)

# Make and consult some of the environments
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
print(env.info())
