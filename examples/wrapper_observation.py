import energym
import numpy as np
import gym


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize the observation to range [-1, 1]."""
    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)

    def observation(self, obs):
        return np.tanh(obs)


def main():
    env_ = gym.make('Eplus-demo-v1')
    env = NormalizeObservation(env_)
    obs_ = env_.reset()
    obs = env.reset()
    print('Previous observation: ', obs_)
    print('Current observation: ', obs)
    env.close()


if __name__ == '__main__':
    main()