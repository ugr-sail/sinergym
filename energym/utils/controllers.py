"""Script for implementing rule-based controllers."""


class RandomController(object):
    """Selects actions randomly."""
    def __init__(self, env):
        self.env = env

    def act(self, observation = None):
        action = self.env.action_space.sample()
        return action