"""Script for implementing rule-based controllers."""

import os
import pkg_resources

from datetime import datetime

from ..utils.common import parse_variables

PKG_DATA_PATH = pkg_resources.resource_filename('energym', 'data/')
VARIABLES_FILE = 'variables.cfg'

YEAR = 2021

class RandomController(object):
    """Selects actions randomly."""
    def __init__(self, env):
        self.env = env

    def act(self, observation = None):
        action = self.env.action_space.sample()
        return action

class RuleBasedController(object):
    """Selects actions based on static rules."""
    def __init__(self, env, range_comfort_winter = (20.0, 23.5), range_comfort_summer = (23.0, 26.0)):
        self.env = env
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer

        self.variables_path = os.path.join(PKG_DATA_PATH, 'variables', VARIABLES_FILE)
        self.variables = parse_variables(self.variables_path)
        self.variables.extend(['day', 'month', 'hour'])

        self.summer_start_date = datetime(YEAR, 6, 1)
        self.summer_final_date = datetime(YEAR, 9, 30)

    def act(self, observation):
        obs_dict = dict(zip(self.variables, observation))

        # Set comfort interval depending on season
        current_dt = datetime(YEAR, int(obs_dict['month']), int(obs_dict['day']))
        range_comfort = self.range_comfort_summer if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date else self.range_comfort_winter

        return self.env.step(range_comfort)

        


