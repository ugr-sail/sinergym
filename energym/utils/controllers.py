"""Implementation of basic controllers."""

import os
import pkg_resources

from datetime import datetime

from ..utils.common import parse_variables


class RandomController(object):

    def __init__(self, env):
        """Random agent. It selects available actions randomly.

        Args:
            env (object): Simulation environment.
        """
        self.env = env

    def act(self, observation=None):
        """Selects a random action from the environment's `action_space`.

        Args:
            observation (object, optional): Perceived observation. Defaults to None.

        Returns:
            object: Action chosen.
        """
        action = self.env.action_space.sample()
        return action


class RuleBasedController(object):

    def __init__(
        self, env, range_comfort_winter=(
            20.0, 23.5), range_comfort_summer=(
            23.0, 26.0)):
        """Agent whose actions are based on static rules.

        Args:
            env (object): Simulation environment.
            range_comfort_winter (tuple, optional): Comfort temperature range for cool season. Defaults to (20.0, 23.5).
            range_comfort_summer (tuple, optional): Comfort temperature range for hot season. Defaults to (23.0, 26.0).
        """

        year = 2021

        self.env = env
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer

        self.variables_path = self.env.variables_path
        self.variables = parse_variables(self.variables_path)
        self.variables['observation'].extend(['day', 'month', 'hour'])

        self.summer_start_date = datetime(year, 6, 1)
        self.summer_final_date = datetime(year, 9, 30)

    def act(self, observation):
        """Select action based on outdoor air drybulb temperature.

        Args:
            observation (object): Perceived observation.

        Returns:
            object: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))
        out_temp = obs_dict['Site Outdoor Air Drybulb Temperature (Environment)']

        if out_temp < 15:  # t < 15
            action = (19, 21)
        elif out_temp < 20:  # 15 <= t < 20
            action = (20, 22)
        elif out_temp < 26:  # 20 <= t < 26
            action = (21, 23)
        elif out_temp < 30:  # 26 <= t < 30
            action = (26, 30)
        else:  # t >= 30
            action = (24, 26)

        return action
