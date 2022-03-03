"""Implementation of basic controllers."""
from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple

from numpy import arange

from ..utils.common import get_season_comfort_range, parse_variables


class RandomController(object):

    def __init__(self, env: Any):
        """Random agent. It selects available actions randomly.

        Args:
            env (Any): Simulation environment.
        """
        self.env = env

    def act(self, observation: Optional[List[Any]] = None) -> Sequence[Any]:
        """Selects a random action from the environment's `action_space`.

        Args:
            observation (Optional[List[Any]], optional): Perceived observation. Defaults to None.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = self.env.action_space.sample()
        return action


class RuleBasedController(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules.

        Args:
            env (Any): Simulation environment
        """

        self.env = env

        self.variables_path = self.env.variables_path
        self.variables = parse_variables(self.variables_path)
        self.variables['observation'].extend(['day', 'month', 'hour'])

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on outdoor air drybulb temperature and daytime.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))

        out_temp = obs_dict['Site Outdoor Air Drybulb Temperature (Environment)']

        day = int(obs_dict['day'])
        month = int(obs_dict['month'])
        hour = int(obs_dict['hour'])

        season_comfort_range = get_season_comfort_range(month, day)

        if out_temp not in arange(
                season_comfort_range[0], season_comfort_range[1], .1):
            if hour in range(6, 18):  # day
                action = (19.44, 25.0)
            elif hour in range(18, 22):  # evening
                action = (20.0, 24.44)
            else:  # night
                action = (18.33, 23.33)
        else:  # maintain setpoints if comfort requirements are already met
            current_cool_setpoint = obs_dict[
                'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)']
            current_heat_setpoint = obs_dict[
                'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)']
            action = (current_heat_setpoint, current_cool_setpoint)

        return action
