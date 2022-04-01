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

    def act(self) -> Sequence[Any]:
        """Selects a random action from the environment's action_space.

        Returns:
            Sequence[Any]: Action chosen.
        """
        action = self.env.action_space.sample()
        return action


class RBC5Zone(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules for controlling 5ZoneAutoDXVAV setpoints.
        Follows the FSEC standard setpoints described in FSEC-CR-2010-13 (2013).

        Args:
            env (Any): Simulation environment
        """

        self.env = env

        self.variables_path = self.env.variables_path
        self.variables = parse_variables(self.variables_path)
        self.variables['observation'].extend(['year', 'month', 'day', 'hour'])

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on outdoor air drybulb temperature and daytime.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))

        out_temp = obs_dict['Site Outdoor Air Drybulb Temperature (Environment)']

        year = int(obs_dict['year'])
        month = int(obs_dict['month'])
        day = int(obs_dict['day'])
        hour = int(obs_dict['hour'])

        season_comfort_range = get_season_comfort_range(year, month, day)
        
        if out_temp < season_comfort_range[0] or out_temp > season_comfort_range[1]:
            if hour in range(6, 18): # day
                action = (19.44, 25.0)
            elif hour in range(18, 22): # evening
                action = (20.0, 24.44)
            else:  # night
                action = (18.33, 23.33)
        else: # maintain setpoints if comfort requirements are already met
            current_cool_setpoint = obs_dict[
                'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)']
            current_heat_setpoint = obs_dict[
                'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)']
            action = (current_heat_setpoint, current_cool_setpoint)

        return action

class RBCDatacenter(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints. 
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).

        Args:
            env (Any): Simulation environment
        """

        self.env = env

        self.variables_path = self.env.variables_path
        self.variables = parse_variables(self.variables_path)
        self.variables['observation'].extend(['year', 'month', 'day', 'hour'])

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on outdoor air drybulb temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))

        out_temp = obs_dict['Site Outdoor Air Drybulb Temperature (Environment)']

        # ASHRAE recommended temperature range = [18, 27] Celsius

        if out_temp < 10: # < 10
            action = (18, 27, 18, 27) 
        elif out_temp < 15: # [10, 15)
            action = (19, 27, 19, 27)
        elif out_temp < 18: # [15, 18)
            action = (20, 27, 20, 27)
        elif out_temp > 40: # > 40
            action = (18, 27, 18, 27)
        elif out_temp > 30: # (30, 40]
            action = (18, 25, 18, 25)
        elif out_temp > 27: # (27, 30]
            action = (18, 23, 18, 23)
        else:
            west_heat = obs_dict['Zone Thermostat Heating Setpoint Temperature (West Zone)']
            west_cool = obs_dict['Zone Thermostat Cooling Setpoint Temperature (West Zone)']
            east_heat = obs_dict['Zone Thermostat Heating Setpoint Temperature (East Zone)']
            east_cool = obs_dict['Zone Thermostat Cooling Setpoint Temperature (East Zone)']
            action = (west_heat, west_cool, east_heat, east_cool)

        return action
