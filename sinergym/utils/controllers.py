"""Implementation of basic controllers."""
from datetime import datetime
from typing import Any, List, Sequence

from ..utils.common import parse_variables


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
        Follows the standard setpoints described in ASHRAE Standard 55-2004: Thermal Environmental Conditions for Human Occupancy.

        Args:
            env (Any): Simulation environment
        """

        self.env = env

        self.variables = env.variables

        self.range_comfort_summer = (23.0, 26.0)
        self.range_comfort_winter = (20.0, 23.5)

        self.range_comfort_summer = (23.0, 26.0)
        self.range_comfort_winter = (20.0, 23.5)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))
        year = int(obs_dict['year'])
        month = int(obs_dict['month'])
        day = int(obs_dict['day'])

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            season_comfort_range = self.range_comfort_summer
        else:
            season_comfort_range = self.range_comfort_winter

        # Update setpoints
        in_temp = obs_dict['Zone Air Temperature (SPACE1-1)']

        current_heat_setpoint = obs_dict[
            'Zone Thermostat Heating Setpoint Temperature (SPACE1-1)']
        current_cool_setpoint = obs_dict[
            'Zone Thermostat Cooling Setpoint Temperature (SPACE1-1)']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if in_temp < season_comfort_range[0]:
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif in_temp > season_comfort_range[1]:
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        return (new_heat_setpoint, new_cool_setpoint)


class RBCDatacenter(object):

    def __init__(self, env: Any) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).

        Args:
            env (Any): Simulation environment
        """

        self.env = env
        self.variables = env.variables

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_comfort_datacenter = (18, 27)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.variables['observation'], observation))

        # West Zone
        west_in_temp = obs_dict['Zone Air Temperature (West Zone)']

        west_current_heat_setpoint = obs_dict[
            'Zone Thermostat Heating Setpoint Temperature (West Zone)']
        west_current_cool_setpoint = obs_dict[
            'Zone Thermostat Cooling Setpoint Temperature (West Zone)']

        west_new_heat_setpoint = west_current_heat_setpoint
        west_new_cool_setpoint = west_current_cool_setpoint

        if west_in_temp < self.range_comfort_datacenter[0]:
            west_new_heat_setpoint = west_current_heat_setpoint + 1
            west_new_cool_setpoint = west_current_cool_setpoint + 1
        elif west_in_temp > self.range_comfort_datacenter[1]:
            west_new_cool_setpoint = west_current_cool_setpoint - 1
            west_new_heat_setpoint = west_current_heat_setpoint - 1

        # East Zone
        east_in_temp = obs_dict['Zone Air Temperature (East Zone)']

        east_current_heat_setpoint = obs_dict[
            'Zone Thermostat Heating Setpoint Temperature (East Zone)']
        east_current_cool_setpoint = obs_dict[
            'Zone Thermostat Cooling Setpoint Temperature (East Zone)']

        east_new_heat_setpoint = east_current_heat_setpoint
        east_new_cool_setpoint = east_current_cool_setpoint

        if east_in_temp < self.range_comfort_datacenter[0]:
            east_new_heat_setpoint = east_current_heat_setpoint + 1
            east_new_cool_setpoint = east_current_cool_setpoint + 1
        elif east_in_temp > self.range_comfort_datacenter[1]:
            east_new_cool_setpoint = east_current_cool_setpoint - 1
            east_new_heat_setpoint = east_current_heat_setpoint - 1

        return (
            west_new_heat_setpoint,
            west_new_cool_setpoint,
            east_new_heat_setpoint,
            east_new_cool_setpoint)
