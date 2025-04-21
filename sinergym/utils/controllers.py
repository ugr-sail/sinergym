"""Implementation of basic controllers."""
from datetime import datetime
from typing import Any, List, Sequence

import numpy as np
from gymnasium import Env

from sinergym.utils.constants import YEAR


class RandomController(object):

    def __init__(self, env: Env):
        """Random agent. It selects available actions randomly.

        Args:
            env (Env): Simulation environment.
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

    def __init__(self, env: Env) -> None:
        """Agent based on static rules for controlling 5ZoneAutoDXVAV setpoints.
        Based on ASHRAE Standard 55-2004: Thermal Environmental Conditions for Human Occupancy.

        Args:
            env (Env): Simulation environment
        """

        self.env = env

        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        self.setpoints_summer = np.array((23.0, 26.0), dtype=np.float32)
        self.setpoints_winter = np.array((20.0, 23.5), dtype=np.float32)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.

        Args:
            observation (List[Any]): Perceived observation.

        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))
        year = int(obs_dict['year']) if obs_dict.get('year', False) else YEAR
        month = int(obs_dict['month'])
        day = int(obs_dict['day_of_month'])

        summer_start_date = datetime(year, 6, 1)
        summer_final_date = datetime(year, 9, 30)

        current_dt = datetime(year, month, day)

        # Get season comfort range
        if current_dt >= summer_start_date and current_dt <= summer_final_date:  # pragma: no cover
            season_range = self.setpoints_summer
        else:  # pragma: no cover
            season_range = self.setpoints_winter

        return season_range


class RBCIncremental5Zone(object):

    def __init__(self, env: Env) -> None:
        """Agent based on rules for controlling 5ZoneAutoDXVAV setpoints in a incremental way.
        Args:
            env (Env): Simulation environment
        """

        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        self.range_winter = (20.0, 23.0)
        self.range_summer = (23.5, 26.0)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.
        Args:
            observation (List[Any]): Perceived observation.
        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))

        # Mean temp in datacenter zones
        mean_temp = np.mean([obs_dict['air_temperature_central'],
                             obs_dict['air_temperature_peripheral1'],
                             obs_dict['air_temperature_peripheral2'],
                             obs_dict['air_temperature_peripheral3'],
                             obs_dict['air_temperature_peripheral4'],])

        current_heat_setpoint = np.clip(obs_dict['htg_setpoint'],
                                        self.env.action_space.low[0],
                                        self.env.action_space.high[0])
        current_cool_setpoint = np.clip(obs_dict['clg_setpoint'],
                                        self.env.action_space.low[1],
                                        self.env.action_space.high[1])

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        # Check if current date is in summer (June 1 - Sept 30)
        month = obs_dict['month']
        day = obs_dict['day_of_month']

        if month in [6, 7, 8, 9]:
            current_range = self.range_summer
        else:
            current_range = self.range_winter

        if mean_temp < current_range[0]:  # pragma: no cover
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif mean_temp > current_range[1]:  # pragma: no cover
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        return np.clip(
            np.array([new_heat_setpoint, new_cool_setpoint], dtype=np.float32),
            self.env.action_space.low,
            self.env.action_space.high)


class RBCDatacenter(object):

    def __init__(self, env: Env) -> None:
        """Agent based on static rules for controlling 2ZoneDataCenterHVAC setpoints.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).

        Args:
            env (Env): Simulation environment
        """

        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = np.array((18, 27), dtype=np.float32)

    def act(self) -> Sequence[Any]:
        """Select same action always, corresponding with comfort range.

        Returns:
            Sequence[Any]: Action chosen.
        """
        return self.range_datacenter


class RBCIncrementalDatacenter(object):

    def __init__(self, env: Env) -> None:
        """Agent based on rules for controlling 2ZoneDataCenterHVAC setpoints in a incremental way.
        Follows the ASHRAE recommended temperature ranges for data centers described in ASHRAE TC9.9 (2016).
        Args:
            env (Env): Simulation environment
        """

        self.env = env
        self.observation_variables = env.get_wrapper_attr(
            'observation_variables')
        self.action_variables = env.get_wrapper_attr('action_variables')

        # ASHRAE recommended temperature range = [18, 27] Celsius
        self.range_datacenter = (18, 27)

    def act(self, observation: List[Any]) -> Sequence[Any]:
        """Select action based on indoor temperature.
        Args:
            observation (List[Any]): Perceived observation.
        Returns:
            Sequence[Any]: Action chosen.
        """
        obs_dict = dict(zip(self.observation_variables, observation))

        # Mean temp in datacenter zones
        mean_temp = np.mean([obs_dict['west_zone_air_temperature'],
                             obs_dict['east_zone_air_temperature']])

        current_heat_setpoint = obs_dict[
            'west_zone_htg_setpoint']
        current_cool_setpoint = obs_dict[
            'west_zone_clg_setpoint']

        new_heat_setpoint = current_heat_setpoint
        new_cool_setpoint = current_cool_setpoint

        if mean_temp < self.range_datacenter[0]:  # pragma: no cover
            new_heat_setpoint = current_heat_setpoint + 1
            new_cool_setpoint = current_cool_setpoint + 1
        elif mean_temp > self.range_datacenter[1]:  # pragma: no cover
            new_cool_setpoint = current_cool_setpoint - 1
            new_heat_setpoint = current_heat_setpoint - 1

        return np.array(
            (new_heat_setpoint,
             new_cool_setpoint),
            dtype=np.float32)
