"""Implementation of different types of rewards."""

from datetime import datetime
from math import exp
from typing import Dict, List, Tuple

YEAR = 2021


class LinearReward():

    def __init__(self,
                 range_comfort_winter: Tuple[float, float] = (20.0, 23.5),
                 range_comfort_summer: Tuple[float, float] = (23.0, 26.0),
                 energy_weight: float = 0.5,
                 lambda_energy: float = 1e-4,
                 lambda_temperature: float = 1.0
                 ):
        """Simple reward considering absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            range_comfort_winter (Tuple[float, float], optional): Temperature comfort range for cold season. Defaults to (20.0, 23.5).
            range_comfort_summer (Tuple[float, float], optional): Temperature comfort range for hot season. Defaults to (23.0, 26.0).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        # Variables
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Periods
        self.summer_start_date = datetime(YEAR, 6, 1)
        self.summer_final_date = datetime(YEAR, 9, 30)

    def calculate(self,
                  power: float,
                  temperatures: List[float],
                  month: int,
                  day: int) \
            -> Tuple[float, Dict[str, float]]:
        """Reward calculus.

        Args:
            power (float): Power consumption.
            temperatures (List[float]): Indoor temperatures (one per zone).
            month (int): Current month.
            day (int): Current day.

        Returns:
            Tuple[float, Dict[str, float]]: Reward value.
        """
        # Energy term
        reward_energy = - self.lambda_energy * power

        # Comfort term
        current_dt = datetime(YEAR, month, day)
        range_T = self.range_comfort_summer if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date else self.range_comfort_winter
        delta_T = 0.0
        for temperature in temperatures:
            delta_T += 0.0 if temperature >= range_T[0] and temperature <= range_T[1] else min(
                abs(range_T[0] - temperature), abs(temperature - range_T[1]))
        reward_comfort = - self.lambda_temp * delta_T

        # Weighted sum of both terms
        reward = self.W_energy * reward_energy + \
            (1.0 - self.W_energy) * reward_comfort
        terms = {'reward_energy': reward_energy,
                 'reward_comfort': reward_comfort}

        return reward, terms


class ExpReward():

    def __init__(self,
                 range_comfort_winter: Tuple[float, float] = (20.0, 23.5),
                 range_comfort_summer: Tuple[float, float] = (23.0, 26.0),
                 energy_weight: float = 0.5,
                 lambda_energy: float = 1e-4,
                 lambda_temperature: float = 1.0
                 ):
        """Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            range_comfort_winter (Tuple[float, float], optional): Temperature comfort range for cold season. Defaults to (20.0, 23.5).
            range_comfort_summer (Tuple[float, float], optional): Temperature comfort range for hot season. Defaults to (23.0, 26.0).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        # Variables
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Periods
        self.summer_start_date = datetime(YEAR, 6, 1)
        self.summer_final_date = datetime(YEAR, 9, 30)

    def calculate(self, power: float, temperatures: List[float],
                  month: int, day: int) -> Tuple[float, Dict[str, float]]:
        """Reward calculus.

        Args:
            power (float): Power consumption.
            temperatures (List[float]): Indoor temperatures (one per zone).
            month (int): Current month.
            day (int): Current day.

        Returns:
            Tuple[float, Dict[str, float]]: Reward value.
        """
        # Energy term
        reward_energy = - self.lambda_energy * power

        # Comfort term
        current_dt = datetime(YEAR, month, day)
        range_T = self.range_comfort_summer if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date else self.range_comfort_winter
        delta_T = 0.0
        for temperature in temperatures:
            delta_T += 0.0 if temperature >= range_T[0] and temperature <= range_T[1] else exp(
                min(abs(range_T[0] - temperature), abs(temperature - range_T[1])))
        reward_comfort = - self.lambda_temp * delta_T

        # Weighted sum of both terms
        reward = self.W_energy * reward_energy + \
            (1.0 - self.W_energy) * reward_comfort
        terms = {'reward_energy': reward_energy,
                 'reward_comfort': reward_comfort}

        return reward, terms
