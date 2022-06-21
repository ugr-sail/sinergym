"""Implementation of reward functions."""


from datetime import datetime
from math import exp
from typing import Any, Dict, List, Tuple, Union

from gym import Env


class BaseReward(object):

    def __init__(self, env):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """
        self.env = env
        self.year = 2021  # just for datetime completion

    def __call__(self):
        """Method for calculating the reward function."""
        raise NotImplementedError(
            "Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: tuple,
        range_comfort_summer: tuple,
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (tuple): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (tuple): Temperature comfort range for hot season. Depends on environment you are using.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(LinearReward, self).__init__(env)

        # Name of the variables
        self.temp_name = temperature_variable
        self.energy_name = energy_variable

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Periods
        self.summer_start_date = datetime(self.year, 6, 1)
        self.summer_final_date = datetime(self.year, 9, 30)

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        # Energy term
        reward_energy = - self.lambda_energy * obs_dict[self.energy_name]

        # Comfort
        comfort, temps = self._get_comfort(obs_dict)
        reward_comfort = - self.lambda_temp * comfort

        # Weighted sum of both terms
        reward = self.W_energy * reward_energy + \
            (1.0 - self.W_energy) * reward_comfort

        reward_terms = {
            'reward_energy': reward_energy,
            'total_energy': obs_dict[self.energy_name],
            'reward_comfort': reward_comfort,
            'abs_comfort': comfort,
            'temperatures': temps
        }

        return reward, reward_terms

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float]]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

        month = obs_dict['month']
        day = obs_dict['day']
        current_dt = datetime(self.year, month, day)

        if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temps = [v for k, v in obs_dict.items() if k in self.temp_name]
        comfort = 0.0
        for T in temps:
            if T < temp_range[0] or T > temp_range[1]:
                comfort += min(abs(temp_range[0] - T), abs(T - temp_range[1]))

        return comfort, temps


class ExpReward(LinearReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: tuple,
        range_comfort_summer: tuple,
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (tuple): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (tuple): Temperature comfort range for hot season. Depends on environment you are using.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(ExpReward, self).__init__(
            env,
            temperature_variable,
            energy_variable,
            range_comfort_winter,
            range_comfort_summer,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float]]:
        """Calculate the comfort term of the reward.

        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

        month = obs_dict['month']
        day = obs_dict['day']
        current_dt = datetime(self.year, month, day)

        if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temps = [v for k, v in obs_dict.items() if k in self.temp_name]
        comfort = 0.0
        for T in temps:
            if T < temp_range[0] or T > temp_range[1]:
                comfort += exp(min(abs(temp_range[0] - T),
                                   abs(T - temp_range[1])))

        return comfort, temps


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        env: Env,
        temperature_variable: Union[str, list],
        energy_variable: str,
        range_comfort_winter: tuple,
        range_comfort_summer: tuple,
        min_energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            range_comfort_winter (tuple): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (tuple): Temperature comfort range for hot season. Depends on environment you are using.
            min_energy_weight (float, optional): Minimum weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            env,
            temperature_variable,
            energy_variable,
            range_comfort_winter,
            range_comfort_summer,
            min_energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours

    def __call__(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Returns:
            Tuple[float, Dict[str, Any]]: Reward and dict with reward terms.
            """
        # Current observation
        obs_dict = self.env.obs_dict.copy()

        # Energy term
        reward_energy = - self.lambda_energy * obs_dict[self.energy_name]

        # Comfort
        comfort, temps = self._get_comfort(obs_dict)
        reward_comfort = - self.lambda_temp * comfort

        # Determine energy weight depending on the hour
        hour = obs_dict['hour']
        if hour >= self.range_comfort_hours[0] and hour <= self.range_comfort_hours[1]:
            weight = self.W_energy
        else:
            weight = 1.0

        # Weighted sum of both terms
        reward = weight * reward_energy + (1.0 - weight) * reward_comfort

        reward_terms = {
            'reward_energy': reward_energy,
            'total_energy': obs_dict[self.energy_name],
            'reward_comfort': reward_comfort,
            'temperatures': temps
        }

        return reward, reward_terms
