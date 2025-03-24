"""Implementation of reward functions."""


from datetime import datetime
from math import exp
from typing import Any, Dict, List, Optional, Tuple, Union

from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from sinergym.utils.logger import TerminalLogger


class BaseReward(object):

    logger = TerminalLogger().getLogger(name='REWARD',
                                        level=LOG_REWARD_LEVEL)

    def __init__(self):
        """
        Base reward class.

        All reward functions should inherit from this class.

        Args:
            env (Env): Gym environment.
        """

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Method for calculating the reward function."""
        raise NotImplementedError(
            "Reward class must have a `__call__` method.")


class LinearReward(BaseReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Linear reward function.

        It considers the energy consumption and the absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0))

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Basic validations
        if not (0 <= energy_weight <= 1):
            self.logger.error(
                f'energy_weight must be between 0 and 1. Received: {energy_weight}')
            raise ValueError
        if not all(isinstance(v, str)
                   for v in temperature_variables + energy_variables):
            self.logger.error('All variable names must be strings.')
            raise TypeError

        # Name of the variables
        self.temp_names = temperature_variables
        self.energy_names = energy_variables

        # Reward parameters
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

        # Summer period
        self.summer_start = summer_start  # (month, day)
        self.summer_final = summer_final  # (month, day)

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the temperature violation (ºC) in each observation's temperature variable.

        Returns:
            List[float]: List with temperature violation in each zone.
        """

        # Current datetime and summer period
        current_dt = datetime(
            YEAR, int(
                obs_dict['month']), int(
                obs_dict['day_of_month']))
        summer_start_date = datetime(YEAR, *self.summer_start)
        summer_final_date = datetime(YEAR, *self.summer_final)

        temp_range = self.range_comfort_summer if \
            summer_start_date <= current_dt <= summer_final_date else \
            self.range_comfort_winter

        temp_values = [obs_dict[v] for v in self.temp_names]

        return [max(temp_range[0] - T, 0, T - temp_range[1])
                for T in temp_values]

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term


class EnergyCostLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        energy_cost_variables: List[str],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.4,
        temperature_weight: float = 0.4,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        lambda_energy_cost: float = 1.0
    ):
        """
        Linear reward function with the addition of the energy cost term.

        Considers energy consumption, absolute difference to thermal comfort and energy cost.

        .. math::
            R = - W_E * lambda_E * power - W_T * lambda_T * (max(T - T_{low}, 0) + max(T_{up} - T, 0)) - (1 - W_P - W_T) * lambda_EC * power_cost

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer s-sum(exp(violation)
                    for violation in temp_violations if violation > 0)ession tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.4.
            temperature_weight (float, optional): Weight given to the temperature term. Defaults to 0.4.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1.0.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            lambda_energy_cost (flota, optional): Constant for removing dimensions from temperature(1/E). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

        self.energy_cost_names = energy_cost_variables
        self.W_temperature = temperature_weight
        self.lambda_energy_cost = lambda_energy_cost

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Energy cost calculation
        energy_cost_values = self._get_money_spent(obs_dict)
        self.total_energy_cost = sum(energy_cost_values)
        self.energy_cost_penalty = -self.total_energy_cost

        # Weighted sum of terms
        reward, energy_term, comfort_term, energy_cost_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_cost_term': energy_cost_term,
            'reward_energy_weight': self.W_energy,
            'reward_temperature_weight': self.W_temperature,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'energy_cost_penalty': self.energy_cost_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'money_spent': self.total_energy_cost
        }

        return reward, reward_terms

    def _get_money_spent(self, obs_dict: Dict[str,
                                              Any]) -> List[float]:
        """Calculate the total money spent in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with money spent in each energy cost variable.
        """
        return [v for k, v in obs_dict.items() if k in self.energy_cost_names]

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using the negative absolute comfort, energy penalty and energy cost penalty calculates previously.

        Returns:
            Tuple[float, ...]: Total reward calculated, reward term for energy, reward term for comfort and reward term for energy cost.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            self.W_temperature * self.comfort_penalty
        energy_cost_term = self.lambda_energy_cost * \
            (1 - self.W_energy - self.W_temperature) * self.energy_cost_penalty

        reward = energy_term + comfort_term + energy_cost_term
        return reward, energy_term, comfort_term, energy_cost_term


class ExpReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        Reward considering exponential absolute difference to temperature comfort.

        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * exp( (max(T - T_{low}, 0) + max(T_{up} - T, 0)) )

        Args:
            temperature_variables (List[str]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight,
            lambda_energy,
            lambda_temperature
        )

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        # Exponential Penalty
        self.comfort_penalty = -sum(exp(violation)
                                    for violation in temp_violations if violation > 0)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms


class HourlyLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        default_energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0,
        range_comfort_hours: tuple = (9, 19),
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            default_energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            range_comfort_hours (tuple, optional): Hours where thermal comfort is considered. Defaults to (9, 19).
        """

        super(HourlyLinearReward, self).__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            default_energy_weight,
            lambda_energy,
            lambda_temperature
        )

        # Reward parameters
        self.range_comfort_hours = range_comfort_hours
        self.default_energy_weight = default_energy_weight

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """
        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Determine reward weight depending on the hour
        self.W_energy = self.default_energy_weight if self.range_comfort_hours[
            0] <= obs_dict['hour'] <= self.range_comfort_hours[1] else 1.0

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy
        }

        return reward, reward_terms


class NormalizedLinearReward(LinearReward):

    def __init__(
        self,
        temperature_variables: List[str],
        energy_variables: List[str],
        range_comfort_winter: Tuple[int, int],
        range_comfort_summer: Tuple[int, int],
        summer_start: Tuple[int, int] = (6, 1),
        summer_final: Tuple[int, int] = (9, 30),
        energy_weight: float = 0.5,
        max_energy_penalty: float = 8,
        max_comfort_penalty: float = 12,
    ):
        """
        Linear reward function with a time-dependent weight for consumption and energy terms.

        Args:
            temperature_variables (List[str]]): Name(s) of the temperature variable(s).
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            range_comfort_winter (Tuple[int,int]): Temperature comfort range for cold season. Depends on environment you are using.
            range_comfort_summer (Tuple[int,int]): Temperature comfort range for hot season. Depends on environment you are using.
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Default weight given to the energy term when thermal comfort is considered. Defaults to 0.5.
            max_energy_penalty (float, optional): Maximum energy penalty value. Defaults to 8.
            max_comfort_penalty (float, optional): Maximum comfort penalty value. Defaults to 12.
        """

        super().__init__(
            temperature_variables,
            energy_variables,
            range_comfort_winter,
            range_comfort_summer,
            summer_start,
            summer_final,
            energy_weight
        )

        # Reward parameters
        self.max_energy_penalty = max_energy_penalty
        self.max_comfort_penalty = max_comfort_penalty

    def _get_reward(self) -> Tuple[float, ...]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Aplying normalization

        Returns:
            Tuple[float, ...]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(
            self.max_energy_penalty, self.energy_penalty)
        self.max_comfort_penalty = max(
            self.max_comfort_penalty, self.comfort_penalty)

        # Calculate normalization
        energy_norm = self.energy_penalty / \
            self.max_energy_penalty if self.max_energy_penalty else 0
        comfort_norm = self.comfort_penalty / \
            self.max_comfort_penalty if self.max_comfort_penalty else 0

        # Calculate reward terms with norm values
        energy_term = self.W_energy * energy_norm
        comfort_term = (1 - self.W_energy) * comfort_norm
        reward = energy_term + comfort_term

        return reward, energy_term, comfort_term


class MultiZoneReward(BaseReward):

    def __init__(
        self,
        energy_variables: List[str],
        temperature_and_setpoints_conf: Dict[str, str],
        comfort_threshold: float = 0.5,
        energy_weight: float = 0.5,
        lambda_energy: float = 1.0,
        lambda_temperature: float = 1.0
    ):
        """
        A linear reward function for environments with different comfort ranges in each zone. Instead of having
        a fixed and common comfort range for the entire building, each zone has its own comfort range, which is
        directly obtained from the setpoints established in the building. This function is designed for buildings
        where temperature setpoints are not controlled directly but rather used as targets to be achieved, while
        other actuators are controlled to reach these setpoints. A setpoint observation variable can be assigned
        per zone if it is available in the specific building. It is also possible to assign the same setpoint
        variable to multiple air temperature zones.

        Args:
            energy_variables (List[str]): Name(s) of the energy/power variable(s).
            temperature_and_setpoints_conf (Dict[str, str]): Dictionary with the temperature variable name (key) and the setpoint variable name (value) of the observation space.
            comfort_threshold (float, optional): Comfort threshold for temperature range (+/-). Defaults to 0.5.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super().__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        self.comfort_ranges = {}

        self.logger.info('Reward function initialized.')

    def __call__(self, obs_dict: Dict[str, Any]
                 ) -> Tuple[float, Dict[str, Any]]:
        """Calculate the reward function value based on observation data.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value)

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        # Energy calculation
        energy_values = self._get_energy_consumed(obs_dict)
        self.total_energy = sum(energy_values)
        self.energy_penalty = -self.total_energy

        # Comfort violation calculation
        temp_violations = self._get_temperature_violation(obs_dict)
        self.total_temp_violation = sum(temp_violations)
        self.comfort_penalty = -self.total_temp_violation

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward()

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_penalty': self.energy_penalty,
            'comfort_penalty': self.comfort_penalty,
            'total_power_demand': self.total_energy,
            'total_temperature_violation': self.total_temp_violation,
            'reward_weight': self.W_energy,
            'comfort_threshold': self.comfort_threshold
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> List[float]:
        """Calculate the energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            List[float]: List with energy consumed in each energy variable.
        """
        return [obs_dict[v] for v in self.energy_names]

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> List[float]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
           List[float]: List with temperature violation (ºC) in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)

        temp_violations = [
            max(0, min(abs(T - comfort_range[0]), abs(T - comfort_range[1])))
            if T < comfort_range[0] or T > comfort_range[1] else 0
            for temp_var, comfort_range in self.comfort_ranges.items()
            if (T := obs_dict[temp_var])
        ]

        return temp_violations

    def _get_comfort_ranges(
            self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        self.comfort_ranges = {
            temp_var: (setpoint - self.comfort_threshold, setpoint + self.comfort_threshold)
            for temp_var, setpoint_var in self.comfort_configuration.items()
            if (setpoint := obs_dict[setpoint_var]) is not None
        }

    def _get_reward(self) -> Tuple[float, ...]:
        """Compute the final reward value.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float, ...]: Total reward calculated and reward terms.
        """
        energy_term = self.lambda_energy * self.W_energy * self.energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * self.comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
