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

        super(LinearReward, self).__init__()

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
        self.summer_start = summer_start  # (month,day)
        self.summer_final = summer_final  # (month,day)

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
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
        }

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [obs_dict[v] for v in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """

        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        temp_values = [obs_dict[v] for v in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        for T in temp_values:
            if T < temp_range[0] or T > temp_range[1]:
                temp_violation = min(
                    abs(temp_range[0] - T), abs(T - temp_range[1]))
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation

        return total_temp_violation, temp_violations

    def _get_energy_penalty(self, energy_values: List[float]) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        energy_penalty = -sum(energy_values)
        return energy_penalty

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(temp_violations)
        return comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * comfort_penalty
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
            summer_start (Tuple[int,int]): Summer session tuple with month and day start. Defaults to (6,1).
            summer_final (Tuple[int,int]): Summer session tuple with month and day end. defaults to (9,30).
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.4.
            temperature_weight (float, optional): Weight given to the temperature term. Defaults to 0.4.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1.0.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
            lambda_energy_cost (flota, optional): Constant for removing dimensions from temperature(1/E). Defaults to 1.0.
        """

        super(EnergyCostLinearReward, self).__init__(
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

        if (self.W_energy + self.W_temperature) > 1.0:
            raise ValueError(
                "The sum of the energy and temperature weights cannot be greater than 1.")

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
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Energy cost calculation
        money_spent, energy_cost_values = self._get_money_spent(obs_dict)
        energy_cost_penalty = self._get_energy_cost_penalty(energy_cost_values)

        # Weighted sum of terms
        reward, energy_term, comfort_term, energy_cost_term = self._get_reward_custom(
            energy_penalty, comfort_penalty, energy_cost_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'energy_cost_term': energy_cost_term,
            'reward_energy_weight': self.W_energy,
            'reward_temperature_weight': self.W_temperature,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'abs_energy_cost_penalty': energy_cost_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation,
            'money_spent': money_spent
        }

        return reward, reward_terms

    def _get_money_spent(self, obs_dict: Dict[str,
                                              Any]) -> Tuple[float,
                                                             List[float]]:
        """Calculate the total money spent in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total money spent (sum of variables) and List with money spent in each energy cost variable.
        """
        energy_cost_values = [
            v for k, v in obs_dict.items() if k in self.energy_cost_names]
        total_energy_cost = sum(energy_cost_values)

        return total_energy_cost, energy_cost_values

    def _get_energy_cost_penalty(
            self, energy_cost_values: List[float]) -> float:
        """Calculate the negative absolute energy cost penalty based on energy cost values

        Args:
            energy_cost_values (List[float]): Energy cost values

        Returns:
            float: Negative absolute energy cost penalty value
        """
        energy_cost_penalty = -sum(energy_cost_values)
        return energy_cost_penalty

    def _get_reward_custom(self,
                           energy_penalty: float,
                           comfort_penalty: float,
                           energy_cost_penalty: float) -> Tuple[float,
                                                                float,
                                                                float,
                                                                float]:
        """It calculates reward value using the negative absolute comfort, energy penalty and energy cost penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.
            energy_cost_penalty (float): Negative obsolute energy cost penalty value.

        Returns:
            Tuple[float,float,float,float]: total reward calculated, reward term for energy, reward term for comfort and reward term for energy cost.
        """
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            self.W_temperature * comfort_penalty
        energy_cost_term = self.lambda_energy_cost * \
            (1 - self.W_energy - self.W_temperature) * energy_cost_penalty

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

        super(ExpReward, self).__init__(
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

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values, using an exponential concept when temperature violation > 0.

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(list(map(lambda temp_violation: exp(
            temp_violation) if temp_violation > 0 else 0, temp_violations)))
        return comfort_penalty


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
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Determine reward weight depending on the hour
        hour = obs_dict['hour']
        if hour >= self.range_comfort_hours[0] and hour <= self.range_comfort_hours[1]:
            self.W_energy = self.default_energy_weight
        else:
            self.W_energy = 1.0

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation
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

        super(NormalizedLinearReward, self).__init__(
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

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using energy consumption and grades of temperature out of comfort range. Aplying normalization

        Args:
            energy (float): Negative absolute energy penalty value.
            comfort (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy and reward term for comfort.
        """
        # Update max energy and comfort
        self.max_energy_penalty = max(self.max_energy_penalty, energy_penalty)
        self.max_comfort_penalty = max(
            self.max_comfort_penalty, comfort_penalty)
        # Calculate normalization
        energy_norm = 0 if energy_penalty == 0 else energy_penalty / self.max_energy_penalty
        comfort_norm = 0 if comfort_penalty == 0 else comfort_penalty / self.max_comfort_penalty
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

        super(MultiZoneReward, self).__init__()

        # Name of the variables
        self.energy_names = energy_variables
        self.comfort_configuration = temperature_and_setpoints_conf
        self.comfort_threshold = comfort_threshold

        # Reward parameters
        self.comfort_ranges = {}

        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature

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
        energy_consumed, energy_values = self._get_energy_consumed(obs_dict)
        energy_penalty = self._get_energy_penalty(energy_values)

        # Comfort violation calculation
        total_temp_violation, temp_violations = self._get_temperature_violation(
            obs_dict)
        comfort_penalty = self._get_comfort_penalty(temp_violations)

        # Weighted sum of both terms
        reward, energy_term, comfort_term = self._get_reward(
            energy_penalty, comfort_penalty)

        reward_terms = {
            'energy_term': energy_term,
            'comfort_term': comfort_term,
            'reward_weight': self.W_energy,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': comfort_penalty,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': total_temp_violation,
            'comfort_threshold': self.comfort_threshold
        }

        # Add temp_violations per zone
        reward_terms.update(
            {
                f'{temperature_variable}_violation': temp_violation for temperature_variable,
                temp_violation in zip(
                    self.comfort_configuration.keys(),
                    temp_violations)})

        return reward, reward_terms

    def _get_energy_consumed(self, obs_dict: Dict[str,
                                                  Any]) -> Tuple[float,
                                                                 List[float]]:
        """Calculate the total energy consumed in the current observation.

        Args:
            obs_dict (Dict[str, Any]): Environment observation.

        Returns:
            Tuple[float, List[float]]: Total energy consumed (sum of variables) and List with energy consumed in each energy variable.
        """

        energy_values = [obs_dict[v] for v in self.energy_names]

        # The total energy is the sum of energies
        total_energy = sum(energy_values)

        return total_energy, energy_values

    def _get_temperature_violation(
            self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """
        # Calculate current comfort range for each zone
        self._get_comfort_ranges(obs_dict)
        total_temp_violation = 0.0
        temp_violations = []
        for temperature_variable, comfort_range in self.comfort_ranges.items():
            temp_violation = 0
            T = obs_dict[temperature_variable]
            if T < comfort_range[0] or T > comfort_range[1]:
                temp_violation = min(
                    abs(comfort_range[0] - T), abs(T - comfort_range[1]))
                total_temp_violation += temp_violation

            temp_violations.append(temp_violation)

        return total_temp_violation, temp_violations

    def _get_comfort_ranges(
            self, obs_dict: Dict[str, Any]):
        """Calculate the comfort range for each zone in the current observation.

        Returns:
            Dict[str, Tuple[float, float]]: Comfort range for each zone.
        """
        # Calculate current comfort range for each zone
        for temperature_variable, setpoint_variable in self.comfort_configuration.items():
            self.comfort_ranges[temperature_variable] = (
                obs_dict[setpoint_variable] - self.comfort_threshold,
                obs_dict[setpoint_variable] + self.comfort_threshold
            )

    def _get_energy_penalty(self, energy_values: List[float]) -> float:
        """Calculate the negative absolute energy penalty based on energy values

        Args:
            energy_values (List[float]): Energy values

        Returns:
            float: Negative absolute energy penalty value
        """
        energy_penalty = -sum(energy_values)
        return energy_penalty

    def _get_comfort_penalty(self, temp_violations: List[float]) -> float:
        """Calculate the negative absolute comfort penalty based on temperature violation values

        Args:
            temp_violations (List[float]): Temperature violation values

        Returns:
            float: Negative absolute comfort penalty value
        """
        comfort_penalty = -sum(temp_violations)
        return comfort_penalty

    def _get_reward(self, energy_penalty: float,
                    comfort_penalty: float) -> Tuple[float, float, float]:
        """It calculates reward value using the negative absolute comfort and energy penalty calculates previously.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            comfort_penalty (float): Negative absolute comfort penalty value.

        Returns:
            Tuple[float,float,float]: total reward calculated, reward term for energy, reward term for comfort.
        """
        energy_term = self.lambda_energy * self.W_energy * energy_penalty
        comfort_term = self.lambda_temp * \
            (1 - self.W_energy) * comfort_penalty
        reward = energy_term + comfort_term
        return reward, energy_term, comfort_term
