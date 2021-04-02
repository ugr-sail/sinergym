"""Script for implementing several types of rewards."""


from datetime import datetime


class SimpleReward():
    """
    Simple reward considering absolute difference to temperature comfort.

    R = - W * lambda_E * power - (1 - W) * lambda_T * (max(T - T_low, 0) + max(T_up - T, 0))
    """
    def __init__(self,
        range_comfort_winter: tuple = (20.0, 23.5),
        range_comfort_summer: tuple = (23.0, 26.0),
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
        """
        Parameters
        ----------
        range_comfort_winter
            Temperature comfort range for winter.
        range_comfort_summer
            Temperature comfort range for summer.
        energy_weight
            Weight given to the energy term.
        lambda_energy
            Constant for removing dimensions from power (1/W)
        lambda_temperature
            Constant for removing dimensions from temperature (1/C)
        """
        # Variables
        self.range_comfort_winter = range_comfort_winter
        self.range_comfort_summer = range_comfort_summer
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_temperature
        # Periods
        self.summer_start_date = datetime(2021, 6, 1)
        self.summer_final_date = datetime(2021, 9, 30)

    def calculate(self, power, temperature, month, day):
        """"""
        # Energy term
        reward_energy = - self.lambda_energy * power
        # Comfort term
        current_dt = datetime(2021, month, day)
        range_T = self.range_comfort_summer if current_dt >= self.summer_start_date and current_dt <= self.summer_final_date else self.range_comfort_winter
        delta_T = 0.0 if temperature >= range_T[0] and temperature <= range_T[1] else min(abs(range_T[0] - temperature), abs(temperature - range_T[1]))
        reward_comfort = - self.lambda_temp * delta_T
        # Weighted sum of both terms
        reward = self.W_energy * reward_energy + (1.0 - self.W_energy) * reward_comfort
        terms = {'reward_energy': reward_energy, 'reward_comfort': reward_comfort}
        return reward, terms