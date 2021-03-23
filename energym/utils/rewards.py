"""Script for implementing several types of rewards."""


class SimpleReward():
    """"""
    def __init__(self,
        target_temperature: float,
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_temperature: float = 1.0
    ):
    """"""

    self.target_temp = target_temperature
    self.W_energy = energy_weight
    self.lambda_energy = lambda_energy
    self.lambda_temp = lambda_temperature

    def get_reward(self, power, temperature):
        """"""
        reward_energy = - self.lambda_energy * power
        reward_comfort = - self.lambda_temp * abs(temperature - self.target_temp)
        reward = self.W_energy * reward_energy + (1.0 - self.W_energy) * reward_comfort
        terms = {'reward_energy': reward_energy, 'reward_comfort': reward_comfort}
        return reward, terms