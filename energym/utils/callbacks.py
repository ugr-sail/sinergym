from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pprint


class LoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(LoggerCallback, self).__init__(verbose)
        self.ep_rewards = []
        self.ep_powers = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

    def _on_training_start(self):
        # deactivate environment logger
        pass

    def _on_step(self) -> bool:
        info = self.locals["infos"][-1]
        obs_dict = dict(zip(self.training_env.get_attr('variables')[
                        0]['observation'], self.locals['new_obs'][0]))
        obs_dict['day'] = info['day']
        obs_dict['month'] = info['month']
        obs_dict['hour'] = info['hour']
        for key in obs_dict:
            self.logger.record('observation/'+key, obs_dict[key])

        # Store episode data
        self.ep_rewards.append(self.locals['rewards'][-1])
        self.ep_powers.append(info['total_power'])
        if(info['comfort_penalty'] != 0):
            self.num_comfort_violation += 1
        self.ep_timesteps += 1

        # If episode ends
        if self.locals['dones'][-1]:

            self.cumulative_reward = np.sum(self.ep_rewards)
            self.mean_reward = np.mean(self.ep_rewards)
            self.mean_power = np.mean(self.ep_powers)
            self.comfort_violation = self.num_comfort_violation/self.ep_timesteps*100
            # reset episode info
            self.ep_rewards = []
            self.ep_powers = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        if(hasattr(self, 'cumulative_reward')):
            self.logger.record('episode/cumulative_reward',
                               self.cumulative_reward)
            self.logger.record('episode/mean_reward', self.mean_reward)
            self.logger.record('episode/mean_power', self.mean_power)
            self.logger.record('episode/comfort_violation(%)',
                               self.comfort_violation)

        return True
