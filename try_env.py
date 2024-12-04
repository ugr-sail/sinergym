import logging

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
    ExtremeFlowControlWrapper,
    HeatPumpEnergyWrapper,
    ReduceObservationWrapper,
    CSVLogger)

# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(
    name='MAIN',
    level=logging.INFO
)

# Creating environment and applying wrappers for normalization and logging
env = gym.make(
    'Eplus-radiant_case1_heating-stockholm-continuous-stochastic-v1')
# env = RoundActionWrapper(env)
# env = HeatPumpEnergyWrapper(env)
env = NormalizeObservation(env)
# env = ExtremeFlowControlWrapper(env)
env = NormalizeAction(env)
env = LoggerWrapper(env)
env = CSVLogger(env)
env = ReduceObservationWrapper(
    env,
    obs_reduction=[
        'radiant_hvac_outlet_temperature_living',
        'radiant_hvac_outlet_temperature_kitchen',
        'radiant_hvac_outlet_temperature_bed1',
        'radiant_hvac_outlet_temperature_bed2',
        'radiant_hvac_outlet_temperature_bed3',
        'flow_rate_living',
        'flow_rate_kitchen',
        'flow_rate_bed1',
        'flow_rate_bed2',
        'flow_rate_bed3',
        'heat_source_load_side_heat_transfer_rate',
        'heat_source_load_side_mass_flow_rate',
        'crf',
        'heat_cap_mod',
        'cop_plr_mod',
        'cop_temp_mod',
        'plr_current'])

# Execute 3 episodes
for i in range(3):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    rewards = []
    truncated = terminated = False
    current_month = 0

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

        rewards.append(reward)

        # Display results every simulated month
        if info['month'] != current_month:
            current_month = info['month']
            logger.info('Reward: {}'.format(sum(rewards)))
            logger.info('Info: {}'.format(info))

    logger.info('Episode {} - Mean reward: {} - Cumulative Reward: {}'.format(i,
                                                                              np.mean(rewards), sum(rewards)))
env.close()
