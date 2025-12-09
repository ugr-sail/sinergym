import logging

import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    DatetimeWrapper,
    ExtremeFlowControlWrapper,
    HeatPumpEnergyWrapper,
    NormalizeAction,
    NormalizeObservation,
    RadiantLoggerWrapper,
    ReduceObservationWrapper,
)

# Logger
terminal_logger = TerminalLogger()
logger = terminal_logger.getLogger(name='MAIN', level=logging.INFO)

# Creating environment and applying wrappers for normalization and logging
env = gym.make('Eplus-radiant_case2_heating-stockholm-continuous-stochastic-v1')
# env = RoundActionWrapper(env)
# env = HeatPumpEnergyWrapper(env)
env = DatetimeWrapper(env)
env = NormalizeObservation(env)
env = ExtremeFlowControlWrapper(env)
env = NormalizeAction(env)
env = RadiantLoggerWrapper(env)
env = CSVLogger(env)
env = ReduceObservationWrapper(
    env,
    obs_reduction=[
        'radiant_hvac_outlet_temperature_living',
        'radiant_hvac_outlet_temperature_kitchen',
        'radiant_hvac_outlet_temperature_bed1',
        'radiant_hvac_outlet_temperature_bed2',
        'radiant_hvac_outlet_temperature_bed3',
        'water_temperature',
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
        'plr_current',
    ],
)

# Execute 3 episodes
for i in range(3):

    # Reset the environment to start a new episode
    obs, info = env.reset()

    truncated = terminated = False

    while not (terminated or truncated):

        # Random action selection
        a = env.action_space.sample()

        # Perform action and receive env information
        obs, reward, terminated, truncated, info = env.step(a)

    logger.info(f'Episode {env.get_wrapper_attr("episode")} finished.')

env.close()
