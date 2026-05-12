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
env = gym.make('Eplus-radiant_digital_twin_rl_cooling-madrid-continuous-stochastic-v1')
env = DatetimeWrapper(env)
env = NormalizeObservation(env)
env = ExtremeFlowControlWrapper(env)
env = NormalizeAction(env)
env = RadiantLoggerWrapper(env)
env = CSVLogger(env)
env = ReduceObservationWrapper(
    env,
    obs_reduction=[
        'radiant_hvac_outlet_temperature_f0_living-kitchen',
        'radiant_hvac_outlet_temperature_f0_bathroom-lobby',
        'radiant_hvac_outlet_temperature_f1_bedroom1',
        'radiant_hvac_outlet_temperature_f1_bedroom2',
        'radiant_hvac_outlet_temperature_f1_bedroom3',
        'radiant_hvac_outlet_temperature_f1_bathroom-corridor',
        'radiant_hvac_outlet_temperature_f1_bathroom-dressing',
        'radiant_hvac_inlet_temperature_f0_living-kitchen',
        'radiant_hvac_inlet_temperature_f0_bathroom-lobby',
        'radiant_hvac_inlet_temperature_f1_bedroom1',
        'radiant_hvac_inlet_temperature_f1_bedroom2',
        'radiant_hvac_inlet_temperature_f1_bedroom3',
        'radiant_hvac_inlet_temperature_f1_bathroom-corridor',
        'radiant_hvac_inlet_temperature_f1_bathroom-dressing',
        'water_temperature',
        'flow_rate_f0_living-kitchen',
        'flow_rate_f0_bathroom-lobby',
        'flow_rate_f1_bedroom1',
        'flow_rate_f1_bedroom2',
        'flow_rate_f1_bedroom3',
        'flow_rate_f1_bathroom-corridor',
        'flow_rate_f1_bathroom-dressing',
        'cooling_hp_load_side_heat_transfer_rate',
        'cooling_hp_load_side_mass_flow_rate',
        'crf',
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
