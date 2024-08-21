import gymnasium as gym
import numpy as np

import sinergym
from sinergym.utils.wrappers import (
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
    ExtremeFlowControlWrapper,
    HeatPumpEnergyWrapper,
    ReduceObservationWrapper,
    CSVLogger)

# Creating environment and applying wrappers for normalization and logging
env = gym.make(
    'Eplus-radiant_free_heating-stockholm-continuous-stochastic-v1')
# env = RoundActionWrapper(env)
env = HeatPumpEnergyWrapper(env)
env = NormalizeObservation(env)
env = ExtremeFlowControlWrapper(env)
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
        'radiant_hvac_inlet_temperature_living',
        'radiant_hvac_inlet_temperature_kitchen',
        'radiant_hvac_inlet_temperature_bed1',
        'radiant_hvac_inlet_temperature_bed2',
        'radiant_hvac_inlet_temperature_bed3',
        'surface_internal_user_specified_location_temperature_living',
        'surface_internal_user_specified_location_temperature_kitchen',
        'surface_internal_user_specified_location_temperature_bed1',
        'surface_internal_user_specified_location_temperature_bed2',
        'surface_internal_user_specified_location_temperature_bed3',
        'people_occupant_living',
        'people_occupant_kitchen',
        'people_occupant_bed1',
        'people_occupant_bed2',
        'people_occupant_bed3',
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
        'heat_cap',
        'plr_current'])

# Execute interactions during 3 episodes
for i in range(1):
    # Reset the environment to start a new episode
    obs, info = env.reset()
    truncated = terminated = False

    while not (terminated or truncated):
        # Random action control
        a = env.action_space.sample()
        # Read observation and reward
        obs, reward, terminated, truncated, info = env.step(a)


# Close the environment
env.close()
