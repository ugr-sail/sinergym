import os

import pytest

# ---------------------------------------------------------------------------- #
#                                 Main methods                                 #
# ---------------------------------------------------------------------------- #


def test_simulator(simulator_5zone, pkg_data_path):
    # Checks status before simulator start
    assert not simulator_5zone.warmup_complete
    assert simulator_5zone.var_handlers is None
    assert simulator_5zone.meter_handlers is None
    assert simulator_5zone.actuator_handlers is None
    assert simulator_5zone.available_data is None
    assert simulator_5zone.energyplus_thread is None
    assert simulator_5zone.energyplus_state is None
    assert not simulator_5zone.initialized_handlers
    assert not simulator_5zone.system_ready
    assert not simulator_5zone.simulation_complete
    assert simulator_5zone._building_path is None
    assert simulator_5zone._weather_path is None
    assert simulator_5zone._output_path is None

    # simulation start
    simulator_5zone.start(
        building_path=os.path.join(
            pkg_data_path,
            'buildings',
            '5ZoneAutoDXVAV.epJSON'),
        weather_path=os.path.join(
            pkg_data_path,
            'weather',
            'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'),
        output_path='./Eplus-TESTSIMULATOR/')

    # Checks status after simulation start
    assert simulator_5zone.energyplus_state is not None
    assert simulator_5zone.energyplus_thread is not None
    assert simulator_5zone._building_path is not None
    assert simulator_5zone._weather_path is not None
    assert simulator_5zone._output_path is not None

    # Checks warmup process
    if not simulator_5zone.warmup_complete:
        value = None
        value = simulator_5zone.warmup_queue.get()
        assert value is not None
    else:
        raise AssertionError
    assert simulator_5zone.warmup_complete

    # Until first observation received, system is not initialized
    assert simulator_5zone.var_handlers is None
    assert simulator_5zone.meter_handlers is None
    assert simulator_5zone.actuator_handlers is None
    assert simulator_5zone.available_data is None
    assert not simulator_5zone.initialized_handlers
    assert not simulator_5zone.system_ready
    assert not simulator_5zone.simulation_complete

    # first observation
    obs = None
    info = None
    obs = simulator_5zone.obs_queue.get()
    info = simulator_5zone.info_queue.get()
    assert len(obs) > 0 and obs is not None
    assert len(info) > 0 and info is not None

    # Now system should be initialized, since first observation has been
    # received
    assert simulator_5zone.var_handlers is not None
    assert simulator_5zone.meter_handlers is not None
    assert simulator_5zone.actuator_handlers is not None
    assert simulator_5zone.available_data is not None
    assert simulator_5zone.initialized_handlers
    assert simulator_5zone.system_ready
    assert not simulator_5zone.simulation_complete

    # first action
    setpoints = [15.0, 22.5]
    assert simulator_5zone.act_queue.empty()
    simulator_5zone.act_queue.put(setpoints, timeout=2)
    assert not simulator_5zone.act_queue.empty()
    setpoints = list(map(lambda x: x + 1, setpoints))

    # Check 4 more interactions
    for i in range(4):
        # Observation and info
        obs = None
        info = None
        obs = simulator_5zone.obs_queue.get()
        info = simulator_5zone.info_queue.get()
        assert len(obs) > 0 and obs is not None
        assert len(info) > 0 and info is not None
        # Actions
        simulator_5zone.act_queue.put(setpoints, timeout=2)
        assert not simulator_5zone.act_queue.empty()
        setpoints = list(map(lambda x: x + 1, setpoints))

    # Check early stop
    assert not simulator_5zone.simulation_complete
    simulator_5zone.stop()
    assert simulator_5zone.obs_queue.empty()
    assert simulator_5zone.info_queue.empty()
    assert simulator_5zone.act_queue.empty()
    assert simulator_5zone.warmup_queue.empty()
    assert simulator_5zone.energyplus_thread is None
    assert not simulator_5zone.warmup_complete
    assert not simulator_5zone.initialized_handlers
    assert not simulator_5zone.system_ready
    assert not simulator_5zone.simulation_complete


def test_make_eplus_args(simulator_5zone):
    simulator_5zone._building_path = 'expected_building'
    simulator_5zone._weather_path = 'expected_weather'
    simulator_5zone._output_path = 'expected_output'

    eplus_args = simulator_5zone.make_eplus_args()
    assert eplus_args == [
        '-w',
        'expected_weather',
        '-d',
        'expected_output',
        'expected_building']
