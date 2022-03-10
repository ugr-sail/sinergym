import os
import signal
import subprocess
import threading
import xml.etree.ElementTree as ET

import pkg_resources
import pytest

from sinergym.simulators.eplus import EnergyPlus


def test_reset(simulator):
    assert not simulator._episode_existed

    output = simulator.reset()

    # Checking output
    # Fist element is a tuple with simulation date + time_elapsed
    assert isinstance(output[0], tuple)
    assert len(output[0]) == 4
    # Last element in first tuple must be time_elapsed 0
    assert output[0][-1] == 0
    assert isinstance(output[1], list)
    assert len(output[1]) == 16
    assert output[2] == False

    # Checking simulator state
    assert simulator._eplus_run_stepsize == 900
    assert simulator._eplus_one_epi_len == 31536000
    assert simulator._curSimTim == 0
    assert simulator._env_working_dir_parent.split(
        '/')[-1] == 'Eplus-env-' + simulator._env_name + '-res1'
    assert simulator._epi_num == 0
    assert simulator._episode_existed
    path_list = simulator._eplus_working_dir.split('/')
    assert path_list[-2] + '/' + path_list[-1] == 'Eplus-env-' + \
        simulator._env_name + '-res1/Eplus-env-sub_run' + str(simulator._epi_num + 1)

    # Checking energyplus subprocess
    assert isinstance(simulator._eplus_process, subprocess.Popen)
    assert '/usr/local/EnergyPlus' in simulator._eplus_process.args

    # Checking next directory for the next simulation episode is created
    # successfully
    simulator.reset()
    assert simulator._epi_num == 1
    path_list = simulator._eplus_working_dir.split('/')
    assert path_list[-2] + '/' + path_list[-1] == 'Eplus-env-' + \
        simulator._env_name + '-res1/Eplus-env-sub_run' + str(simulator._epi_num + 1)


def test_step(simulator):
    output = simulator.step(action=[20.0, 24.0])

    # Checking output
    # Fist element is a tuple with simulation date + time_elapsed
    assert isinstance(output[0], tuple)
    assert len(output[0]) == 4
    # Last element in first tuple must be time_elapsed > 0 (since we have a
    # step executed)
    assert output[0][-1] > 0
    assert isinstance(output[1], list)
    assert len(output[1]) == 16

    # Check simulation advance with step
    assert simulator._curSimTim > 0
    # Simulation time elapsed at each timestep is defined in the simulator
    assert simulator._curSimTim == simulator._eplus_run_stepsize
    assert simulator._curSimTim == output[0][-1]
    assert isinstance(output[1], list)
    assert len(output[1]) == 16
    assert (simulator._curSimTim >= simulator._eplus_one_epi_len) == output[2]

    assert simulator._last_action == [20.0, 24.0]

    # Another step
    output = simulator.step(action=[20.0, 24.0])

    # Check simulation advance with step
    assert simulator._curSimTim == output[0][-1]
    assert simulator._curSimTim > 0
    # Simulation time elapsed at each timestep is defined in the simulator
    assert simulator._curSimTim == simulator._eplus_run_stepsize * 2


def test_episode_transition_with_steps(simulator):

    is_terminal = False

    while(not is_terminal):
        output = simulator.step(action=[20.0, 24.0])[2]
        is_terminal = output

    # When we raise a terminal state it is only because our Current Simulation
    # Time is greater or equeal to episode length
    assert simulator._curSimTim >= simulator._eplus_one_epi_len

    # If we try to do one step more, it shouldn't change environment
    # One step more...
    with pytest.raises(RuntimeError):
        output = simulator.step(action=[23.0, 25.0])


def test_get_file_name(simulator, idf_path):
    expected = '5ZoneAutoDXVAV.idf'
    assert simulator._get_file_name(idf_path) == expected


# def test_rm_past_history_dir(cur_eplus_working_dir, dir_sig):


def test_create_socket_cfg(simulator, sinergym_path):

    # creating a socket.cfg example in tests/socket.cfg
    tests_path = sinergym_path + '/tests'
    simulator._create_socket_cfg(simulator._host, simulator._port, tests_path)
    # Check its content
    with open(tests_path + '/' + 'socket.cfg', 'r+') as socket_file:
        tree = ET.parse(socket_file)
        root = tree.getroot()
        assert root.tag == 'BCVTB-client'
        assert root[0].tag == 'ipc'
        socket_attrs = root[0][0].attrib
        socket_tag = root[0][0].tag
        assert socket_tag == 'socket'
        assert socket_attrs['hostname'] == simulator._host
        assert socket_attrs['port'] == str(simulator._port)

    # delete socket.cfg created during simulator_tests
    os.remove(tests_path + '/socket.cfg')


def test_create_eplus(simulator, eplus_path, weather_path, idf_path):
    eplus_working_dir = simulator._env_working_dir_parent
    out_path = eplus_working_dir + '/output'
    eplus_process = simulator._create_eplus(
        eplus_path, weather_path, idf_path, out_path, eplus_working_dir)
    assert 'ERROR' not in str(eplus_process.stdout.read())


def test_get_is_eplus_running(simulator):
    # Like our simulator has an episode active, we should see True value
    assert simulator.get_is_eplus_running()

# def test_get_is_eplus_running(self):


def test_end_episode(simulator):
    # In this point, we have a simulation running second episode which is
    # terminated
    assert simulator._episode_existed
    simulator.end_episode()
    # Now, let's check simulator
    assert not simulator._episode_existed
    assert simulator._conn is None
    assert not simulator.get_is_eplus_running()


def test_end_env(simulator):
    # start new episode because we finished it in last test
    simulator.reset()
    assert simulator._episode_existed
    assert '[closed]' not in str(simulator._socket)
    # This end episode and close simulator socket
    simulator.end_env()
    assert not simulator._episode_existed
    assert '[closed]' in str(simulator._socket)


# def test_run_eplus_outputProcessing(self):


def test_assembleMsg(simulator):
    header = 0
    action = simulator._last_action
    curSimTim = simulator._curSimTim
    Dblist = [num for num in range(16)]
    msg = simulator._assembleMsg(
        header, 0, len(action), 0, 0, curSimTim, Dblist)
    assert msg == '0 0 2 0 0 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00 2.000000000000000e+00 3.000000000000000e+00 4.000000000000000e+00 5.000000000000000e+00 6.000000000000000e+00 7.000000000000000e+00 8.000000000000000e+00 9.000000000000000e+00 1.000000000000000e+01 1.100000000000000e+01 1.200000000000000e+01 1.300000000000000e+01 1.400000000000000e+01 1.500000000000000e+01 \n'


def test_disassembleMsg(simulator):
    msg = '0 0 2 0 0 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00 2.000000000000000e+00 3.000000000000000e+00 4.000000000000000e+00 5.000000000000000e+00 6.000000000000000e+00 7.000000000000000e+00 8.000000000000000e+00 9.000000000000000e+00 1.000000000000000e+01 1.100000000000000e+01 1.200000000000000e+01 1.300000000000000e+01 1.400000000000000e+01 1.500000000000000e+01 \n'
    (version, flag, nDb, nIn, nBl, curSimTim,
     Dblist) = simulator._disassembleMsg(msg)
    assert version == 0
    assert flag == 0
    assert nDb == 2
    assert nIn == 0
    assert nBl == 0
    assert curSimTim == 0
    assert [num for num in range(16)] == Dblist
