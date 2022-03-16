import os
import shutil
from glob import glob  # to find directories with patterns

import pkg_resources
import pytest
from opyplus import Epm, Idd, WeatherData

import sinergym.utils.rewards as R
from sinergym.envs.eplus_env import EplusEnv
from sinergym.simulators.eplus import EnergyPlus
from sinergym.utils.config import Config
from sinergym.utils.wrappers import (LoggerWrapper, MultiObsWrapper,
                                     NormalizeObservation)

############### ROOT DIRECTORY ###############


@pytest.fixture(scope='session')
def sinergym_path():
    return os.path.abspath(
        os.path.join(
            pkg_resources.resource_filename(
                'sinergym',
                ''),
            os.pardir))

############### SIMULATORS ###############


@pytest.fixture(scope='session')
def eplus_path():
    return os.environ['EPLUS_PATH']


@pytest.fixture(scope='session')
def bcvtb_path():
    return os.environ['BCVTB_PATH']


@pytest.fixture(scope='session')
def pkg_data_path():
    return pkg_resources.resource_filename('sinergym', 'data/')


@pytest.fixture(scope='session')
def idf_path(pkg_data_path):
    return os.path.join(pkg_data_path, 'buildings', '5ZoneAutoDXVAV.idf')


@pytest.fixture(scope='session')
def variable_path(pkg_data_path):
    return os.path.join(pkg_data_path, 'variables', 'variablesDXVAV.cfg')


@pytest.fixture(scope='session')
def space_path(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'variables',
        '5ZoneAutoDXVAV_spaces.cfg')


@pytest.fixture(scope='session')
def weather_path(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw')


@pytest.fixture(scope='session')
def ddy_path(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.ddy')


@pytest.fixture(scope='session')
def weather_path2(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw')


@pytest.fixture(scope='session')
def ddy_path2(pkg_data_path):
    return os.path.join(
        pkg_data_path,
        'weather',
        'USA_AZ_Davis-Monthan.AFB.722745_TMY3.ddy')


@pytest.fixture(scope='session')
def simulator(eplus_path, bcvtb_path, idf_path, variable_path, weather_path):
    env_name = 'TEST'
    return EnergyPlus(
        eplus_path,
        weather_path,
        bcvtb_path,
        variable_path,
        idf_path,
        env_name,
        act_repeat=1,
        max_ep_data_store_num=10)


@pytest.fixture(scope='function')
def config(idf_path, weather_path2):
    env_name = 'TESTCONFIG'
    max_ep_store = 10
    extra_config = {'timesteps_per_hour': 2}
    return Config(
        idf_path=idf_path,
        weather_path=weather_path2,
        env_name=env_name,
        max_ep_store=max_ep_store,
        extra_config=extra_config)

############### ENVIRONMENTS, WRAPPERS AND RULE BASED CONTROLLER AGENT####


@pytest.fixture(scope='module')
def env_demo(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=True,
        weather_variability=None)


@pytest.fixture(scope='function')
def env_demo_discrete(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=True,
        weather_variability=None)


@pytest.fixture(scope='function')
def env_demo_discrete(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=True,
        weather_variability=None)


@pytest.fixture(scope='function')
def env_demo_continuous(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    return EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=False,
        weather_variability=None)


@pytest.fixture(scope='function')
def env_wrapper_logger(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    env = EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=False,
        weather_variability=None)
    return LoggerWrapper(env=env, flag=True)


@pytest.fixture(scope='function')
def env_all_wrappers(idf_path, weather_path, variable_path, space_path):
    idf_file = idf_path.split('/')[-1]
    weather_file = weather_path.split('/')[-1]
    variables_file = variable_path.split('/')[-1]
    spaces_file = space_path.split('/')[-1]

    env = EplusEnv(
        env_name='TESTGYM',
        idf_file=idf_file,
        weather_file=weather_file,
        variables_file=variables_file,
        spaces_file=spaces_file,
        discrete_actions=False,
        weather_variability=None)
    env = NormalizeObservation(env=env)
    env = LoggerWrapper(env=env, flag=True)
    env = MultiObsWrapper(env=env, n=5)
    return env


############### COMMONS ###############


@pytest.fixture(scope='session')
def epm(idf_path, eplus_path):
    idd = Idd(os.path.join(eplus_path, 'Energy+.idd'))
    return Epm.from_idf(idf_path, idd_or_version=idd)


@pytest.fixture(scope='session')
def weather_data(weather_path):
    return WeatherData.from_epw(weather_path)

############### REWARDS ###############


@pytest.fixture(scope='session')
def simple_reward():
    return R.LinearReward(
        energy_weight=0.5,
        lambda_energy=1e-4,
        lambda_temperature=1.0
    )

############### WHEN TESTS HAVE FINISHED ###############


def pytest_sessionfinish(session, exitstatus):
    """ whole test run finishes. """
    # Deleting all temporal directories generated during tests
    directories = glob('Eplus-env-TEST*/')
    for directory in directories:
        shutil.rmtree(directory)

    # Deleting new random weather files once it has been checked
    files = glob('sinergym/data/weather/*Random*.epw')
    for file in files:
        os.remove(file)
