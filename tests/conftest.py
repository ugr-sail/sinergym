import pytest
from energym.simulators.eplus_old import EnergyPlus
import os
import pkg_resources
from opyplus import Epm, WeatherData
import energym.utils.rewards as R
from glob import glob #to find directories with patterns
import shutil

############### ROOT DIRECTORY ###############

@pytest.fixture(scope="session")
def energym_path():
	return os.path.abspath(os.path.join(pkg_resources.resource_filename('energym', ''), os.pardir))

############### SIMULATOR ###############

@pytest.fixture(scope="session")
def eplus_path():
	return os.environ["EPLUS_PATH"]

@pytest.fixture(scope="session")
def bcvtb_path():
	return os.environ["BCVTB_PATH"]

@pytest.fixture(scope="session")
def pkg_data_path():
	return pkg_resources.resource_filename('energym', 'data/')

@pytest.fixture(scope="session")
def idf_path(pkg_data_path):
	 return os.path.join(pkg_data_path, 'buildings', "5ZoneAutoDXVAV.idf")

@pytest.fixture(scope="session")
def variable_path(pkg_data_path):
	return os.path.join(pkg_data_path, 'variables', "variables.cfg")

@pytest.fixture(scope="session")
def weather_path(pkg_data_path):
	return os.path.join(pkg_data_path, 'weather', "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw")

@pytest.fixture(scope="session")
def simulator(eplus_path, bcvtb_path,idf_path,variable_path,weather_path):
	env_name="TEST"
	return EnergyPlus(eplus_path, weather_path, bcvtb_path, variable_path, idf_path, env_name,act_repeat=1,max_ep_data_store_num = 10)

############### COMMONS ###############

@pytest.fixture(scope="session")
def epm(idf_path):
	return Epm.from_idf(idf_path)

@pytest.fixture(scope="session")
def weather_data(weather_path):
	return WeatherData.from_epw(weather_path)

############### REWARD ###############

@pytest.fixture(scope="session")
def simple_reward():
	return R.SimpleReward(
		range_comfort_winter = (20.0, 23.5),
		range_comfort_summer = (23.0, 26.0),
		energy_weight        =          0.5,
		lambda_energy        =         1e-4,
		lambda_temperature   =          1.0
		)

############### WHEN TESTS HAVE FINISHED ###############

def pytest_sessionfinish(session, exitstatus):
	""" whole test run finishes. """
	# Deleting all temporal directories generated during tests
	directories=glob("Eplus-env-TEST-res*/")
	for directory in directories:
		shutil.rmtree(directory)