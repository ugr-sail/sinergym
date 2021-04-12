import pytest
from energym.envs.eplus_env import EplusEnv
import numpy as np

@pytest.fixture
def env():
	idf_file = "5ZoneAutoDXVAV.idf"
	weather_file = "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw"
	discrete_actions = True
	weather_variability = None
	return EplusEnv(idf_file,weather_file,discrete_actions,weather_variability)


def test_reset(env):
	salida = env.reset()
	expecteds=[0.00000000e+00, 9.50000000e+01, 4.10000000e+00, 2.90000000e+02, 0.00000000e+00, 0.00000000e+00, 2.10000000e+01, 2.50000000e+01, 2.09999897e+01, 0.00000000e+00, 3.95222748e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.00986453e+03, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00]

	for i in range(len(expecteds)):
		assert round(salida[i],5) == round(expecteds[i],5)
