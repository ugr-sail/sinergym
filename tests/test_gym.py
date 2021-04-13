import pytest
from energym.envs.eplus_env import EplusEnv
import numpy as np

@pytest.fixture(scope="session")
def env_discrete():
	idf_file = "5ZoneAutoDXVAV.idf"
	weather_file = "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw"
	discrete_actions = True
	weather_variability = None
	return EplusEnv(idf_file,weather_file,discrete_actions,weather_variability)

@pytest.fixture(scope="session")
def env_continuous():
	idf_file = "5ZoneAutoDXVAV.idf"
	weather_file = "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw"
	discrete_actions = False
	weather_variability = None
	return EplusEnv(idf_file,weather_file,discrete_actions,weather_variability)

@pytest.fixture(scope="session")
def env_stochastic():
	idf_file = "5ZoneAutoDXVAV.idf"
	weather_file = "USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw"
	discrete_actions = False
	weather_variability = (1,0.2) #(mean, std)
	return EplusEnv(idf_file,weather_file,discrete_actions,weather_variability)

@pytest.mark.parametrize(
	"env,obs",
	[
		(
			env_discrete,
			[0.00000000e+00, 9.50000000e+01, 4.10000000e+00, 2.90000000e+02, 0.00000000e+00, 0.00000000e+00, 2.10000000e+01, 2.50000000e+01, 2.09999897e+01, 0.00000000e+00, 3.95222748e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.00986453e+03, 1.00000000e+00, 1.00000000e+00, 0.00000000e+00]
		),
	]
)
def test_reset(env,obs):
	out = env.reset()

	for i in range(len(obs)):
		assert round(out[i],5) == round(obs[i],5)


@pytest.mark.parametrize(
	"env,action,obs,reward,done,info",
	[
		(
			env_discrete,
			3,  #action discrete
			[1.80000000e+00, 9.52500000e+01, 4.10000000e+00, 2.65000000e+02,
		       0.00000000e+00, 0.00000000e+00, 1.80000000e+01, 2.70000000e+01,
		       1.89682012e+01, 1.89971546e+01, 4.46953922e+01, 7.49999995e-01,
		       3.17571430e+01, 0.00000000e+00, 2.09999897e+01, 3.13368921e+03,
		       1.00000000e+00, 1.00000000e+00, 0.00000000e+00],
       		-0.6725838607374635,
       		False,
       		{'timestep': 900.0, 'day': 1, 'month': 1, 'hour': 0, 'total_power': 3133.689213604861, 'total_power_no_units': -0.3133689213604861, 'comfort_penalty': -1.0317988001144407, 'temperature': 18.96820119988556, 'out_temperature': 1.8}
		),
	]
)
def test_step(env,action,obs,reward,done,info):

	out = env.step(action)
	
	for i in range(len(obs)):
		assert round(obs[i],5) == round(out[0][i],5) 
	assert reward == out[1]
	assert done == out[2]
	assert info == out[3]
	
