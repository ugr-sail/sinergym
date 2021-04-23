import pytest
from random import randint

def test_reset(env_demo):
	obs=env_demo.reset()
	assert len(obs)==19
	assert env_demo.simulator._episode_existed

def test_step(env_demo):
	action=randint(0, 9)
	obs, reward, done, info = env_demo.step(action)

	assert len(obs)==19
	assert type(reward)!=None
	assert not done
	assert list(info.keys())==['timestep', 'day', 'month', 'hour', 'total_power', 'total_power_no_units', 'comfort_penalty', 'temperature', 'out_temperature']
	assert info['timestep']==env_demo.simulator._eplus_run_stepsize *1

	action=randint(0, 9)
	obs, reward, done, info = env_demo.step(action)

	assert len(obs)==19
	assert type(reward)!=None
	assert not done
	assert list(info.keys())==['timestep', 'day', 'month', 'hour', 'total_power', 'total_power_no_units', 'comfort_penalty', 'temperature', 'out_temperature']
	assert info['timestep']==env_demo.simulator._eplus_run_stepsize *2

	#This is a discrete environment, action cannot be a tuple with continuous values
	with pytest.raises(KeyError):
		obs, reward, done, info = env_demo.step((21.2,23.5))

def test_close(env_demo):
	env_demo.close()
	assert not env_demo.simulator._episode_existed
	assert env_demo.simulator._conn==None