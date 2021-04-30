import pytest
from random import randint
import gym
import energym
import os

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

def test_all_environments():
	
	envs_id = [env_spec.id for env_spec in gym.envs.registry.all() if env_spec.id.startswith("Eplus")]
	for env_id in envs_id:
		#Create env with TEST name 
		env = gym.make(env_id)

		initial_obs = env.reset()
		assert len(initial_obs)>0

		a = env.action_space.sample()
		assert a is not None

		obs, reward, done, info = env.step(a)
		assert len(initial_obs)==len(obs)
		assert reward!=0
		assert done is not None
		assert type(info)==dict and len(info)>0

		#Rename directory with name TEST for future remove
		os.rename(env.simulator._env_working_dir_parent,"Eplus-env-TEST"+env.simulator._env_working_dir_parent.split("/")[-1])

		# env.close()
		# assert not env.simulator._episode_existed
		# assert env.simulator._conn==None