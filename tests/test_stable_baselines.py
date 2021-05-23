import argparse

import gym
import energym
import numpy as np
import pprint
import pytest

import stable_baselines3

TIMESTEPS = 1000


@pytest.mark.parametrize(
    'env_name',
    [
        (
            'env_demo_discrete'
        ),
        (
            'env_demo_continuous'
        ),
    ]
)
def test_stable_PPO(env_name, request):
    env = request.getfixturevalue(env_name)
    model = stable_baselines3.PPO('MlpPolicy', env, verbose=1,
                                  learning_rate=.0003,
                                  n_steps=2048,
                                  batch_size=64,
                                  n_epochs=10,
                                  gamma=.99,
                                  gae_lambda=.95,
                                  clip_range=.2,
                                  ent_coef=0,
                                  vf_coef=.5,
                                  max_grad_norm=.5)

    model.learn(total_timesteps=TIMESTEPS)

    # Check model state
    assert model.action_space == env.action_space
    assert model.env.action_space == env.action_space

    assert type(
        model.policy) == stable_baselines3.common.policies.ActorCriticPolicy

    # Check model works

    obs = env.reset()
    a, _ = model.predict(obs)
    obs, reward, done, info = env.step(a)

    assert reward is not None and reward < 0
    assert a is not None
    assert type(done) == bool
    assert info['timestep'] == 1

    env.close()


@pytest.mark.parametrize(
    'env_name',
    [
        (
            'env_demo_discrete'
        ),
        (
            'env_demo_continuous'
        ),
    ]
)
def test_stable_A2C(env_name, request):
    env = request.getfixturevalue(env_name)
    model = stable_baselines3.A2C('MlpPolicy', env, verbose=1,
                                  learning_rate=.0007,
                                  n_steps=5,
                                  gamma=.99,
                                  gae_lambda=1.0,
                                  ent_coef=0,
                                  vf_coef=.5,
                                  max_grad_norm=.5,
                                  rms_prop_eps=1e-05)

    model.learn(total_timesteps=TIMESTEPS)

    # Check model state
    assert model.action_space == env.action_space
    assert model.env.action_space == env.action_space

    assert type(
        model.policy) == stable_baselines3.common.policies.ActorCriticPolicy

    # Check model works

    obs = env.reset()
    a, _ = model.predict(obs)
    obs, reward, done, info = env.step(a)

    assert reward is not None and reward < 0
    assert a is not None
    assert type(done) == bool
    assert info['timestep'] == 1

    env.close()


@pytest.mark.parametrize(
    'env_name',
    [
        (
            'env_demo_discrete'
        ),
        (
            'env_demo_continuous'
        ),
    ]
)
def test_stable_DQN(env_name, request):
    env = request.getfixturevalue(env_name)
    # DQN must fail in continuous environments
    if env_name == 'env_demo_continuous':
        with pytest.raises(AssertionError):
            model = stable_baselines3.DQN('MlpPolicy', env, verbose=1,
                                          learning_rate=.0001,
                                          buffer_size=1000000,
                                          learning_starts=50000,
                                          batch_size=32,
                                          tau=1.0,
                                          gamma=.99,
                                          train_freq=4,
                                          gradient_steps=1,
                                          target_update_interval=10000,
                                          exploration_fraction=.1,
                                          exploration_initial_eps=1.0,
                                          exploration_final_eps=.05,
                                          max_grad_norm=10)
        pass

    else:
        model = stable_baselines3.DQN('MlpPolicy', env, verbose=1,
                                      learning_rate=.0001,
                                      buffer_size=1000000,
                                      learning_starts=50000,
                                      batch_size=32,
                                      tau=1.0,
                                      gamma=.99,
                                      train_freq=4,
                                      gradient_steps=1,
                                      target_update_interval=10000,
                                      exploration_fraction=.1,
                                      exploration_initial_eps=1.0,
                                      exploration_final_eps=.05,
                                      max_grad_norm=10)

        model.learn(total_timesteps=TIMESTEPS)

        # Check model state
        assert model.action_space == env.action_space
        assert model.env.action_space == env.action_space

        assert type(
            model.policy) == stable_baselines3.dqn.policies.DQNPolicy

        # Check model works

        obs = env.reset()
        a, _ = model.predict(obs)
        obs, reward, done, info = env.step(a)

        assert reward is not None and reward < 0
        assert a is not None
        assert type(done) == bool
        assert info['timestep'] == 1

        env.close()
