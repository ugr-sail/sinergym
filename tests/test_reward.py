import pytest


def test_reward(env_linear_reward):
    _ = env_linear_reward.reset()
    a = env_linear_reward.action_space.sample()
    _, R, _, info = env_linear_reward.step(a)
    env_linear_reward.close()
    assert R <= 0.0
    assert env_linear_reward.reward_fn.W_energy == 0.5
    assert info.get('comfort_penalty') <= 0.0


def test_reward_kwargs(env_linear_reward_args):
    _ = env_linear_reward_args.reset()
    a = env_linear_reward_args.action_space.sample()
    _, R, _, info = env_linear_reward_args.step(a)
    env_linear_reward_args.close()
    assert R <= 0.0
    assert env_linear_reward_args.reward_fn.W_energy == 0.2
    assert env_linear_reward_args.reward_fn.range_comfort_summer == (18.0, 20.0)
    assert info.get('comfort_penalty') <= 0.0

def test_custom_reward(env_custom_reward):
    _ = env_custom_reward.reset()
    a = env_custom_reward.action_space.sample()
    _, R, _, _ = env_custom_reward.step(a)
    env_custom_reward.close()
    assert R == -1.0