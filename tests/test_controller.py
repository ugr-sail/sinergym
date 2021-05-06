import gym
import numpy as np


def test_rule_based_controller(rule_controller_agent, env_demo):
    obs = env_demo.reset()
    for i in range(3):
        action = rule_controller_agent.act(obs)
        assert type(action) == tuple
        for value in action:
            assert value is not None
        obs, reward, done, info = env_demo.step(action)

    env_demo.close()
