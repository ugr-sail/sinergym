import numpy as np
import pytest


def test_random_controller(random_controller):
    env = random_controller.env
    env.reset()

    for _ in range(3):
        action = random_controller.act()
        assert action is not None
    env.close()


def test_controller_5Zone(zone5_controller):
    env = zone5_controller.env
    obs, info = env.reset()

    for _ in range(3):
        action = zone5_controller.act(obs)
        assert isinstance(action, np.ndarray)
        assert len(action) == 2
        for value in action:
            assert value is not None
        obs, _, _, _, info = env.step(action)

        assert (info['action'] == action).all()

    env.close()


def test_controller_datacenter(datacenter_controller):
    action = datacenter_controller.act()
    assert (action == datacenter_controller.range_datacenter).all()


def test_incremental_controller_datacenter(datacenter_incremental_controller):
    env = datacenter_incremental_controller.env
    obs, info = env.reset()

    for _ in range(3):
        action = datacenter_incremental_controller.act(obs)
        assert isinstance(action, np.ndarray)
        for value in action:
            assert value is not None
        obs, _, _, _, info = env.step(action)

        assert (info['action'] == action).all()

    env.close()
