"""Env checker functionality, adapted from Stable-Baselines: https://github.com/DLR-RM/stable-baselines3"""
import warnings
from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _check_obs(obs: Union[tuple,
                          dict,
                          np.ndarray,
                          int],
               observation_space: spaces.Space,
               method_name: str) -> None:
    """
    Check that the observation returned by the environment
    correspond to the declared one.
    """
    if not isinstance(observation_space, spaces.Tuple):
        assert not isinstance(obs, tuple), f"The observation returned by the `{
            method_name}()` method should be a single value, not a tuple"

    if isinstance(observation_space, spaces.Discrete):
        assert isinstance(obs, int), f"The observation returned by `{
            method_name}()` method must be an int"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(obs, np.ndarray), f"The observation returned by `{
            method_name}()` method must be a numpy array"

    assert observation_space.contains(obs), f"The observation returned by the `{
        method_name}()` method does not match the given observation space"


def _check_returned_values(
        env: gym.Env,
        observation_space: spaces.Space,
        action_space: spaces.Space) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gym.Env, we assume that `reset()` and `step()`
    # methods exists
    obs, info = env.reset()

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict), "The observation returned by `reset()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "reset")
            except AssertionError as e:
                raise AssertionError(
                    f"Error while checking key={key}: " + str(e))
    else:
        _check_obs(obs, observation_space, "reset")

    # Sample a random action
    action = action_space.sample()
    data = env.step(action)

    assert len(
        data) == 5, "The `step()` method must return four values: obs, reward, terminated, truncated, info"

    # Unpack
    obs, reward, terminated, truncated, info = data

    if isinstance(observation_space, spaces.Dict):
        assert isinstance(
            obs, dict), "The observation returned by `step()` must be a dictionary"
        for key in observation_space.spaces.keys():
            try:
                _check_obs(obs[key], observation_space.spaces[key], "step")
            except AssertionError as e:
                raise AssertionError(
                    f"Error while checking key={key}: " + str(e))

    else:
        _check_obs(obs, observation_space, "step")

    # We also allow int because the reward will be cast to float
    assert isinstance(reward, (float, int)
                      ), "The reward returned by `step()` must be a float"
    assert isinstance(
        terminated, bool), "The `terminated` signal must be a boolean"
    assert isinstance(
        truncated, bool), "The `truncated` signal must be a boolean"
    assert isinstance(
        info, dict), "The `info` returned by `step()` must be a python dictionary"


def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined
    and inherit from gymnasium.spaces.Space.
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert env.has_wrapper_attr(
        'observation_space'), 'You must specify an observation space (cf gym.spaces)' + gym_spaces
    assert env.has_wrapper_attr(
        'action_space'), 'You must specify an action space (cf gym.spaces)' + gym_spaces

    assert isinstance(env.observation_space,
                      spaces.Space), "The observation space must inherit from gymnasium.spaces" + gym_spaces
    assert isinstance(
        env.action_space, spaces.Space), "The action space must inherit from gymnasium.spaces" + gym_spaces


# Check render cannot be covered by CI
def _check_render(env: gym.Env, warn: bool = True, headless: bool = False) -> None:  # pragma: no cover
    """
    Check the declared render modes and the `render()`/`close()`
    method of the environment.

    :param env: The environment to check
    :param warn: Whether to output additional warnings
    :param headless: Whether to disable render modes
        that require a graphical interface. False by default.
    """
    render_modes = env.metadata.get("render.modes")
    if render_modes is None:
        if warn:
            warnings.warn(
                "No render modes was declared in the environment "
                " (env.metadata['render.modes'] is None or not defined), "
                "you may have trouble when calling `.render()`"
            )

    else:
        # Don't check render mode that require a
        # graphical interface (useful for CI)
        if headless and "human" in render_modes:
            render_modes.remove("human")
        # Check all declared render modes
        for render_mode in render_modes:
            env.render(mode=render_mode)
        env.close()


# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def check_env(
        env: gym.Env,
        skip_render_check: bool = True) -> None:
    """
    Check that an environment follows Gym API.
    This is particularly useful when using a custom environment.
    Please take a look at https://github.com/openai/gym/blob/master/gym/core.py
    for more information about the API.

    This env_checker has been adapted from Stable-Baselines: https://github.com/DLR-RM/stable-baselines3
    It also optionally check that the environment is compatible with Stable-Baselines.

    :param env: The Gym environment that will be checked
    :param warn: Whether to output additional warnings
        mainly related to the interaction with Stable Baselines
    :param skip_render_check: Whether to skip the checks for the render method.
        True by default (useful for the CI)
    """
    assert isinstance(
        env, gym.Env
    ), "Your environment must inherit from the gym.Env class cf https://github.com/openai/gym/blob/master/gym/core.py"

    # ---------------------------------------------------------------------------- #
    #                   Check the spaces (observation and action)                  #
    # ---------------------------------------------------------------------------- #
    _check_spaces(env)
    # ---------------------- Define aliases for convenience ---------------------- #
    observation_space = env.observation_space
    action_space = env.action_space

    # ---------------------------------------------------------------------------- #
    #                           Check the returned values                          #
    # ---------------------------------------------------------------------------- #
    _check_returned_values(env, observation_space, action_space)

    # ---------------------------------------------------------------------------- #
    #             Check the render method and the declared render modes            #
    # ---------------------------------------------------------------------------- #
    if not skip_render_check:
        _check_render(env)  # pragma: no cover
