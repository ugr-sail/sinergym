"""Env checker functionality, adapted from Stable-Baselines: https://github.com/DLR-RM/stable-baselines3"""
import warnings
from typing import Union

import gym
import numpy as np
from gym import spaces


def _is_numpy_array_space(space: spaces.Space) -> bool:
    """
    Returns False if provided space is not representable as a single numpy array
    (e.g. Dict and Tuple spaces return False)
    """
    return not isinstance(space, (spaces.Dict, spaces.Tuple))


def _check_unsupported_spaces(
        env: gym.Env,
        observation_space: spaces.Space,
        action_space: spaces.Space) -> None:
    """Emit warnings when the observation space or action space used is not supported by Stable-Baselines."""

    if isinstance(observation_space, spaces.Dict):
        nested_dict = False
        for space in observation_space.spaces.values():
            if isinstance(space, spaces.Dict):
                nested_dict = True
        if nested_dict:
            warnings.warn(
                "Nested observation spaces are not supported by Stable Baselines3 "
                "(Dict spaces inside Dict space). "
                "You should flatten it to have only one level of keys."
                "For example, `dict(space1=dict(space2=Box(), space3=Box()), spaces4=Discrete())` "
                "is not supported but `dict(space2=Box(), spaces3=Box(), spaces4=Discrete())` is.")

    if isinstance(observation_space, spaces.Tuple):
        warnings.warn(
            "The observation space is a Tuple,"
            "this is currently not supported by Stable Baselines3. "
            "However, you can convert it to a Dict observation space "
            "(cf. https://github.com/openai/gym/blob/master/gym/spaces/dict.py). "
            "which is supported by SB3.")

    if not _is_numpy_array_space(action_space):
        warnings.warn(
            "The action space is not based off a numpy array. Typically this means it's either a Dict or Tuple space. "
            "This type of action space is currently not supported by Stable Baselines 3. You should try to flatten the "
            "action using a wrapper.")


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
        assert not isinstance(
            obs, tuple
        ), f"The observation returned by the `{method_name}()` method should be a single value, not a tuple"

    if isinstance(observation_space, spaces.Discrete):
        assert isinstance(
            obs, int), f"The observation returned by `{method_name}()` method must be an int"
    elif _is_numpy_array_space(observation_space):
        assert isinstance(
            obs, np.ndarray), f"The observation returned by `{method_name}()` method must be a numpy array"

    assert observation_space.contains(
        obs
    ), f"The observation returned by the `{method_name}()` method does not match the given observation space"


def _check_returned_values(
        env: gym.Env,
        observation_space: spaces.Space,
        action_space: spaces.Space) -> None:
    """
    Check the returned values by the env when calling `.reset()` or `.step()` methods.
    """
    # because env inherits from gym.Env, we assume that `reset()` and `step()`
    # methods exists
    obs = env.reset()

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
        data) == 4, "The `step()` method must return four values: obs, reward, done, info"

    # Unpack
    obs, reward, done, info = data

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
    assert isinstance(done, bool), "The `done` signal must be a boolean"
    assert isinstance(
        info, dict), "The `info` returned by `step()` must be a python dictionary"


def _check_spaces(env: gym.Env) -> None:
    """
    Check that the observation and action spaces are defined
    and inherit from gym.spaces.Space.
    """
    # Helper to link to the code, because gym has no proper documentation
    gym_spaces = " cf https://github.com/openai/gym/blob/master/gym/spaces/"

    assert hasattr(
        env, "observation_space"), "You must specify an observation space (cf gym.spaces)" + gym_spaces
    assert hasattr(
        env, "action_space"), "You must specify an action space (cf gym.spaces)" + gym_spaces

    assert isinstance(env.observation_space,
                      spaces.Space), "The observation space must inherit from gym.spaces" + gym_spaces
    assert isinstance(
        env.action_space, spaces.Space), "The action space must inherit from gym.spaces" + gym_spaces


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


def check_env(
        env: gym.Env,
        warn: bool = True,
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

    # ============= Check the spaces (observation and action) ================
    _check_spaces(env)

    # Define aliases for convenience
    observation_space = env.observation_space
    action_space = env.action_space

    # Warn the user if needed.
    # A warning means that the environment may run but not work properly with
    # Stable Baselines algorithms
    if warn:
        _check_unsupported_spaces(env, observation_space, action_space)

        # Check for the action space, it may lead to hard-to-debug issues
        if isinstance(action_space, spaces.Box) and (
            np.any(np.abs(action_space.low) != np.abs(action_space.high))
            or np.any(action_space.low != -1)
            or np.any(action_space.high != 1)
        ):
            warnings.warn(
                "We recommend you to use a symmetric and normalized Box action space (range=[-1, 1]) "
                "cf https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html")

        if isinstance(action_space, spaces.Box):
            assert np.all(
                np.isfinite(np.array([action_space.low, action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        if isinstance(
                action_space,
                spaces.Box) and action_space.dtype != np.dtype(
                np.float32):
            warnings.warn(
                f"Your action space has dtype {action_space.dtype}, we recommend using np.float32 to avoid cast errors."
            )

    # ============ Check the returned values ===============
    _check_returned_values(env, observation_space, action_space)

    # ==== Check the render method and the declared render modes ====
    if not skip_render_check:
        _check_render(env, warn=warn)  # pragma: no cover
