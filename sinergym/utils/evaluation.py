"""Custom policy evaluations for Evaluation Callbacks."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None
) -> Dict[str, list]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward and other Sinergym metrics.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    result = {
        'episodes_length': [],
        'episodes_cumulative_reward': [],
        'episodes_mean_reward': [],
        'episodes_cumulative_power': [],
        'episodes_mean_power': [],
        'episodes_cumulative_comfort_penalty': [],
        'episodes_mean_comfort_penalty': [],
        'episodes_cumulative_energy_penalty': [],
        'episodes_mean_energy_penalty': [],
        'episodes_comfort_violation': [],
        'episodes_cumulative_temperature_violation': [],
        'episodes_mean_temperature_violation': []
    }
    episodes_executed = 0
    while episodes_executed < n_eval_episodes:
        obs, info = env.reset()
        terminated, state = False, None
        episode_cumulative_reward = 0.0
        episode_length = 0
        episode_steps_comfort_violation = 0
        episode_cumulative_power = 0.0
        episode_cumulative_comfort_penalty = 0.0
        episode_cumulative_energy_penalty = 0.0
        episode_cumulative_temperature_violation = 0.0
        # ---------------------------------------------------------------------------- #
        #                     Running episode and accumulate values                    #
        # ---------------------------------------------------------------------------- #
        while not terminated:
            action, state = model.predict(
                obs, state=state, deterministic=deterministic)
            obs, reward, terminated, _, info = env.step(action)
            episode_cumulative_reward += reward
            episode_cumulative_power += info['abs_energy']
            episode_cumulative_energy_penalty += info['energy_term']
            episode_cumulative_comfort_penalty += info['comfort_term']
            episode_cumulative_temperature_violation += info['abs_comfort']
            if info['comfort_term'] < 0:
                episode_steps_comfort_violation += 1
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episodes_executed += 1
        # ---------------------------------------------------------------------------- #
        #                     Storing accumulated values in result                     #
        # ---------------------------------------------------------------------------- #
        result['episodes_length'].append(episode_length)
        result['episodes_cumulative_reward'].append(episode_cumulative_reward)
        result['episodes_mean_reward'].append(
            episode_cumulative_reward / episode_length)
        result['episodes_cumulative_power'].append(episode_cumulative_power)
        result['episodes_mean_power'].append(
            episode_cumulative_power / episode_length)
        result['episodes_cumulative_comfort_penalty'].append(
            episode_cumulative_comfort_penalty)
        result['episodes_mean_comfort_penalty'].append(
            episode_cumulative_comfort_penalty / episode_length)
        result['episodes_cumulative_energy_penalty'].append(
            episode_cumulative_energy_penalty)
        result['episodes_mean_energy_penalty'].append(
            episode_cumulative_energy_penalty / episode_length)
        try:
            result['episodes_comfort_violation'].append(
                episode_steps_comfort_violation / episode_length * 100)
        except ZeroDivisionError:
            result['episodes_comfort_violation'].append(np.nan)
        result['episodes_cumulative_temperature_violation'].append(
            episode_cumulative_temperature_violation)
        result['episodes_mean_temperature_violation'].append(
            episode_cumulative_temperature_violation / episode_length)

    return result
