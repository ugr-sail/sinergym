"""Custom policy evaluations for Evaluation Callbacks."""

import warnings
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(model: "base_class.BaseAlgorithm",
                    env: Union[gym.Env,
                               VecEnv],
                    n_eval_episodes: int = 5,
                    deterministic: bool = True,
                    render: bool = False,
                    callback: Optional[Callable[[Dict[str,
                                                      Any],
                                                 Dict[str,
                                                      Any]],
                                                None]] = None,
                    ) -> Any:
    """Runs policy for n_eval_episodes episodes and returns average reward. This is made to work only with one env.
        .. note:: If environment has not been wrapped with Monitor wrapper, reward and
        episode lengths are counted as it appears with env.step calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with Monitor
        wrapper before anything else.
        :param model: The RL agent you want to evaluate.
        :param env: The gym environment. In the case of a VecEnv this must contain only one environment.
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param deterministic: Whether to use deterministic or stochastic actions
        :param render: Whether to render the environment or not
        :param callback: callback function to do additional checks, called after each step. Gets locals() and globals() passed as parameters.
        :param reward_threshold: Minimum expected reward per episode, this will raise an error if the performance is not met
        :param return_episode_rewards: If True, a list of rewards and episode lengths per episode will be returned instead of the mean.
        :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when return_episode_rewards is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).

    """
    result = {
        'episodes_rewards': [],
        'episodes_lengths': [],
        'episodes_powers': [],
        'episodes_comfort_violations': [],
        'episodes_comfort_penalties': [],
        'episodes_power_penalties': []
    }
    episodes_executed = 0
    not_reseted = True
    while episodes_executed < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            # obs = list(map(
            #     lambda obs_dict: np.array(list(obs_dict.values()), dtype=np.float32),
            #     env.get_attr('obs_dict')))
            obs = env.reset()
            not_reseted = False
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_steps_comfort_violation = 0
        episode_power = 0.0
        episode_comfort_penalty = 0.0
        episode_power_penalty = 0.0
        # ---------------------------------------------------------------------------- #
        #                     Running episode and accumulate values                    #
        # ---------------------------------------------------------------------------- #
        while not done:
            action, state = model.predict(
                obs, state=state, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_power += info[0]['total_power']
            episode_power_penalty += info[0]['total_power_no_units']
            episode_comfort_penalty += info[0]['comfort_penalty']
            if info[0]['comfort_penalty'] != 0:
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
        result['episodes_rewards'].append(episode_reward)
        result['episodes_lengths'].append(episode_length)
        result['episodes_powers'].append(episode_power)
        try:
            result['episodes_comfort_violations'].append(
                episode_steps_comfort_violation / episode_length * 100)
        except ZeroDivisionError:
            result['episodes_comfort_violations'].append(np.nan)
        result['episodes_comfort_penalties'].append(episode_comfort_penalty)
        result['episodes_power_penalties'].append(episode_power_penalty)

    return result
