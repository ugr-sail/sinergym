from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import gym
import os
from energym.utils.wrappers import NormalizeObservation

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class LoggerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(LoggerCallback, self).__init__(verbose)
        self.ep_rewards = []
        self.ep_powers = []
        self.ep_term_comfort = []
        self.ep_term_energy = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

    def _on_training_start(self):
        self.training_env.env_method('deactivate_logger')

    def _on_step(self) -> bool:
        # OBSERVATION
        variables = self.training_env.get_attr('variables')[0]['observation']

        # log normalized and original values
        if self.training_env.env_is_wrapped(wrapper_class=NormalizeObservation)[0]:
            obs_normalized = self.locals['new_obs'][0]
            obs = self.training_env.env_method('get_unwrapped_obs')[0]
            for i, variable in enumerate(variables):
                self.logger.record(
                    'normalized_observation/'+variable, obs_normalized[i])
                self.logger.record(
                    'observation/'+variable, obs[i])
        # Only original values
        else:
            obs = self.locals['new_obs'][0]
            for i, variable in enumerate(variables):
                self.logger.record(
                    'observation/'+variable, obs[i])

        # ACTION
        variables = self.training_env.get_attr('variables')[0]['action']
        action = self.locals['actions'][0]
        for i, variable in enumerate(variables):
            self.logger.record(
                'action/'+variable, action[i])

        # Store episode data
        info = self.locals['infos'][0]
        self.ep_rewards.append(self.locals['rewards'][0])
        self.ep_powers.append(info['total_power'])
        self.ep_term_comfort.append(info['comfort_penalty'])
        self.ep_term_energy.append(info['total_power_no_units'])
        if(info['comfort_penalty'] != 0):
            self.num_comfort_violation += 1
        self.ep_timesteps += 1

        # If episode ends
        if self.locals['dones'][0]:

            self.cumulative_reward = np.sum(self.ep_rewards)
            self.mean_reward = np.mean(self.ep_rewards)
            self.mean_power = np.mean(self.ep_powers)
            self.mean_term_comfort = np.mean(self.ep_term_comfort)
            self.mean_term_power = np.mean(self.ep_term_energy)
            self.comfort_violation = self.num_comfort_violation/self.ep_timesteps*100
            # reset episode info
            self.ep_rewards = []
            self.ep_powers = []
            self.ep_term_comfort = []
            self.ep_term_energy = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        # In the first episode, logger doesn't have these attributes
        if(hasattr(self, 'cumulative_reward')):
            self.logger.record('episode/cumulative_reward',
                               self.cumulative_reward)
            self.logger.record('episode/mean_reward', self.mean_reward)
            self.logger.record('episode/mean_power', self.mean_power)
            self.logger.record('episode/comfort_violation(%)',
                               self.comfort_violation)
            self.logger.record('episode/mean_comfort_penalty',
                               self.mean_term_comfort)
            self.logger.record('episode/mean_power_penalty',
                               self.mean_term_power)

        return True

    def on_training_end(self):
        self.training_env.env_method('activate_logger')


class LoggerEvalCallback(EvalCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(LoggerEvalCallback, self).__init__(eval_env=eval_env, callback_on_new_best=callback_on_new_best, n_eval_episodes=n_eval_episodes,
                                                 eval_freq=eval_freq, log_path=log_path, best_model_save_path=best_model_save_path, deterministic=deterministic, render=render, verbose=verbose, warn=warn)
        self.evaluations_power_consumption = []
        self.evaluations_comfort_violation = []
        self.evaluations_comfort_penalty = []
        self.evaluations_power_penalty = []

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_powers, episode_comfort_violations, episode_comfort_penalties, episode_power_penalties = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_power_consumption.append(episode_powers)
                self.evaluations_comfort_violation.append(
                    episode_comfort_violations)
                self.evaluations_comfort_penalty.append(
                    episode_comfort_penalties)
                self.evaluations_power_penalty.append(episode_power_penalties)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    ep_powers=self.evaluations_power_consumption,
                    ep_comfort_violations=self.evaluations_comfort_violation,
                    episode_comfort_penalties=self.evaluations_comfort_penalty,
                    episode_power_penalties=self.evaluations_power_penalty,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(
                episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(
                episode_lengths), np.std(episode_lengths)
            mean_ep_power, std_ep_power = np.mean(
                episode_powers), np.std(episode_powers)
            mean_ep_comfort_violation, mean_std_comfort_violation = np.mean(
                episode_comfort_violations), np.std(episode_comfort_violations)
            self.last_mean_reward = mean_reward
            mean_ep_comfort_penalty = np.mean(episode_comfort_penalties)
            mean_ep_power_penalty = np.mean(episode_power_penalties)

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_power_consumption", mean_ep_power)
            self.logger.record("eval/mean_comfort_violation(%)",
                               mean_ep_comfort_violation)
            self.logger.record("eval/mean_power_penalty",
                               mean_ep_power_penalty)
            self.logger.record("eval/mean_comfort_penalty",
                               mean_ep_comfort_penalty)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(
                        self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,


):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.
    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
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
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"
        is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]
    else:
        is_monitor_wrapped = is_wrapped(env, Monitor)

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards, episode_lengths, episode_powers, episode_comfort_violations, episode_comfort_penalties, episode_power_penalties = [], [], [], [], [], []
    not_reseted = True
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(env, VecEnv) or not_reseted:
            obs = env.reset()
            not_reseted = False
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        episode_steps_comfort_violation = 0
        episode_power = 0.0
        episode_comfort_penalty = 0.0
        episode_power_penalty = 0.0
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

        if is_monitor_wrapped:
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards.append(info["episode"]["r"])
                episode_lengths.append(info["episode"]["l"])
        else:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_powers.append(episode_power)
            episode_comfort_violations.append(
                episode_steps_comfort_violation/episode_length*100)
            episode_comfort_penalties.append(episode_comfort_penalty)
            episode_power_penalties.append(episode_power_penalty)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    # mean_power = np.mean(episode_powers)
    # std_power = np.std(episode_powers)
    # mean_comfort_violation= np.mean(episode_comfort_violations)
    # std_comfort_violation= np.std(episode_comfort_violations)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_powers, episode_comfort_violations, episode_comfort_penalties, episode_power_penalties
    return mean_reward, std_reward
