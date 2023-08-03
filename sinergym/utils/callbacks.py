"""Custom Callbacks for stable baselines 3 algorithms."""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv,
                                              sync_envs_normalization)

from sinergym.utils.evaluation import evaluate_policy
from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation


class LoggerCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, sinergym_logger=False, verbose=0):
        """Custom callback for plotting additional values in tensorboard.
        Args:
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(LoggerCallback, self).__init__(verbose)

        self.sinergym_logger = sinergym_logger

        self.ep_rewards = []
        self.ep_powers = []
        self.ep_term_comfort = []
        self.ep_term_energy = []
        self.num_comfort_violation = 0
        self.ep_timesteps = 0

    def _on_training_start(self):
        # sinergym logger
        if is_wrapped(self.training_env, LoggerWrapper):
            if self.sinergym_logger:
                self.training_env.env_method('activate_logger')
            else:
                self.training_env.env_method('deactivate_logger')

        # record method depending on the type of algorithm

        if 'OnPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record
        elif 'OffPolicyAlgorithm' in self.globals.keys():
            self.record = self.logger.record_mean
        else:
            raise KeyError

    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]

        # OBSERVATION
        variables = self.training_env.get_attr('observation_variables')[0]
        # log normalized and original values
        if self.training_env.env_is_wrapped(
                wrapper_class=NormalizeObservation)[0]:
            obs_normalized = self.locals['new_obs'][-1]
            obs = self.training_env.env_method('get_unwrapped_obs')[-1]
            for i, variable in enumerate(variables):
                self.record(
                    'normalized_observation/' + variable, obs_normalized[i])
                self.record(
                    'observation/' + variable, obs[i])
        # Only original values
        else:
            obs = self.locals['new_obs'][-1]
            for i, variable in enumerate(variables):
                self.record(
                    'observation/' + variable, obs[i])

        # ACTION
        variables = self.training_env.get_attr('action_variables')[0]
        action = None
        # sinergym action received inner its own setpoints range
        action_ = info['action']
        try:
            # network output clipped with gym action space
            action = self.locals['clipped_actions'][-1]
        except KeyError:
            try:
                action = self.locals['action'][-1]
            except KeyError:
                try:
                    action = self.locals['actions'][-1]
                except KeyError:
                    raise KeyError(
                        'Algorithm action key in locals dict unknown.')

        if self.training_env.get_attr('flag_discrete')[0]:
            action = self.training_env.get_attr('action_mapping')[0][action]
        for i, variable in enumerate(variables):
            if action is not None:
                self.record(
                    'action/' + variable, action[i])

            self.record(
                'action_simulation/' + variable, action_[i])

        # Store episode data
        try:
            self.ep_rewards.append(self.locals['rewards'][-1])
        except KeyError:
            try:
                self.ep_rewards.append(self.locals['reward'][-1])
            except KeyError:
                print('Algorithm reward key in locals dict unknown')

        self.ep_powers.append(info['total_energy'])
        self.ep_term_comfort.append(info['reward_comfort'])
        self.ep_term_energy.append(info['reward_energy'])
        if (info['reward_comfort'] != 0):
            self.num_comfort_violation += 1
        self.ep_timesteps += 1

        # If episode ends, store summary of episode and reset
        try:
            done = self.locals['dones'][-1]
        except KeyError:
            try:
                done = self.locals['done'][-1]
            except KeyError:
                print('Algorithm done key in locals dict unknown')
        if done:
            # store last episode metrics
            self.episode_metrics = {}
            self.episode_metrics['ep_length'] = self.ep_timesteps
            self.episode_metrics['cumulative_reward'] = np.sum(
                self.ep_rewards)
            self.episode_metrics['mean_reward'] = np.mean(self.ep_rewards)
            self.episode_metrics['mean_power'] = np.mean(self.ep_powers)
            self.episode_metrics['cumulative_power'] = np.sum(self.ep_powers)
            self.episode_metrics['mean_comfort_penalty'] = np.mean(
                self.ep_term_comfort)
            self.episode_metrics['cumulative_comfort_penalty'] = np.sum(
                self.ep_term_comfort)
            self.episode_metrics['mean_power_penalty'] = np.mean(
                self.ep_term_energy)
            self.episode_metrics['cumulative_power_penalty'] = np.sum(
                self.ep_term_energy)
            try:
                self.episode_metrics['comfort_violation_time(%)'] = self.num_comfort_violation / \
                    self.ep_timesteps * 100
            except ZeroDivisionError:
                self.episode_metrics['comfort_violation_time(%)'] = np.nan

            # reset episode info
            self.ep_rewards = []
            self.ep_powers = []
            self.ep_term_comfort = []
            self.ep_term_energy = []
            self.ep_timesteps = 0
            self.num_comfort_violation = 0

        # During first episode, as it not finished, it shouldn't be recording
        if hasattr(self, 'episode_metrics'):
            for key, metric in self.episode_metrics.items():
                self.logger.record(
                    'episode/' + key, metric)

        return True

    def on_training_end(self):
        if is_wrapped(self.training_env, LoggerWrapper):
            self.training_env.env_method('activate_logger')


class LoggerEvalCallback(EventCallback):
    """Callback for evaluating an agent.
        :param eval_env: The environment used for initialization
        :param callback_on_new_best: Callback to trigger when there is a new best model according to the mean reward
        :param n_eval_episodes: The number of episodes to test the agent
        :param eval_freq: Evaluate the agent every eval freq call of the callback.
        :param log_path: Path to a folder where the evaluations (evaluations.npz) will be saved. It will be updated at each evaluation.
        :param best_model_save_path: Path to a folder where the best model according to performance on the eval env will be saved.
        :param deterministic: Whether the evaluation should use a stochastic or deterministic actions.
        :param render: Whether to render or not the environment during evaluation
        :param verbose:
        :param warn: Passed to evaluate policy (warns if eval env has not been wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        # if not isinstance(eval_env, VecEnv):
        #     eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.log_metrics = {
            'timesteps': [],
            'mean_rewards': [],
            'cumulative_rewards': [],
            'ep_lengths': [],
            'ep_powers': [],
            'ep_comfort_violations': [],
            'episodes_comfort_penalties': [],
            'episodes_power_penalties': [],
        }
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

        # Custom metrics for Sinergym
        self.evaluations_power_consumption = []
        self.evaluations_comfort_violation = []
        self.evaluations_comfort_penalty = []
        self.evaluations_power_penalty = []
        self.evaluation_metrics = {}

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type"
                f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(
            self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["terminated"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above.") from e

            # Reset success rate buffer
            self._is_success_buffer = []

            # We close training env before to start the evaluation
            self.training_env.close()

            # episodes_rewards, episodes_lengths, episodes_powers, episodes_comfort_violations, episodes_comfort_penalties, episodes_power_penalties
            episodes_data = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            # We close evaluation env and starts training env again
            self.eval_env.close()
            self.training_env.reset()

            if self.log_path is not None:
                self.log_metrics['timesteps'].append(self.num_timesteps)
                self.log_metrics['cumulative_rewards'].append(
                    episodes_data['episodes_cumulative_rewards'])
                self.log_metrics['mean_rewards'].append(
                    episodes_data['episodes_mean_rewards'])
                self.log_metrics['ep_lengths'].append(
                    episodes_data['episodes_lengths'])
                self.log_metrics['ep_powers'].append(
                    episodes_data['episodes_powers'])
                self.log_metrics['ep_comfort_violations'].append(
                    episodes_data['episodes_comfort_violations'])
                self.log_metrics['episodes_comfort_penalties'].append(
                    episodes_data['episodes_comfort_penalties'])
                self.log_metrics['episodes_power_penalties'].append(
                    episodes_data['episodes_power_penalties'])

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                kwargs.update(self.log_metrics)

                np.savez(
                    self.log_path,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(
                episodes_data['episodes_mean_rewards']), np.std(
                episodes_data['episodes_mean_rewards'])
            mean_cumulative_reward, std_cumulative_reward = np.mean(
                episodes_data['episodes_cumulative_rewards']), np.std(
                episodes_data['episodes_cumulative_rewards'])
            mean_ep_length, std_ep_length = np.mean(
                episodes_data['episodes_lengths']), np.std(
                episodes_data['episodes_lengths'])
            self.last_reward = mean_cumulative_reward

            self.evaluation_metrics['mean_reward'] = mean_reward
            self.evaluation_metrics['std_reward'] = std_reward
            self.evaluation_metrics['mean_cumulative_reward'] = mean_cumulative_reward
            self.evaluation_metrics['std_cumulative_reward'] = std_cumulative_reward
            self.evaluation_metrics['mean_ep_length'] = mean_ep_length
            self.evaluation_metrics['mean_power_consumption'] = np.mean(
                episodes_data['episodes_powers'])
            self.evaluation_metrics['comfort_violation(%)'] = np.mean(
                episodes_data['episodes_comfort_violations'])
            self.evaluation_metrics['comfort_penalty'] = np.mean(
                episodes_data['episodes_comfort_penalties'])
            self.evaluation_metrics['power_penalty'] = np.mean(
                episodes_data['episodes_power_penalties'])

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_cumulative_reward:.2f} +/- {std_cumulative_reward:.2f}")
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger (our custom metrics)
            for key, metric in self.evaluation_metrics.items():
                self.logger.record('eval/' + key, metric)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct
            # timestep
            self.logger.record(
                "time/total_timesteps",
                self.num_timesteps,
                exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # HERE IS THE CONDITION TO DETERMINE WHEN A MODEL IS BETTER THAN
            # OTHER
            if mean_cumulative_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(
                            self.best_model_save_path,
                            "model.zip"))
                self.best_mean_reward = mean_cumulative_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)
