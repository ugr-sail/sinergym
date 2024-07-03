"""Custom Callbacks for stable baselines 3 algorithms."""

import os
import warnings
from copy import deepcopy
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
    """Custom callback for plotting additional values in Stable Baselines 3 algorithms.
        :param ep_rewards: Here will be stored all rewards during episode.
        :param ep_powers: Here will be stored all consumption data during episode.
        :param ep_term_comfort: Here will be stored all comfort terms (reward component) during episode.
        :param ep_term_energy: Here will be stored all energy terms (reward component) during episode.
        :param num_comfort_violation: Number of timesteps in which comfort has been violated.
        :param ep_timesteps: Each timestep during an episode, this value increment 1.
    """

    def __init__(self, dump_frequency=100, sinergym_logger=False, verbose=0):
        """Custom callback for plotting additional values in Stable Baselines 3 algorithms.
        Args:
            dump_frequency (int): This is the timestep frequency in which all data recorded is dumped, ignoring algorithm log interval. Defaults to 100.
            sinergym_logger (boolean): Indicate if CSVLogger inner Sinergym will be activated or not.
        """
        super(LoggerCallback, self).__init__(verbose)

        # New attributes
        self.sinergym_logger = sinergym_logger
        self.dump_frequency = dump_frequency

        # Episode lists of metrics in each timestep.
        self.episode_logs = {
            'rewards': [],
            'reward_energy_terms': [],
            'reward_comfort_terms': [],
            'abs_energy_penalties': [],
            'abs_comfort_penalties': [],
            'total_power_demands': [],
            'total_temperature_violations': [],
            'total_timesteps': 0,
            'comfort_violation_timesteps': 0
        }

    def _on_training_start(self):
        # sinergym logger
        if is_wrapped(self.training_env, LoggerWrapper):
            if self.sinergym_logger:
                self.training_env.env_method('activate_logger')
            else:
                self.training_env.env_method('deactivate_logger')
        # Global training step and episode count
        self.timestep = 1
        self.episode_num = 1

    def _on_step(self) -> bool:

        # New timestep
        self.timestep += 1
        info = self.locals['infos'][-1]

        # OBSERVATION LOG
        observation_variables = self.training_env.get_attr(
            'observation_variables')[-1]
        # log normalized and original values
        if self.training_env.env_is_wrapped(
                wrapper_class=NormalizeObservation)[0]:
            obs_normalized = self.locals['new_obs'][-1]
            obs = self.training_env.get_attr('unwrapped_observation')[-1]
            for i, variable in enumerate(observation_variables):
                self.logger.record(
                    'normalized_observation/' + variable, obs_normalized[i])
                self.logger.record(
                    'observation/' + variable, obs[i])
        # Only original values
        else:
            obs = self.locals['new_obs'][-1]
            for i, variable in enumerate(observation_variables):
                self.logger.record(
                    'observation/' + variable, obs[i])

        # ACTION LOG
        action_variables = self.training_env.get_attr('action_variables')[
            -1]
        action = None
        # sinergym action received inner its own setpoints range
        action_ = info['action']
        # Search network action returned, depends on the algorithm
        if 'clipped_actions' in self.locals.keys():
            action = self.locals['clipped_actions'][-1]
        elif 'action' in self.locals.keys():
            action = self.locals['action'][-1]
        elif 'actions' in self.locals.keys():
            action = self.locals['actions'][-1]
        else:
            raise KeyError('Algorithm action key in locals dict unknown.')

        if self.training_env.get_attr('is_discrete')[-1]:
            self.logger.record(
                'action_network/index', action)
            for i, variable in enumerate(action_variables):
                self.logger.record(
                    'action_simulation/' + variable, action_[i])
        else:
            for i, variable in enumerate(action_variables):
                if action is not None:
                    self.logger.record(
                        'action_network/' + variable, action[i])
                self.logger.record(
                    'action_simulation/' + variable, action_[i])

        # EPISODE LOG
        # Store episode data (key depends on algorithm)
        if 'rewards' in self.locals.keys():
            self.episode_logs['rewards'].append(self.locals['rewards'][-1])
        elif 'reward' in self.locals.keys():
            self.episode_logs['rewards'].append(self.locals['reward'][-1])
        else:
            raise KeyError('Algorithm reward key in locals dict unknown.')
        # Rest of the info store
        self.episode_logs['reward_energy_terms'].append(info['energy_term'])
        self.episode_logs['reward_comfort_terms'].append(info['comfort_term'])
        self.episode_logs['abs_energy_penalties'].append(
            info['abs_energy_penalty'])
        self.episode_logs['abs_comfort_penalties'].append(
            info['abs_comfort_penalty'])
        self.episode_logs['total_power_demands'].append(
            info['total_power_demand'])
        self.episode_logs['total_temperature_violations'].append(
            info['total_temperature_violation'])
        if (info['comfort_term'] < 0):
            self.episode_logs['comfort_violation_timesteps'] += 1
        self.episode_logs['total_timesteps'] += 1

        # If episode ends, store summary of episode and reset
        if 'dones' in self.locals.keys():
            done = self.locals['dones'][-1]
        elif 'done' in self.locals.keys():
            done = self.locals['done'][-1]
        else:
            raise KeyError('Algorithm done key in locals dict unknown.')

        if done:
            # store last episode metrics
            # 'comfort_violation_time (%)',
            # 'length (timesteps)',
            # 'time_elapsed (hours)'
            self.episode_metrics = {}
            self.episode_metrics['episode_num'] = self.episode_num
            self.episode_metrics['episode_length'] = self.episode_logs['total_timesteps']
            self.episode_metrics['cumulative_reward'] = np.sum(
                self.episode_logs['rewards'])
            self.episode_metrics['mean_reward'] = np.mean(
                self.episode_logs['rewards'])
            self.episode_metrics['cumulative_reward_energy_term'] = np.sum(
                self.episode_logs['reward_energy_terms'])
            self.episode_metrics['mean_reward_energy_term'] = np.mean(
                self.episode_logs['reward_energy_terms'])
            self.episode_metrics['cumulative_reward_comfort_term'] = np.sum(
                self.episode_logs['reward_comfort_terms'])
            self.episode_metrics['mean_reward_comfort_term'] = np.mean(
                self.episode_logs['reward_comfort_terms'])
            self.episode_metrics['cumulative_abs_energy_penalty'] = np.sum(
                self.episode_logs['abs_energy_penalties'])
            self.episode_metrics['mean_abs_energy_penalty'] = np.mean(
                self.episode_logs['abs_energy_penalties'])
            self.episode_metrics['cumulative_abs_comfort_penalty'] = np.sum(
                self.episode_logs['abs_comfort_penalties'])
            self.episode_metrics['mean_abs_comfort_penalty'] = np.mean(
                self.episode_logs['abs_comfort_penalties'])
            self.episode_metrics['cumulative_power_demand'] = np.sum(
                self.episode_logs['total_power_demands'])
            self.episode_metrics['mean_power_demand'] = np.mean(
                self.episode_logs['total_power_demands'])
            self.episode_metrics['cumulative_temperature_violation'] = np.sum(
                self.episode_logs['total_temperature_violations'])
            self.episode_metrics['mean_temperature_violation'] = np.mean(
                self.episode_logs['total_temperature_violations'])
            try:
                self.episode_metrics['comfort_violation_time(%)'] = self.episode_logs['comfort_violation_timesteps'] / \
                    self.episode_logs['total_timesteps'] * 100
            except ZeroDivisionError:
                self.episode_metrics['comfort_violation_time(%)'] = np.nan

            # reset episode info
            self.episode_logs = {
                'rewards': [],
                'reward_energy_terms': [],
                'reward_comfort_terms': [],
                'abs_energy_penalties': [],
                'abs_comfort_penalties': [],
                'total_power_demands': [],
                'total_temperature_violations': [],
                'total_timesteps': 0,
                'comfort_violation_timesteps': 0
            }

            # Record the episode and dump
            for key, metric in self.episode_metrics.items():
                self.logger.record(
                    'episode/' + key, metric)
            self.logger.dump(step=self.timestep)

            # Count next episode
            self.episode_num += 1

        # DUMP with the frequency specified
        if self.timestep % self.dump_frequency == 0:
            self.logger.dump(step=self.timestep)

        return True

    def _on_training_end(self):
        self.logger.dump(step=self.timestep)
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
        train_env: Union[gym.Env, VecEnv],
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
        # [Declined for EnergyPlus API particularities]
        # if not isinstance(eval_env, VecEnv):
        #     eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.train_env = train_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.log_metrics = {
            'timesteps': [],
            'episodes_length': [],
            'episodes_cumulative_reward': [],
            'episodes_mean_reward': [],
            'episodes_cumulative_reward_energy_term': [],
            'episodes_mean_reward_energy_term': [],
            'episodes_cumulative_reward_comfort_term': [],
            'episodes_mean_reward_comfort_term': [],
            'episodes_cumulative_absolute_energy_penalty': [],
            'episodes_mean_absolute_energy_penalty': [],
            'episodes_cumulative_absolute_comfort_penalty': [],
            'episodes_mean_absolute_comfort_penalty': [],
            'episodes_cumulative_power_demand': [],
            'episodes_mean_power_demand': [],
            'episodes_cumulative_temperature_violation': [],
            'episodes_mean_temperature_violation': [],
            'episodes_comfort_violation': [],
        }

        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []
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

        if locals_["truncated"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            # if self.model.get_vec_normalize_env() is not None:
            #     try:
            #         sync_envs_normalization(self.training_env, self.eval_env)
            #     except AttributeError as e:
            #         raise AssertionError(
            #             "Training and eval env are not wrapped the same way, "
            #             "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
            #             "and warning above.") from e

            # Reset success rate buffer
            self._is_success_buffer = []

            # We close training env before to start the evaluation
            self.train_env.close()

            self._sync_envs()

            # GET evaluation episodes data:
            episodes_data = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                callback=self._log_success_callback,
            )

            # We close evaluation env and starts training env again
            self.eval_env.close()
            self.train_env.reset()

            if self.log_path is not None:
                for key, value in episodes_data.items():
                    self.log_metrics[key].append(value)
                self.log_metrics['timesteps'].append(self.num_timesteps)

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

            # Logging episodes' metrics
            cumulative_reward, std_cumulative_reward = np.mean(
                episodes_data['episodes_cumulative_reward']), np.std(
                episodes_data['episodes_cumulative_reward'])
            mean_ep_length, std_ep_length = np.mean(
                episodes_data['episodes_length']), np.std(
                episodes_data['episodes_length'])
            self.last_reward = cumulative_reward

            self.evaluation_metrics['episode_length'] = np.mean(
                episodes_data['episodes_length'])
            self.evaluation_metrics['cumulative_reward'] = np.mean(
                episodes_data['episodes_cumulative_reward'])
            self.evaluation_metrics['std_cumulative_reward'] = np.std(
                episodes_data['episodes_cumulative_reward'])
            self.evaluation_metrics['mean_reward'] = np.mean(
                episodes_data['episodes_mean_reward'])
            self.evaluation_metrics['std_mean_reward'] = np.std(
                episodes_data['episodes_mean_reward'])
            self.evaluation_metrics['cumulative_reward_energy_term'] = np.mean(
                episodes_data['episodes_cumulative_reward_energy_term'])
            self.evaluation_metrics['mean_reward_energy_term'] = np.mean(
                episodes_data['episodes_mean_reward_energy_term'])
            self.evaluation_metrics['cumulative_reward_comfort_term'] = np.mean(
                episodes_data['episodes_cumulative_reward_comfort_term'])
            self.evaluation_metrics['mean_reward_comfort_term'] = np.mean(
                episodes_data['episodes_mean_reward_comfort_term'])
            self.evaluation_metrics['cumulative_absolute_energy_penalty'] = np.mean(
                episodes_data['episodes_cumulative_absolute_energy_penalty'])
            self.evaluation_metrics['mean_absolute_energy_penalty'] = np.mean(
                episodes_data['episodes_mean_absolute_energy_penalty'])
            self.evaluation_metrics['cumulative_absolute_comfort_penalty'] = np.mean(
                episodes_data['episodes_cumulative_absolute_comfort_penalty'])
            self.evaluation_metrics['mean_absolute_comfort_penalty'] = np.mean(
                episodes_data['episodes_mean_absolute_comfort_penalty'])
            self.evaluation_metrics['cumulative_power_demand'] = np.mean(
                episodes_data['episodes_cumulative_power_demand'])
            self.evaluation_metrics['mean_power_demand'] = np.mean(
                episodes_data['episodes_mean_power_demand'])
            self.evaluation_metrics['cumulative_temperature_violation'] = np.mean(
                episodes_data['episodes_cumulative_temperature_violation'])
            self.evaluation_metrics['mean_temperature_violation'] = np.mean(
                episodes_data['episodes_mean_temperature_violation'])
            self.evaluation_metrics['comfort_violation(%)'] = np.mean(
                episodes_data['episodes_comfort_violation'])

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={
                    cumulative_reward: .2f} + /- {std_cumulative_reward: .2f}")
                print(f"Episode length: {
                    mean_ep_length: .2f} + /- {std_ep_length: .2f}")
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
            if cumulative_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(
                            self.best_model_save_path,
                            "model.zip"))
                self.best_mean_reward = cumulative_reward
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

    def _sync_envs(self):
        # normalization
        if is_wrapped(
                self.train_env,
                NormalizeObservation) and is_wrapped(
                self.eval_env,
                NormalizeObservation):
            self.eval_env.get_wrapper_attr('deactivate_update')()
            self.eval_env.get_wrapper_attr('set_mean')(
                self.train_env.get_wrapper_attr('mean'))
            self.eval_env.get_wrapper_attr('set_var')(
                self.train_env.get_wrapper_attr('var'))
