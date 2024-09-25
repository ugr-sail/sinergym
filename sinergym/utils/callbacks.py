"""Custom Callbacks for stable baselines 3 algorithms."""

import os
from typing import Any, Dict, List, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv

from sinergym.utils.constants import LOG_CALLBACK_LEVEL
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (BaseLoggerWrapper, NormalizeObservation,
                                     WandBLogger)


class LoggerEvalCallback(EventCallback):

    logger = TerminalLogger().getLogger(
        name='EVALUATION',
        level=LOG_CALLBACK_LEVEL)

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        train_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq_episodes: int = 5,
        deterministic: bool = True,
        excluded_metrics: List[str] = [
            'episode_num',
            'length (timesteps)',
            'time_elapsed (hours)'],
        verbose: int = 1,
    ):
        """ Callback for evaluating an agent during training process logging all important data in WandB platform if is activated. It must be wrapped with BaseLoggerWrapper child class.

        Args:
            eval_env (Union[gym.Env, VecEnv]): Environment to evaluate the agent.
            train_env (Union[gym.Env, VecEnv]): Environment used for training.
            n_eval_episodes (int, optional): Number of episodes to evaluate the agent. Defaults to 5.
            eval_freq_episodes (int, optional): Evaluate the agent every eval_freq call of the callback. Defaults to 5.
            deterministic (bool, optional): Whether the evaluation should use a stochastic or deterministic actions. Defaults to True.
            excluded_metrics (List[str], optional): List of metrics to exclude from the evaluation. Defaults to ['episode_num', 'length (timesteps)', 'time_elapsed (hours)'].
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super().__init__(verbose=verbose)

        assert is_wrapped(
            train_env, BaseLoggerWrapper), 'Training environment must be wrapped with BaseLoggerWrapper in order to be compatible with this callback.'

        # Attributes
        self.eval_env = eval_env
        self.train_env = train_env
        self.n_eval_episodes = n_eval_episodes
        # Last train model step generate a env.reset() automatically, we want
        # to avoid this
        self.eval_freq = eval_freq_episodes * \
            train_env.get_wrapper_attr('timestep_per_episode') - eval_freq_episodes
        self.save_path = self.train_env.get_wrapper_attr(
            'workspace_path') + '/evaluation'
        # Make dir if not exists
        os.makedirs(self.save_path, exist_ok=True)
        self.deterministic = deterministic
        self.evaluation_num = 0

        # Best mean reward record
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

        # wandb flag
        self.wandb_log = is_wrapped(self.train_env, WandBLogger)

        self.evaluation_columns = [col for col in eval_env.get_wrapper_attr(
            'summary_metrics') if col not in excluded_metrics]
        self.evaluation_metrics = pd.DataFrame(
            columns=self.evaluation_columns)

        # Define metric for evaluation as X axis if WandB session is activated
        if self.wandb_log:
            self.train_env.get_wrapper_attr('wandb_run').define_metric(
                'Evaluation/*',
                step_metric='Evaluation/evaluation_num')

    def _on_step(self) -> bool:

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            self._on_event()

        return continue_training

    def _on_event(self) -> None:

        # Increment evaluation index
        self.evaluation_num += 1
        # Close current training environment to execute an evaluation
        if self.wandb_log:
            self.train_env.get_wrapper_attr('set_wandb_finish')(False)
            self.train_env.close()
            self.train_env.get_wrapper_attr('set_wandb_finish')(True)
        else:
            self.train_env.close()

        # We sincronize the evaluation and training envs (for example, for
        # normalization calibration data)
        self._sync_envs()

        # -------------------------------- Evaluation -------------------------------- #

        # Execute evaluation and extract episodes dataframe
        evaluation_episodes = self._evaluate_policy()

        # ---------------------- Process evaluation information ---------------------- #

        # Process episodes data in means
        evaluation_summary = {'evaluation_num': self.evaluation_num}
        for key in evaluation_episodes.keys():
            if key in self.evaluation_columns:
                evaluation_summary[key] = np.mean(
                    evaluation_episodes[key])

        # ------------------------------ Log information ----------------------------- #

        # Add evaluation summary to the evaluation metrics (CSV)
        self.evaluation_metrics = self.evaluation_metrics._append(
            evaluation_summary, ignore_index=True)
        self.evaluation_metrics.to_csv(
            self.save_path + '/evaluation_metrics.csv')

        # Add evaluation metrics to wandb plots if enabled
        if self.wandb_log:
            self.train_env.get_wrapper_attr('_log_data')(
                data={'Evaluation': evaluation_summary})

        # Terminal information when verbose is active
        if self.verbose >= 1:
            self.logger.info(
                'Evaluation num_timesteps={}, episode_reward={} +/- {}'.format(
                    self.num_timesteps,
                    evaluation_summary['mean_reward'],
                    evaluation_summary['std_reward']
                )
            )

        # ------------------------ Save best model if required ----------------------- #

        # Condition to determine when a modes is the best
        if evaluation_summary['mean_reward'] > self.best_mean_reward:
            if self.verbose >= 1:
                self.logger.info('New best mean reward!')

            # Save new best model
            self.model.save(
                os.path.join(
                    self.save_path,
                    'best_model.zip'))
            self.best_mean_reward = evaluation_summary['mean_reward']

        # We close evaluation env and starts training env again
        self.eval_env.close()
        self.train_env.reset()

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

    def _evaluate_policy(self) -> Dict[str, List[Any]]:
        """
        Runs policy for ``n_eval_episodes`` episodes and returns average reward and other Sinergym metrics, depending its backend logger.

        Returns:
            Dict[str, List[Any]]: Dictionary with logger summary metrics for each evaluation episode executed. Keys depend on the logger used.
        """

        result = {key: [] for key in self.evaluation_columns}

        for _ in range(self.n_eval_episodes):

            obs, _ = self.eval_env.reset()
            state = None
            truncated = terminated = False
            # ---------------------------------------------------------------------------- #
            #                     Running episode and accumulate values                    #
            # ---------------------------------------------------------------------------- #
            while not (truncated or terminated):
                action, state = self.model.predict(
                    obs, state=state, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(
                    action)

            # ---------------------------------------------------------------------------- #
            #                     Storing last episode in results dict                     #
            # ---------------------------------------------------------------------------- #
            summary = self.eval_env.get_wrapper_attr('get_episode_summary')()
            # Append values to result dictionary
            for key in result.keys():
                result[key].append(summary[key])

        return result
