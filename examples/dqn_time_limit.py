"""
Evaluation of time limits using DQN and sinergym.
https://arxiv.org/abs/1712.00378 (Pardo et al., 2017)

The authors of the paper argue that termination conditions due
to time limits should be treated specially, since they do not
represent a real terminal state. Therefore, instead of assigning
them a V(s) = 0, we should bootstrap as if they episode would continue.

In this script an implementation of this method (Partial-Episode Bootstrapping)
is applied to a simple DQN algorithm with the hot discrete sinergym environment.
"""


import argparse
from typing import Optional, Tuple, Union

import gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common import logger
from torch.nn import functional as F

from sinergym.utils.wrappers import NormalizeObservation


class PartialDQN(DQN):
    """Slight modification of DQN for partial-episode bootstrapping"""

    def __init__(
        self,
        policy: str,
        env: gym.Env,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        verbose: int = 0,
        seed: Optional[int] = None
    ):

        super(PartialDQN, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            seed=seed
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(
                    replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                # target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
                target_q_values = replay_data.rewards + self.gamma * next_q_values
            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long())
            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        # Increase update counter
        self._n_updates += gradient_steps
        logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))


def main():
    """Run an experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo",
        help="which version of DQN to use",
        type=str,
        required=True)
    parser.add_argument("-s", "--seed", help="seed", type=int, default=42)
    args = parser.parse_args()
    assert args.algo in ['simple', 'peb']
    # Environments
    env = NormalizeObservation(gym.make('Eplus-discrete-hot-v1'))
    eval_env = NormalizeObservation(gym.make('Eplus-discrete-hot-v1'))
    # Parameters
    GAMMA = 0.9
    LEARNING_STARTS = 20_000
    TARGET_UPDATE_INTERVAL = 5_000
    EXPLORATION_FRACTION = 0.8
    INITIAL_EPS = 1.0
    FINAL_EPS = 0.05
    TOTAL_TIMESTEPS = 250_000
    EVAL_FREQ = 25_000
    # Models
    if args.algo == 'simple':
        agent = DQN
        results_path = 'dqn_simple_%d' % args.seed
    else:
        agent = PartialDQN
        results_path = 'dqn_peb_%d' % args.seed

    model = agent("MlpPolicy", env, verbose=1,
                  gamma=GAMMA,
                  learning_starts=LEARNING_STARTS,
                  target_update_interval=TARGET_UPDATE_INTERVAL,
                  exploration_fraction=EXPLORATION_FRACTION,
                  exploration_initial_eps=INITIAL_EPS,
                  exploration_final_eps=FINAL_EPS,
                  seed=args.seed)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=1,
        eval_log_path=results_path)


if __name__ == '__main__':
    main()
