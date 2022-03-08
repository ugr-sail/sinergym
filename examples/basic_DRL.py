
from datetime import datetime

import gym
import mlflow
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from sinergym.utils.callbacks import LoggerEvalCallback
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import LoggerWrapper

environment ="Eplus-demo-v1"
episodes = 4
experiment_date = datetime.today().strftime('%Y-%m-%d %H:%M')
#---------------------------------------------------------------------------------------------#
# register run name
experiment_date = datetime.today().strftime('%Y-%m-%d %H:%M')
name ='DQN-' +environment +'-episodes_'+episodes
name += '(' + experiment_date + ')'

with mlflow.start_run(run_name=name):

    env = gym.make(environment, reward=LinearReward())
    env = LoggerWrapper(env)

    ######################## TRAINING ########################

    # Defining model(algorithm)
    model = DQN('MlpPolicy', env, verbose=1,
                    learning_rate=.0007,
                    buffer_size=1000000,
                    learning_starts=50000,
                    batch_size=32,
                    tau=0.005,
                    gamma=0.005,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=10000,
                    exploration_fraction=.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=.05,
                    max_grad_norm=.5)

    #--------------------------------------------------------#

    # Calculating n_timesteps_episode for training
    n_timesteps_episode = env.simulator._eplus_one_epi_len / \
        env.simulator._eplus_run_stepsize
    timesteps = episodes * n_timesteps_episode

    # For callbacks processing
    env_vec = DummyVecEnv([lambda: env])

    # Using Callbacks for training
    callbacks = []

    # Set up Evaluation and saving best model
    eval_callback = LoggerEvalCallback(
        env_vec,
        best_model_save_path='best_model/' + name + '/',
        log_path='best_model/' + name + '/',
        eval_freq=n_timesteps_episode * 2,
        deterministic=True,
        render=False,
        n_eval_episodes=2)
    callbacks.append(eval_callback)

    callback = CallbackList(callbacks)

    # Training
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        log_interval=1)
    model.save(env.simulator._env_working_dir_parent + '/' + name)

    # End mlflow run
    mlflow.end_run()
