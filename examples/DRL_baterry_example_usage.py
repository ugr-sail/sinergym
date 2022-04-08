import subprocess
import sys


## Basic call (all default values)
args = ["DRL_battery.py",
        "--environment", "Eplus-demo-v1"]
subprocess.Popen([sys.executable or 'python'] + args)


## Complete call (all parameters)
args = ["DRL_battery.py",
        "--environment", "Eplus-5Zone-cool-continuous-v1",
        '--episodes', '3',
        '--algorithm', 'DDPG',
        '--reward', 'exponential',
        '--normalization',
        '--multiobs',
        '--logger',
        '--tensorboard', './tensorboard_log',
        '--evaluation',
        '--eval_freq', '3',
        '--eval_length', '3',
        '--log_interval', '3',
        '--seed', '344561',
        '--remote_store',
        '--mlflow_store',
        '--group_name', 'DRL_DDPG',
        '--auto_delete',
        '--n_steps', '5',
        '--learning_starts', '100',
        '--sigma', '0.1']
subprocess.Popen([sys.executable or 'python'] + args)


## A2C
args = ["DRL_battery.py",
        "--environment", "Eplus-5Zone-cool-discrete-v1",
        '--episodes', '12',
        '--algorithm', 'A2C',
        '--reward', 'exponential',
        '--normalization',
        '--multiobs',
        '--logger',
        '--tensorboard', './tensorboard_log',
        '--evaluation',
        '--eval_freq', '3',
        '--eval_length', '3',
        '--log_interval', '3',
        '--seed', '55555',
        '--remote_store',
        '--mlflow_store',
        '--group_name', 'DRL_A2C',
        '--auto_delete',
        '--learning_rate', '.02',
        '--gamma', '.9',
        '--n_steps', '5',
        '--gae_lambda', '1.5',
        '--ent_coef', '0.15',
        '--vf_coef', '0.9',
        '--rms_prop_eps', '1e-05',
        '--learning_starts', '100',]
subprocess.Popen([sys.executable or 'python'] + args)