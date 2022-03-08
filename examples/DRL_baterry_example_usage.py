import subprocess
import sys

"""
## Basic call (all default values)
args = ["DRL_battery.py",
        "--environment", "Eplus-demo-v1"]
subprocess.Popen([sys.executable or 'python'] + args)
"""

## Complete call (all parameters)
args = ["DRL_battery.py",
        "--environment", "Eplus-demo-v1",
        '--episodes', '3',
        '--algorithm', 'DDPG',
        '--reward', 'exponential',
        # '--normalization', #We cant use normalization on "Eplus-demo-v1"
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
        '--group_name', 'DRL',
        '--auto_delete',
        '--learning_rate', '.02',
        '--gamma', '.8',
        '--n_steps', '10',
        '--gae_lambda', '1',
        '--ent_coef', '0.1',
        '--vf_coef', '0.5',
        '--max_grad_norm', '0.5',
        '--rms_prop_eps', '1e-05',
        '--buffer_size', '2000000',
        '--learning_starts', '100',
        '--tau', '0.005',
        '--sigma', '0.1']
subprocess.Popen([sys.executable or 'python'] + args)
