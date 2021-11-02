import os
from setuptools import setup

with open(os.path.join("sinergym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='sinergym',
      version=__version__,
      install_requires=reqs,
      include_package_data=True,
      extras_require={
          'extras': [
              'matplotlib',  # visualization
              'stable-baselines3',  # DRL with pytorch
              'mlflow',  # tracking ML experiments
              'tensorflow',
              'tensorboard_plugin_profile',  # Training logger
              'pytest',  # Unit test repository
              'sphinx',  # documentation
              'sphinx-rtd-theme'  # documentation theme
          ],
          'test': [
              'pytest'
          ],
          'DRL': [
              'stable-baselines3',
              'mlflow',
              'tensorflow',
              'tensorboard_plugin_profile'
          ],
          'doc': [
              'sphinx',
              'sphinx-rtd-theme'
          ],
          'visualization': [
              'matplotlib',
          ],
          'gcloud': [
              'google-api-python-client',
              'oauth2client',
              'google-cloud-storage'
          ]
      }
      )
