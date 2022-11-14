import os

from setuptools import find_packages, setup

with open(os.path.join("sinergym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='sinergym',
      version=__version__,
      packages=[package for package in find_packages(
      ) if package.startswith("sinergym")],
      license='MIT',
      author='J. Jiménez, J. Gómez, M. Molina, A. Manjavacas, A. Campoy',
      author_email='alejandroac79@gmail.com',
      description='The goal of sinergym is to create an environment following OpenAI Gym interface for wrapping simulation engines for building control using deep reinforcement learning.',
      url='https://github.com/ugr-sail/sinergym',
      keywords='control reinforcement-learning buildings reinforcement-learning-environments',
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
              'sphinx-rtd-theme',  # documentation theme
              'sphinxcontrib-spelling',  # documentation spelling
              'sphinx-multiversion',  # documentation versioning
              'sphinx-multitoc-numbering',  # Section numbering
              'pyenchant',
              'google-api-python-client',
              'oauth2client',
              'google-cloud-storage',
              'nbsphinx',
              'nbsphinx_link',
              'IPython'
          ],
          'test': [
              'pytest',
              'stable-baselines3'
          ],
          'dev': ['pytest'],
          'DRL': [
              'stable-baselines3',
              'mlflow',
              'tensorflow',
              'tensorboard_plugin_profile'
          ],
          'doc': [
              'sphinx',
              'sphinx-rtd-theme',
              'sphinxcontrib-spelling',
              'sphinx-multiversion',
              'sphinx-multitoc-numbering',
              'pyenchant',
              'nbsphinx',
              'nbsphinx_link',
              'IPython'
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
