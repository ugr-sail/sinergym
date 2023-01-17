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
              'stable-baselines3==1.6.0',  # DRL with pytorch
              'mlflow==1.28.0',  # tracking ML experiments
              'tensorflow==2.9.1',
              'tensorboard_plugin_profile==2.8.0',  # Training logger
              'pytest==7.1.2',  # Unit test repository
              'sphinx==5.1.1',  # documentation
              'sphinx-rtd-theme==1.0.0',  # documentation theme
              'sphinxcontrib-spelling==7.7.0',  # documentation spelling
              'sphinx-multiversion==0.2.4',  # documentation versioning
              'sphinx-multitoc-numbering==0.1.3',  # Section numbering
              'pyenchant==3.2.0',
              'nbsphinx==0.8.9',
              'nbsphinx_link==1.3.0',
              'google-api-python-client==2.58.0',
              'oauth2client==4.1.3',
              'google-cloud-storage==2.3.2',
              'IPython'
          ],
          'test': [
              'pytest==7.1.2',
              'stable-baselines3==1.6.0'
          ],
          'dev': ['pytest==7.1.2'],
          'DRL': [
              'stable-baselines3==1.6.0',
              'mlflow==1.28.0',
              'tensorflow==2.9.1',
              'tensorboard_plugin_profile==2.8.0'
          ],
          'doc': [
              'sphinx==5.1.1',
              'sphinx-rtd-theme==1.0.0',
              'sphinxcontrib-spelling==7.7.0',
              'sphinx-multiversion==0.2.4',
              'sphinx-multitoc-numbering==0.1.3',
              'pyenchant==3.2.0',
              'nbsphinx==0.8.9',
              'nbsphinx_link==1.3.0',
              'IPython'
          ],
          'visualization': [
              'matplotlib',
          ],
          'gcloud': [
              'google-api-python-client==2.58.0',
              'oauth2client==4.1.3',
              'google-cloud-storage==2.3.2',
          ]
      }
      )
