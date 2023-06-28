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
              # DRL with pytorch
              'stable-baselines3',
              'wandb',
              'pytest',
              'pytest-cov',
              'pytest-xdist',  # Unit test repository
              'sphinx',  # documentation
              'sphinx-rtd-theme',  # documentation theme
              'sphinxcontrib-spelling',  # documentation spelling
              'sphinxcontrib-jquery',
              # documentation versioning
              'sphinx-multiversion @ git+https://github.com/Holzhaus/sphinx-multiversion#egg=sphinx-multiversion',
              'sphinx-multitoc-numbering',  # Section numbering
              'pyenchant',
              'nbsphinx',
              'nbsphinx_link',
              'google-api-python-client==2.58.0',
              'oauth2client==4.1.3',
              'google-cloud-storage==2.5.0',
              'IPython'
          ],
          'test': [
              'pytest',
              'pytest-cov',
              'pytest-xdist',
          ],
          'DRL': [
              'stable-baselines3',
              'wandb'
          ],
          'doc': [
              'sphinx',
              'sphinx-rtd-theme',
              'sphinxcontrib-spelling',
              'sphinxcontrib-jquery',
              'sphinx-multiversion @ git+https://github.com/Holzhaus/sphinx-multiversion#egg=sphinx-multiversion',
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
              'google-api-python-client==2.58.0',
              'oauth2client==4.1.3',
              'google-cloud-storage==2.5.0',
          ]
      }
      )
