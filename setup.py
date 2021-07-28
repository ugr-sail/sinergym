from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='energym',
      version='1.0.0',
      install_requires=reqs,
      include_package_data=True,
      extras_require={
          'extras': [
              'matplotlib',  # visualization
              'stable-baselines3',  # DRL with pytorch
              'mlflow',  # tracking ML experiments
              'tensorboard', # Training logger
              'pytest',  # Unit test repository
              'sphinx',  # documentation
              'sphinx-rtd-theme' #documentation theme
          ],
          'test': [
              'pytest'
          ],
          'DRL': [
              'stable-baselines3',
              'mlflow',
              'tensorboard'
          ],
          'doc': [
              'sphinx',
              'sphinx-rtd-theme'
          ],
          'visualization': [
              'matplotlib'
          ]
      }
      )
