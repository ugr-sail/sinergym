from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='energym',
      version='0.0.1',
      install_requires=reqs,
      include_package_data=True,
)