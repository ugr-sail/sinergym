# energym

OpenAI Gym environment for wrapping simulation engines for building control.

Currently only EnergyPlus is supported, but the package should be agnostic of the engine. Additionally, we will include several buildings and weather settings for testing and creating benchmarking environments.

## Installation process

To install energym, follow these steps.

First, it is recommended to create a virtual environment. You can do so by:

```sh
$ sudo apt-get install python-virtualenv virtualenv
$ virtualenv env_energym --python=python3
$ source env_energym/bin/activat
```

Then, clone this repository using this command:
```
$ git clone https://github.com/jajimer/energym.git
```

### 1. Install Energy Plus 8.6.0

Next step is downloading Energy Plus. Currently only version 8.6.0 has been tested,
but code may also work for other versions.

Follow the instructions [here](https://energyplus.net/downloads) and install Linux platform.
Choose any location to install the software (not needed to be inside the repository folder).

Once installed, a folder called ``Energyplus-8-6-0`` should appear.

### 2. Install BCVTB software

Follow the instructions [here](https://simulationresearch.lbl.gov/bcvtb/Download) for installing BCVTB software.
Another option is to copy the ``bcvtb`` folder from [this repository](https://github.com/zhangzhizza/Gym-Eplus/tree/master/eplus_env/envs).

### 3. Set environment variables

Two environment variables must be set: ``EPLUS_PATH`` and ``BCVTB_PATH``, with the locations where EnergyPlus and BCVTB are installed respectively.

### 4. Install the package

Finally, energym can be installed by running this command where ``setup.py`` is located.

```sh
pip install -e .
```

And that's all!

## Usage example

Run ``python test_env.py`` for testing the demo environment. Energym uses the standard openAI gym API. So basic loop should be something like:

```python

import gym
import energym

env = gym.make('Eplus-discrete-v1')
obs = env.reset()
done = False
R = 0.0
while not done:
    a = env.action_space.sample() # action selection
    obs, reward, done, info = env.step(a) # get new observation and reward
    R += reward
print('Total reward for the episode: %.4f' % R)
env.close()
````

This code executes a control episode from January 1st to March 31st. Current implemented environment performs the control of the heating setpoint of an HVAC system. 
There are 9 possible discrete actions (setpoint from 15C to 24C), and the observation is a vector with 16 components. 
The reward combines the power used by the HVAC system and the thermal discomfort:

```python

reward = - beta * power - penalty_comfort
```

where ``beta = 1e-4`` and the penalty comfort is the difference between current room temperature and the comfort range (between 20C and 22C). Rewards are always negative, so an agent must learn to minimize power consumption while satisfying comfort constraint.

## To do list

  - [ ] Refactor EnergyPlus class (currently in eplus_old.py, but eplus.py should be used).
  - [x] Include reward calculation
  - [x] Data paths are included in the environment, so no hard-coding needed
  - [ ] Try an example using SB3
  - [ ] Create and test Dockerfile
