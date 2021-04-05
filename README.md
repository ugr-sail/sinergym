# energym

<div align="center">
  <img src="images/logo.png" width=40%><br><br>
</div>

**Welcome to energym!**

This is a project based on Zhiang Zhang and Khee Poh Lam [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus).

The goal of this project is to create an environment following OpenAI Gym interface for wrapping simulation engines for building control using deep reinforcement learning.

The main functionalities of Energym are the following :

  - **Benchmark environments**. Similarly to Atari or Mujoco environments for RL community, we are designing a set of environments for benchmarking and testing deep RL algorithms. These environments may include different buildings, weathers or action spaces.
  - **Develop different experimental settings**. We aim to provide a package that allows to modify experimental settings in an easy manner. For example several reward functions or observation variables may be defined. 
  - **Include different simulation engines**. Communication between Python and Energy Plus is established using BCVTB. Since this tool allows for interacting with several simulation engines, more of them (e.g. Modelica) could be included in the backend while maintaining the Gym API.
  - Many more!

_This is a work in progress project. Stay tuned for upcoming releases._

## List of available environments

| Env. name                            | Location        | IDF file           | Weather type (*)           | Action space | Simulation period |
|--------------------------------------|-----------------|--------------------|----------------------------|--------------|-------------------|
| Eplus-demo-v1                        | Pittsburgh, USA | 5ZoneAutoDXVAV.idf |             -              | Discrete(10) |   01/01 - 31/03   |
| Eplus-discrete-hot-v1                | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B)        | Discrete(10) |   01/01 - 31/12   |
| Eplus-discrete-mixed-v1              | New York, USA   | 5ZoneAutoDXVAV.idf |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
| Eplus-discrete-cool-v1               | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
| Eplus-continuous-hot-v1              | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B)        | Box(2)       |   01/01 - 31/12   |
| Eplus-continuous-mixed-v1            | New York, USA   | 5ZoneAutoDXVAV.idf |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/12   |
| Eplus-continuous-cool-v1             | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C)      | Box(2)       |   01/01 - 31/12   |
| Eplus-discrete-stochastic-cool-v1    | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
| Eplus-continuous-stochastic-hot-v1   | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B) (**)   | Box(2)       |   01/01 - 31/12   |

(*) Weather types according to [DOE's classification](https://www.energycodes.gov/development/commercial/prototype_models#TMY3).

(**) In these environments, weather series change from episode to episode. Gaussian noise with 0 mean and 2.5 std is added to the original values.

## Installation process

To install energym, follow these steps.

First, it is recommended to create a virtual environment. You can do so by:

```sh
$ sudo apt-get install python-virtualenv virtualenv
$ virtualenv env_energym --python=python3.7
$ source env_energym/bin/activate
```

Then, clone this repository using this command:
```
$ git clone https://github.com/jajimer/energym.git
```

We include a Dockerfile for installing all dependencies and setting up the image for running energym. If you prefer installing it manually, follow the steps below.

### 1. Install Energy Plus 8.6.0

Next step is downloading Energy Plus. Currently only version 8.6.0 has been tested,
but code may also work with other versions.

Follow the instructions [here](https://energyplus.net/downloads) and install for Linux system (only Ubuntu is supported).
Choose any location to install the software. Once installed, a folder called ``Energyplus-8-6-0`` should appear in the selected location.

### 2. Install BCVTB software

Follow the instructions [here](https://simulationresearch.lbl.gov/bcvtb/Download) for installing BCVTB software.
Another option is to copy the ``bcvtb`` folder from [this repository](https://github.com/zhangzhizza/Gym-Eplus/tree/master/eplus_env/envs).

### 3. Set environment variables

Two environment variables must be set: ``EPLUS_PATH`` and ``BCVTB_PATH``, with the locations where EnergyPlus and BCVTB are installed respectively.

### 4. Install the package

Finally, energym can be installed by running this command at the repo root folder:

```sh
pip install -e .
```

Extra libraries can be installed by typing ``pip install -e .[extras]``. They are intended for running and analysing DRL algorithms over energym, but they are
not a requirement of the package.   

And that's all!

## Usage example

Energym uses the standard openAI gym API. So basic loop should be something like:

```python

import gym
import energym

env = gym.make('Eplus-demo-v1')
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

Notice that a folder will be created in the working directory after creating the environment. They are used for saving EnergyPlus outputs during the simulation.
