# Sinergym

<div align="center">
  <img src="images/logo.png" width=40%><br><br>
</div>

</p>
  <p align="center">
    <a href="https://github.com/ugr-sail/sinergym/releases">
      <img alt="Github latest release" src="https://img.shields.io/github/release-date/ugr-sail/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/commits/main">
      <img alt="Github last commit" src="https://img.shields.io/github/last-commit/ugr-sail/sinergym" />
    </a>
    <a href="https://pypi.org/project/sinergym/">
      <img alt="Pypi version" src="https://img.shields.io/pypi/v/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/stargazers">
      <img alt="Pypi downloads" src="https://img.shields.io/pypi/dm/sinergym" />
    </a>
    <a href="https://codecov.io/gh/ugr-sail/sinergym">
      <img src="https://codecov.io/gh/ugr-sail/sinergym/branch/main/graph/badge.svg" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/graphs/contributors">
      <img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/ugr-sail/sinergym" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/issues">
      <img alt="Github issues" src="https://img.shields.io/github/issues/ugr-sail/sinergym?color=0088ff" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/pulls">
      <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/ugr-sail/sinergym?color=0088ff" />
    </a>
    <a href="https://github.com/ugr-sail/sinergym/blob/main/LICENSE">
      <img alt="Github License" src="https://img.shields.io/github/license/ugr-sail/sinergym" />
    </a>
    <a href="https://www.python.org/downloads/release/python-3100/">
      <img alt="Pypi Python version" src="https://img.shields.io/pypi/pyversions/sinergym" />
    </a>
    <br />
    <br />
    <a href="https://code.visualstudio.com/">
      <img src="https://img.shields.io/badge/Supported%20by-VSCode%20Power%20User%20%E2%86%92-gray.svg?colorA=655BE1&colorB=4F44D6&style=for-the-badge"/>
    </a>
  </p>

**Welcome to Sinergym!**

<div align="center">
  <img src="images/general_blueprint.png" width=80%><br><br>
</div>

The goal of this project is to create an environment following [Gymnasium interface](https://gymnasium.farama.org/), for wrapping simulation engines for building control using **deep reinforcement learning**.

Please, help us to improve by **reporting your questions and issues** [here](https://github.com/ugr-sail/sinergym/issues). It is easy, just 2 clicks using our issue templates (questions, bugs, improvements, etc.). More detailed info on how to report issues [here](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). Don't forget to take a look at [CONTRIBUTING.md](https://github.com/ugr-sail/sinergym/blob/main/CONTRIBUTING.md) if you're thinking about contributing to Sinergym.

The main functionalities of *Sinergym* are the following :

-  **Include different simulation engines**. Communication between
   Python and [EnergyPlus](https://energyplus.net/) is established
   using [BCVTB](https://simulationresearch.lbl.gov/bcvtb/FrontPage) middleware.
   Since this tool allows for interacting with several simulation
   engines, more of them (e.g.
   [OpenModelica](https://openmodelica.org/)) could be included in
   the backend while maintaining the Gymnasium API.

-  **Benchmark environments**. Similarly to *Atari* or *Mujoco* environments
   for RL community, we are designing a set of environments for
   benchmarking and testing deep RL algorithms. These environments may
   include different buildings, weathers, action/observation spaces, function rewards, etc.

-  **Customizable environments**. We aim to provide a
   package that allows to modify experimental settings in an easy
   manner. The user can create his own environments defining his own
   building model, weather, reward, observation/action space and variables, environment name, etc.
   The user can also use these pre-configured environments available in *Sinergym* 
   and change some aspect of it (for example, the weather) in such 
   a way that he does not  have to make an entire definition of the 
   environment and can start from one pre-designed by us.
   Some parameters directly associated with the simulator can be set as **extra configuration** 
   as well, such as people occupant, time-steps per simulation hour, run-period, etc.

-  **Customizable components**: *Sinergym* is easily scalable by third parties.
   Following the structure of the implemented classes, new custom components 
   can be created for new environments such as function rewards, wrappers,
   controllers, etc.

-  **Automatic Building Model adaptation to user changes**: Building models (*IDF*) will be
   adapted to specification of each simulation by the user. For example, ``Designdays`` and 
   ``Location`` components from *IDF* files will be adapted to weather file (*EPW*) specified in
   *Sinergym* simulator backend without any intervention by the user (only the environment definition).
   *BCVTB middleware* external interface in *IDF* model and *variables.cfg* file is generated when 
   simulation starts by *Sinergym*, this definition depends on action and observation space and variables defined.
   In short, *Sinergym* automates the whole process of model adaptation so that the user 
   only has to define what he wants for his environment.

-  **Automatic external interface integration for actions**. Sinergym provides functionality to obtain information 
   about the environments such as the zones or the schedulers available in the environment model. Using that information,
   which is possible to export in a excel, users can know which controllers are available in the building and, then, control 
   them with an external interface from an agent. To do this, users will make an **action definition** in which it is
   indicated which default controllers they want to replace in a specific format and *Sinergym* will take care of the relevant internal 
   changes in the model.

-  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been customized by our team in order to test easily these environments
   with deep reinforcement learning algorithms. 
   This tool can be used with any other DRL library that supports the * Gymnasium* interface as well.

-  **Weights & Biases tracking and visualization**. One of Sinergym's objectives is to automate
   and facilitate the training, reproducibility and comparison of agents in simulation-based 
   building control problems, managing and monitoring model lifecycle from training to deployment. [WandB](https://wandb.ai/site) 
   is an open-source platform for the machine learning lifecycle helping us with this issue. 
   It lets us register experiments hyperparameters, visualize data recorded in real-time, 
   and store artifacts with experiment outputs and best obtained models. 

-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with *Sinergym*, we tell you some details about how to do it.

-  **Notebooks examples**. *Sinergym* develops code in notebook format with the purpose of offering use cases to 
   the users in order to help them become familiar with the tool. They are constantly updated, along with the updates 
   and improvements of the tool itself.

-  This project is accompanied by extensive **documentation**, **unit tests** and **github actions workflows** to make 
   *Sinergym* an efficient ecosystem for both understanding and development.

-  Many more!

_This is a project in active development. Stay tuned for upcoming releases._

<div align="center">
  <img src="images/operation_diagram.png"><br><br>
</div>

## List of available environments

If you would like to see a complete and updated list of our available environments, please visit [our list](https://ugr-sail.github.io/sinergym/compilation/main/pages/environments.html#) in the official *Sinergym* documentation.

## Installation

Please, visit [INSTALL.md](https://github.com/ugr-sail/sinergym/blob/main/INSTALL.md) for more information about Sinergym installation.

## Usage example

If you used our Dockerfile during installation, you should have the *try_env.py* file in your workspace as soon as you enter in. In case you have installed everything on your local machine directly, place it inside our cloned repository. In any case, we start from the point that you have at your disposal a terminal with the appropriate python version and *Sinergym* running correctly.

*Sinergym* uses the standard Gymnasium API. So basic loop should be something like:

```python

import gymnasium as gym
import sinergym
# Create the environment
env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1')
# Initialize the episode
obs, info = env.reset()
terminated = False
R = 0.0
while not terminated:
    a = env.action_space.sample() # random action selection
    obs, reward, terminated, truncated, info = env.step(a) # get new observation and reward
    R += reward
print('Total reward for the episode: %.4f' % R)
env.close()
```

Notice that a folder will be created in the working directory after creating the environment. It will contain the EnergyPlus outputs produced during the simulation.

:pencil: For more examples and details, please visit our [usage examples](https://ugr-sail.github.io/sinergym/compilation/main/pages/notebooks/basic_example.html#Basic-example) documentation section.

## Google Cloud Platform support

For more information about this functionality, please, visit our documentation [here](https://ugr-sail.github.io/sinergym/compilation/main/pages/gcloudAPI.html#sinergym-with-google-cloud).

## Projects using Sinergym

The following are some of the projects benefiting from the advantages of Sinergym:

- [Demosthen/ActiveRL](https://github.com/Demosthen/ActiveRL)
- [VectorInstitute/HV-Ai-C](https://github.com/VectorInstitute/HV-Ai-C)
- [rdnfn/beobench](https://github.com/rdnfn/beobench)

:pencil: If you want to appear in this list, do not hesitate to send us a PR and include the following badge in your repository:

<a href="https://ugr-sail.github.io/sinergym/compilation/main/index.html">
    <img src="https://img.shields.io/badge/Powered%20by-Sinergym%20%E2%86%92-gray.svg?colorA=00BABF&colorB=4BF2F7&style=for-the-badge"/>
</a>

## Citing Sinergym

If you use *Sinergym* in your work, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3486611.3488729):

```bibtex
@inproceedings{2021sinergym,
    title={Sinergym: A Building Simulation and Control Framework for Training Reinforcement Learning Agents}, 
    author={Jiménez-Raboso, Javier and Campoy-Nieves, Alejandro and Manjavacas-Lucas, Antonio and Gómez-Romero, Juan and Molina-Solana, Miguel},
    year={2021},
    isbn = {9781450391146},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3486611.3488729},
    doi = {10.1145/3486611.3488729},
    booktitle = {Proceedings of the 8th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation},
    pages = {319–323},
    numpages = {5},
}
```
