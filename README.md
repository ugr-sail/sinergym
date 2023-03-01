# Sinergym

> :warning: Stable Baselines 3 are working in order to have [gymnasium support](https://github.com/DLR-RM/stable-baselines3/pull/780). It is possible that SB3 algorithms don't work correctly with Sinergym environments temporally.

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

Please, help us to improve by **reporting your questions and issues** [here](https://github.com/ugr-sail/sinergym/issues). It is easy, just 2 clicks using our issue templates (questions, bugs, improvements, etc.). More detailed info on how to report issues [here](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). 

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

-  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with *Sinergym*, we tell you some details about how doing it.

-  **Mlflow tracking server**. [Mlflow](https://mlflow.org/) is an open source platform for the machine
   learning lifecycle. This can be used with Google Cloud remote server (if you have Google Cloud account) 
   or using local store. This will help you to manage and store your runs and artifacts generated in an orderly
   manner.

-  **Data Visualization**. Using *Sinergym* logger or Tensorboard server to visualize training and evaluation information
   in real-time.

-  **Notebooks examples**. *Sinergym* develops code in notebook format with the purpose of offering use cases to 
   the users in order to help them become familiar with the tool. They are constantly updated, along with the updates 
   and improvements of the tool itself.

-  This project is accompanied by extensive **documentation**, **unit tests** and **github actions workflows** to make 
   *Sinergym* an efficient ecosystem for both understanding and development.

-  Many more!

_This is a work in progress project. Stay tuned for upcoming releases._

<div align="center">
  <img src="images/operation_diagram.png"><br><br>
</div>

## List of available environments

If you would like to see a complete and updated list of our available environments, please visit [our list](https://ugr-sail.github.io/sinergym/compilation/main/pages/environments.html#) in the official *Sinergym* documentation.

## Installation

For more detailed information, please visit our [documentation](https://ugr-sail.github.io/sinergym/compilation/main/index.html).

### Docker container

We include a **Dockerfile** for installing all dependencies and setting
up the image for running *Sinergym*. 

By default, Dockerfile will do `pip install -e .[extras]`, if you want to install a different setup, you will have to do in root repository:

```sh
  $ docker build -t <tag_name> --build-arg SINERGYM_EXTRAS=[<setup_tag(s)>] .
```

For example, if you want a container with only documentation libraries and testing:

```sh
  $ docker build -t example1/sinergym:latest --build-arg SINERGYM_EXTRAS=[doc,test] .
```

On the other hand, if you don't want any extra library, it's necessary to write an empty value like this:

```sh
  $ docker build -t example1/sinergym:latest --build-arg SINERGYM_EXTRAS= .
```

:pencil: You can install directly our container from `Docker Hub repository <https://hub.docker.com/repository/docker/sailugr/sinergym>`__, all releases of this project are there.

:pencil: If you use [Visual Studio Code](https://code.visualstudio.com/), by simply opening the root directory and clicking on the pop-up button *Reopen in container*, all the dependencies will be installed automatically and you will be able to run *Sinergym* in an isolated environment. For more information about how to use this functionality, check [VSCode Containers extension documentation](https://code.visualstudio.com/docs/remote/containers).

### Manual installation

To install *Sinergym* manually instead of through the container (not recommended), follow these steps:

#### 1. Configure Python environment

- First, clone this repository:

```sh
  $ git clone https://github.com/ugr-sail/sinergym.git
  $ cd sinergym
```

- Then, it is recommended to create a **virtual environment**. You can do so by:

```sh
  $ sudo apt-get install python-virtualenv virtualenv
  $ virtualenv env_sinergym --python=python3.10
  $ source env_sinergym/bin/activate
  $ pip install -e .[extras]
```

- There are other alternatives like **conda environments** (recommended). Conda is very comfortable to use and we have a file to configure it automatically:

```sh
  $ cd sinergym
  $ conda env create -f python_environment.yml
  $ conda activate sinergym
```

Sinergym has been updating the compatibility with different components, here it is a summary about important versions support:

| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** |
| -------------------- | ------------------ | ------------------ | ---------------------- |
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  |
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              |
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  |
| **1.9.5**            | **22.04 LTS**      | **3.10**           | 9.5.0                  |

- Now, we have a correct python version with required modules to run *Sinergym*. Let's continue with the rest of the programs that are needed outside of Python to run the simulations:

#### 2. Install EnergyPlus 9.5.0

Install EnergyPlus. Currently it has been update compatibility to 9.5.0 and it has
been tested, but code may also work with other versions. Other combination may works, but they don't have been tested.

Follow the instructions [here](https://energyplus.net/downloads) and
install it for Linux (only Ubuntu is supported). Choose any location
to install the software. Once installed, a folder called
`Energyplus-9-5-0` should appear in the selected location.

#### 3. Install BCVTB software

Follow the instructions
[here](https://simulationresearch.lbl.gov/bcvtb/Download) for
installing BCVTB software. Another option is to copy the `bcvtb`
folder from [this repository](https://github.com/zhangzhizza/Gym-Eplus/tree/master/eplus_env/envs)

#### 4. Set environment variables

Two environment variables must be set: `EPLUS_PATH` and
`BCVTB_PATH`, with the locations where EnergyPlus and BCVTB are
installed respectively.


## About Sinergym package

As we have told you in section *Manual Installation*, Python environment can be set up using *python_environment.yml* with *conda*.
However, we can make an installation using the Github repository itself:

```sh
  $ cd sinergym
  $ pip install -e .
```

Extra libraries can be installed by typing ``pip install -e .[extras]``.
*extras* include all optional libraries which have been considered in this project such as 
testing, visualization, Deep Reinforcement Learning, monitoring , etc.
It's possible to select a subset of these libraries instead of 'extras' tag in which we select all optional libraries, for example:

```sh
  $ pip install -e .[test,doc]
```

In order to check all our tag list, visit `setup.py <https://github.com/ugr-sail/sinergym/blob/main/setup.py>`__ in *Sinergym* root repository. In any case, they are not a requirement of the package.

You can also install from `oficial pypi repository <https://pypi.org/project/sinergym/>`__ with last stable version by default:

```sh
  $ pip install sinergym[extras]
```

## Check Installation

This project is automatically supervised using tests developed specifically for it. If you want to check *Sinergym* has been installed successfully, run next command:

```sh
$ pytest tests/ -vv
```
Anyway, every time *Sinergym* repository is updated, the tests will run automatically in a remote container using the Dockerfile to build it. `Github Action <https://docs.github.com/es/actions/>`__ will do that job (see our documentation for more information).

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

Cloud Computing 

For more information about this functionality, please, visit our documentation [here](https://ugr-sail.github.io/sinergym/compilation/main/pages/gcloudAPI.html#sinergym-with-google-cloud).

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
