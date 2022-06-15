# sinergym

<div align="center">
  <img src="images/logo.png" width=40%><br><br>
</div>

**Welcome to sinergym!**

This is a project based on Zhiang Zhang and Khee Poh Lam [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus).

The goal of this project is to create an environment following OpenAI Gym interface for wrapping simulation engines for building control using **deep reinforcement learning**.

The main functionalities of Sinergym are the following :

  - **Benchmark environments**. Similarly to Atari or Mujoco environments for RL community, we are designing a set of environments for benchmarking and testing deep RL algorithms. These environments may include different buildings, weathers or action/observation spaces.
  - **Develop different experimental settings**. We aim to provide a package that allows to modify experimental settings in an easy manner. For example, several reward functions or observation variables may be defined. 
  - **Include different simulation engines**. Communication between Python and [EnergyPlus](https://energyplus.net/) is established using [BCVTB](https://simulationresearch.lbl.gov/bcvtb/FrontPage). Since this tool allows for interacting with several simulation engines, more of them (e.g. [OpenModelica](https://openmodelica.org/)) could be included in the backend while maintaining the Gym API.
  -  **Building Models configuration automatically**: Building models will be
   adapted to specification of each simulation. For example, *Designdays* and 
   *Location* from IDF files will be adapted to weather file specified in
   Sinergym simulator backend without any intervention by the user.
  -  **Extra configuration facilities**: Our team aim to provide extra parameters
   in order to amplify the context space for the experiments with this tool.
   Sinergym will modify building model automatically based on parameters set.
   For example: People occupant, timesteps per simulation hour, observation
   and action spaces, etc.
  -  **Stable Baseline 3 Integration**. Some functionalities like callbacks
   have been developed by our team in order to test easily these environments
   with deep reinforcement learning algorithms.
  -  **Google Cloud Integration**. Whether you have a Google Cloud account and you want to
   use your infrastructure with Sinergym, we tell you some details about how doing it.
  - **Mlflow tracking server**. [Mlflow](https://mlflow.org/) is an open source platform for the machine
   learning lifecycle. This can be used with Google Cloud remote server (if you have Google Cloud account) 
   or using local store. This will help you to manage and store your runs and artifacts generated in an orderly
   manner.
  -  **Data Visualization**. Using Sinergym logger or Tensorboard server to visualize training information
   in real-time.
  - Many more!

_This is a work in progress project. Stay tuned for upcoming releases._

<div align="center">
  <img src="images/operation_diagram.jpg"><br><br>
</div>

## List of available environments

If you would like to see a complete and updated list of our available environments, please visit [our list](https://ugr-sail.github.io/sinergym/compilation/html/pages/environments.html) in the official Sinergym documentation.

## Installation

For more detailed information, please visit our [documentation](https://ugr-sail.github.io/sinergym/compilation/html/index.html).

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

:pencil: You can install directly our container from `Docker Hub repository <https://hub.docker.com/repository/docker/alejandrocn7/sinergym>`__, all releases of this project are there.

:pencil: If you use [Visual Studio Code](https://code.visualstudio.com/), by simply opening the root directory and clicking on the pop-up button *Reopen in container*, all the dependencies will be installed automatically and you will be able to run *Sinergym* in an isolated environment. For more information about how to use this functionality, check [VSCode Containers extension documentation](https://code.visualstudio.com/docs/remote/containers).

### Manual installation

To install *Sinergym* manually instead of through the container (recommended), follow these steps:

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

- Now, we have a correct python version with required modules to run sinergym. Let's continue with the rest of the programs that are needed outside of Python to run the simulations:

#### 2. Install EnergyPlus 9.5.0

Install EnergyPlus. Currently it has been update compatibility to 9.5.0 and it has
been tested, but code may also work with other versions. Sinergym ensure this support:

| Sinergym Version | EnergyPlus version |
| :--------------: | :----------------: |
| 1.0.0 or before  |       8.6.0        |
|  1.1.0 or later  |       9.5.0        |

Other combination may works, but they don't have been tested.

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

In order to check all our tag list, visit `setup.py <https://github.com/ugr-sail/sinergym/blob/main/setup.py>`__ in Sinergym root repository. In any case, they are not a requirement of the package.

You can also install from `oficial pypi repository <https://pypi.org/project/sinergym/>`__ with last stable version by default:

```sh
  $ pip install sinergym[extras]
```

## Check Installation

This project is automatically supervised using tests developed specifically for it. If you want to check sinergym has been installed successfully, run next command:

```sh
$ pytest tests/ -vv
```
Anyway, every time sinergym repository is updated, the tests will run automatically in a remote container using the Dockerfile to build it. `Github Action <https://docs.github.com/es/actions/>`__ will do that job (see our documentation for more information).

## Usage example

If you used our Dockerfile during installation, you should have the *try_env.py* file in your workspace as soon as you enter in. In case you have installed everything on your local machine directly, place it inside our cloned repository. In any case, we start from the point that you have at your disposal a terminal with the appropriate python version and Sinergym running correctly.

Sinergym uses the standard OpenAI gym API. So basic loop should be something like:

```python

import gym
import sinergym
# Create the environment
env = gym.make('Eplus-datacenter-mixed-continuous-stochastic-v1')
# Initialize the episode
obs = env.reset()
done = False
R = 0.0
while not done:
    a = env.action_space.sample() # random action selection
    obs, reward, done, info = env.step(a) # get new observation and reward
    R += reward
print('Total reward for the episode: %.4f' % R)
env.close()
```

Notice that a folder will be created in the working directory after creating the environment. It will contain the EnergyPlus outputs produced during the simulation.

:pencil: For more examples and details, please visit our [usage examples](https://ugr-sail.github.io/sinergym/compilation/html/pages/usage-example.html) documentation section

## Google Cloud Platform support

Cloud Computing 

For more information about this functionality, please, visit our documentation [here](https://ugr-sail.github.io/sinergym/compilation/html/pages/gcloudAPI.html).

## Citing Sinergym

If you use Sinergym in your work, please cite our [paper](https://dl.acm.org/doi/abs/10.1145/3486611.3488729):

```bibtex
@inproceedings{jimenez2021sinergym,
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
