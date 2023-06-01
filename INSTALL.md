For more detailed information, please visit our [documentation](https://ugr-sail.github.io/sinergym/compilation/main/index.html).

## Docker container (recommended)

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

## Manual installation

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

| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** | **Building model file** |
| -------------------- | ------------------ | ------------------ | ---------------------- | ----------------------- |
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  | IDF                     |
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              | IDF                     |
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  | IDF                     |
| **1.9.5**            | **22.04 LTS**      | **3.10**           | 9.5.0                  | IDF                     |
| **2.4.0**            | 22.04 LTS          | 3.10.6             | 9.5.0                  | **epJSON**              |
| **2.5.0**            | 22.04 LTS          | 3.10.6             | **23.1.0**             | epJSON                  |


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
installing BCVTB software.

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