For more detailed information, please visit our [documentation](https://ugr-sail.github.io/sinergym/compilation/main/index.html).

# Installation

*Sinergym* relies on several dependencies, the specifics of which vary by version. 
The table below provides a summary of the versions supported by *Sinergym* across its releases:

| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** | **Building model file**   |
|----------------------|--------------------|--------------------|------------------------|---------------------------|
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  | IDF                       |
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              | IDF                       |
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  | IDF                       |
| **1.9.5**            | **22.04 LTS**      | **3.10.6**         | 9.5.0                  | IDF                       |
| **2.4.0**            | 22.04 LTS          | 3.10.6             | 9.5.0                  | **epJSON**                |
| **2.5.0**            | 22.04 LTS          | 3.10.6             | **23.1.0**             | epJSON                    |
| **3.3.6**            | **24.04 LTS**      | **3.12.3**         | 23.1.0                 | epJSON                    |
| **3.5.8**            | 24.04 LTS          | 3.12.3             | **24.1.0**             | epJSON                    |


We recommend always using the latest version of *Sinergym* that is supported by the container. 
This approach helps you avoid the complexities of the installation process. However, 
if you prefer to manually install *Sinergym* on your computer, we provide the necessary 
documentation in the subsequent sections.

## Docker container

We provide a **Dockerfile** to install all dependencies and prepare the 
image for running *Sinergym*. This is the **recommended** option, since it
ensures that all dependencies and versions are correctly installed and configured.

In essence, this Dockerfile installs the compatible operating system, EnergyPlus, 
Python, and **Sinergym**, along with the necessary dependencies for its proper functioning. 
If you have cloned the repository, it can be used as follows:

```bash
$ docker build -t <tag_name> .
```

However, Sinergym has a set of optional dependencies that enhance the tool's usage. These 
dependencies can be installed in the following way when building the image:

```bash
$ docker build -t <tag_name> --build-arg SINERGYM_EXTRAS=format,test,doc,DRL,gcloud .
```

These optional dependencies allow you to format code, run tests, generate documentation, 
have pre-existing DRL algorithms available, etc. For more information, please refer to 
the `pyproject.toml` file at the root of the repository. If you desire to install all optional
packages, you can use `dev` directly in the `SINERGYM_EXTRAS` argument.

> :memo: **Note:** Our container can also be directly installed from the [Docker Hub repository](https://hub.docker.com/repository/docker/sailugr/sinergym). It contains all the project's releases with secondary dependencies or lite versions.

Once the container image is ready, you can execute any command as follows:

```bash
$ docker run -it --rm <tag_name> <command>
```

By default, the command executed is `python scripts/try_env.py`, which is a minimal working example.

If you want to run a DRL experiment, for example, you can do it like this:

```bash
$ docker build -t example/sinergym:latest --build-arg SINERGYM_EXTRAS=DRL,platforms .
$ docker run -e WANDB_API_KEY=$WANDB_API_KEY -it --rm example/sinergym:latest python scripts/train/train_agent.py -conf scripts/train/train_agent_PPO.json
```

If the script you want to use requires a WandB account, remember to include the environment variable 
in the container with the token.

It’s also possible to keep a session open in the image to copy and run your own scripts. For more 
information, please refer to the Docker documentation. This can be useful when you want to run your 
own scripts in the container.

```bash
$ docker run -it <tag_name> /bin/bash
```

> :memo: **Note:** For [Visual Studio Code](https://code.visualstudio.com/) users, simply open the root directory and click on the *Reopen in container* pop-up button. This action will automatically install all dependencies and enable you to run *Sinergym* in an isolated environment. For more details on this feature, refer to the [VSCode Containers extension documentation](https://code.visualstudio.com/docs/remote/containers).

## Manual installation

If you prefer not to use containers and have everything installed natively on your system, we’ll explain 
how to do it.

First, make sure that you meet the compatibility matrix; otherwise, no support or guarantees of 
functionality will be provided by the development team.

### Configure Python Environment

Start by installing the desired version of Python and pip. It is recommended to set up a working 
environment for Python (multiple options are available). Finally, install the necessary 
dependencies for Sinergym in that environment:

```sh
$ pip install sinergym
```

You can also install the optional packages from here, just like in the Docker container:

```sh
$ pip install sinergym[format,test,doc,DRL,gcloud]
```

If you want to install the cloned repository directly, you can do so by running the following 
command in its root.

```sh
$ poetry install --with format,test,doc,DRL,gcloud
```

With this, you have the correct Python version and the necessary modules to run 
*Sinergym*. Let's proceed with the installation of other required programs 
outside of Python to run the simulations.

### Install EnergyPlus 24.1.0

You need to install *EnergyPlus*. We have tested and confirmed compatibility 
with version `24.1.0`. The code might work with other versions, but we 
have not tested them.

Follow the instructions [here](https://energyplus.net/downloads) to install 
it for Linux (we only support **Ubuntu**). You can choose any location for the 
installation. After installation, a folder named `Energyplus-24-1-0` should 
appear in the chosen location.

### Include Energyplus Python API in Python Path

*Sinergym* uses the *Energyplus* Python API as its backend. The modules of this 
API are located in the *Energyplus* folder that you installed in the previous 
step. You must add this installation path to the `PYTHONPATH` environment 
variable so that the interpreter can access these modules.


## Verify Installation

This project is automatically monitored using **tests** specifically developed for it. 
To verify that *Sinergym* has been installed correctly, execute `pytest tests/ -vv` 
in the **repository root**.

Furthermore, each time the *Sinergym* repository is updated, the tests are automatically executed in a remote container 
built using the Dockerfile. This task is performed by [Github Action](https://docs.github.com/es/actions/) 
(refer to the `Github Actions` section for more details).

## Cloud Computing

You also have the option to run your experiments in the Cloud. We utilize [Google Cloud](https://cloud.google.com/) 
for this purpose. Our team is working on setting up an account to run our *Sinergym* container 
with **remote storage** and **Weights&Biases tracking**.
For more information on installation and preparing the Google Cloud SDK to run your experiments, 
please visit our `Preparing Google Cloud` section.