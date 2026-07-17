For detailed information, please refer to the [documentation](https://ugr-sail.github.io/sinergym/compilation/main/index.html).

# Installation

*Sinergym* relies on several dependencies, the specifics of which vary by version.

The table below provides a summary of the dependencies versions supported by *Sinergym* across its releases:

| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** | **Building model file** |
| -------------------- | ------------------ | ------------------ | ---------------------- | ----------------------- |
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  | IDF                     |
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              | IDF                     |
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  | IDF                     |
| **1.9.5**            | **22.04 LTS**      | **3.10.6**         | 9.5.0                  | IDF                     |
| **2.4.0**            | 22.04 LTS          | 3.10.6             | 9.5.0                  | **epJSON**              |
| **2.5.0**            | 22.04 LTS          | 3.10.6             | **23.1.0**             | epJSON                  |
| **3.3.6**            | **24.04 LTS**      | **3.12.3**         | 23.1.0                 | epJSON                  |
| **3.5.8**            | 24.04 LTS          | 3.12.3             | **24.1.0**             | epJSON                  |
| **3.11.0**           | 24.04 LTS          | 3.12.3             | **25.1.0**             | epJSON                  |


We recommend to always use the latest version of *Sinergym* that is supported by the container. This will help you to avoid the complexities of manual installation.
However, if you prefer to manually install *Sinergym* on your computer, we provide the necessary documentation in the subsequent sections.

## Docker container

We provide a **Dockerfile** to install all dependencies and prepare the image for running *Sinergym*. This is the **recommended** option, since it
ensures that all dependencies and versions are correctly installed and configured.

This Dockerfile installs the compatible operating system, EnergyPlus, Python, and *Sinergym*, along with the necessary dependencies for its proper functioning. 

If you have cloned the repository, run the following command:

```bash
$ docker build -t <tag_name> .
```

*Sinergym* has a set of optional dependencies that enhance its usage. These dependencies can be installed in the following way when building the image:

```bash
$ docker build -t <tag_name> --build-arg SINERGYM_EXTRAS="drl notebooks gcloud" .
```

These optional dependencies allow you to use `stable-baselines3`, `wandb`, `notebooks` and `gcloud`. For more information, please refer to the `pyproject.toml` file at the root of the repository (``[tool.poetry.extras]`` section). 

If you want to install all optional packages, use `extras` in the `SINERGYM_EXTRAS` argument.

### Multi-stage architecture

The `Dockerfile` in the repository root uses a [multi-stage build](https://docs.docker.com/build/building/multi-stage/) with three stages. Knowing which one you are building helps you pick the right one for your use case:

| **Stage**   | **What it contains** | **When to use it** |
| ----------- | -------------------- | ----------------- |
| `builder`   | EnergyPlus, Python venv with Poetry and all deps. Intermediate stage, source of artifacts copied into the other two. | Internal — not built directly. |
| `dev`       | Everything in `builder` plus the development dependency groups (`format`, `typing`, `test`, `doc`, `drl`, `gcloud`, `plots`). | Local development and devcontainer. Build with `--target dev`. |
| `runtime`   | Minimal image: EnergyPlus + venv with only the packages from `SINERGYM_EXTRAS`. No Poetry, no build tools, no `dev` dependencies. | Running simulations and tests in CI. This is the **default** target when no `--target` is specified. |

```bash
# Runtime image (default) — minimal, only SINERGYM_EXTRAS packages
$ docker build -t sinergym:runtime .

# Dev image — adds format/typing/test/doc/drl/gcloud/plots groups
$ docker build --target dev -t sinergym:dev .
```

A fourth file, `.devcontainer/Dockerfile_lite`, does not build from source but instead pulls a pre-built image from the [Docker Hub repository](https://hub.docker.com/repository/docker/sailugr/sinergym). It is useful when you want a devcontainer without paying the cost of a local build.

> :memo: **Note:** the container can also be directly installed from the [Docker Hub repository](https://hub.docker.com/repository/docker/sailugr/sinergym). It contains all the project's releases with secondary dependencies or lite versions.

Once the container image is ready, you can execute any command as follows:

```bash
$ docker run -it --rm <tag_name> <command>
```

By default, the command executed is `python scripts/try_env.py`, which is a minimal working example.

If you want to run a DRL experiment, for example, you can do it like this:

```bash
$ docker build -t example/sinergym:latest --build-arg SINERGYM_EXTRAS="drl" .
$ docker run -e WANDB_API_KEY=$WANDB_API_KEY -it --rm example/sinergym:latest python scripts/train/local_confs/train_agent_local_conf.py -conf scripts/train/local_confs/conf_examples/train_agent_PPO.yaml
```

> :warning: **Important:** The `WANDB_API_KEY` is **not** baked into the image at build time
> (to avoid leaking your token in `docker history`). You must pass it
> at runtime with `-e WANDB_API_KEY=$WANDB_API_KEY` (or `--env-file`)
> on every `docker run` that executes code using Weights & Biases. The
> same applies to the devcontainer: `.devcontainer/devcontainer.json`
> forwards the variable from your local environment via `containerEnv`,
> so make sure it is exported in your shell before opening the container.

If the script you want to use requires a Weights and Biases account, remember to include the corresponding API token in a container environment variable.

It is also possible to keep an open session in the image. For more information, please refer to the official Docker documentation. This may help to run your own scripts in the container.

```bash
$ docker run -it <tag_name> /bin/bash
```

> :memo: **Note:** For [Visual Studio Code](https://code.visualstudio.com/) users, simply open the root directory and click the *Reopen in container* pop-up button. This will automatically install all dependencies and enable you to run *Sinergym* in an isolated environment. For more details about this feature, refer to the [VSCode Containers extension documentation](https://code.visualstudio.com/docs/remote/containers).

## Manual installation

If you prefer not to use containers and install it natively on your system, we provide some guidance on how to do it.

First, ensure that your system meets the previously specified software compatibility requirements. Without this, we cannot provide support or guarantees of functionality.

### Configure Python Environment

Start by installing the desired version of Python and *pip*. It is recommended to set up a working environment for Python. Finally, install the necessary dependencies of *Sinergym* in that environment:

```sh
$ pip install sinergym
```

You can also install the optional packages by running:

```sh
$ pip install sinergym[extras]
```

To directly install *Sinergym* from the cloned repository, run:

```sh
$ poetry install --no-interaction --extras <optional_extras>
# or
$ pip install .[<optional_extras>]
```

Now the correct Python version and the necessary modules to run *Sinergym* will be installed.

Let's now proceed with the installation of external software.

### Install EnergyPlus

We have tested and confirmed compatibility with **EnergyPlus** version `24.1.0`. *Sinergym* might not work with other non-tested versions.

Follow the instructions detailed [here](https://energyplus.net/downloads) to install it in Linux (we only guarantee proper testing and support for **Ubuntu**). 

After installation, the folder `Energyplus-24-1-0` should appear in the selected location.

### Include EnergyPlus Python API in Python path

*Sinergym* relies on the *Energyplus* Python API. The modules of this API are located in the *EnergyPlus* folder that was created in the previous step. You must add this installation path to the `PYTHONPATH` environment variable so that the interpreter can access these modules.

## Develop in Sinergym

Whether you have chosen to use Docker or a manual installation, we offer facilities for developing and contributing to *Sinergym*.

- If you have used a container, Visual Studio Code will set up a development environment with all the necessary packages automatically configured, including documentation, tests, etc.

The devcontainer (`.devcontainer/devcontainer.json`) targets the `dev` stage of the root `Dockerfile` (see [Multi-stage architecture](#multi-stage-architecture) above). This means the same Dockerfile serves both development and production — there is no longer a separate `Dockerfile` inside `.devcontainer`. If you are building a development image manually (without VS Code), use:

```bash
$ docker build --target dev -t sinergym:dev .
```

This installs the `dev` Poetry group, which includes `format`, `typing`, `test`, `doc`, `drl`, `gcloud` and `plots` (see `pyproject.toml`). To get only a subset, drop `--with dev` and use `--extras` instead:

```dockerfile
RUN poetry install --no-interaction --extras "test drl"
```

If you have manually installed the project, you can install the development packages from **poetry** in the same way. Once the repository is cloned, run:

```sh
$ poetry install --no-interaction --with dev
```

If you prefer to use pip, please note that not all development packages will be installed, since pip is intended primarily for package distribution rather than a full development environment.

```sh
$ pip install -e .[extras]
```

> :memo: For more information about how poetry dependencies work, visit its [official documentation](https://python-poetry.org/docs/dependency-specification/).


## Verify Installation

This project is automatically monitored using **tests**. To verify that *Sinergym* has been installed correctly, execute `pytest tests/ -vv` from the repository root directory. Tests dependencies should have been previously installed.

Each time the *Sinergym* repository is updated, the tests are automatically executed in a remote container built using the Dockerfile. This task is performed via [Github actions](https://docs.github.com/es/actions/) (refer to the `Github actions` documentation section for additional details).

## Cloud Computing

We provide some features to execute experiments in [Google Cloud](https://cloud.google.com/). For more information visit the `Google Cloud integration` section of the documentation.