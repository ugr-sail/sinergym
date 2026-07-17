############
Installation
############

*Sinergym* relies on several dependencies, the specifics of which vary by version. The following table provides a summary of the versions supported by *Sinergym* across its releases:

+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** | **Building model file format**   |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  | IDF                              |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              | IDF                              |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  | IDF                              |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **1.9.5**            | **22.04 LTS**      | **3.10.6**         | 9.5.0                  | IDF                              |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **2.4.0**            | 22.04 LTS          | 3.10.6             | 9.5.0                  | **epJSON**                       |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **2.5.0**            | 22.04 LTS          | 3.10.6             | **23.1.0**             | epJSON                           |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **3.3.6**            | **24.04 LTS**      | **3.12.3**         | 23.1.0                 | epJSON                           |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **3.5.8**            | 24.04 LTS          | 3.12.3             | **24.1.0**             | epJSON                           |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+
| **3.11.0**           | 24.04 LTS          | 3.12.3             | **25.1.0**             | epJSON                           |
+----------------------+--------------------+--------------------+------------------------+----------------------------------+

.. important:: Starting from version 3.6.0, Sinergym dependencies are managed using **Poetry**, although installation using **pip** is still available.

We recommend using the latest version of *Sinergym* that is supported by the Docker container. This approach helps you to avoid the complexities of the installation process. However, if you prefer to manually install *Sinergym* on your computer, we provide the necessary documentation in the subsequent sections.

****************
Docker container
****************

We provide a **Dockerfile** to install dependencies and prepare the image for running *Sinergym*. This is the **recommended** option to set up Sinergym, since it ensures that all dependencies and versions are correctly installed and configured.

In essence, this Dockerfile installs the compatible operating system, EnergyPlus, Python, and *Sinergym*, along with the necessary dependencies for its proper functioning. Once the repository is cloned, it can be used as follows:

.. code:: sh

    $ docker build -t <tag_name> .

Sinergym has a set of optional dependencies. These dependencies can be installed in the following way when building the image:

.. code:: sh

    $ docker build -t <tag_name> --build-arg SINERGYM_EXTRAS="drl notebooks gcloud" .

These optional dependencies allow you to use ``stable-baselines3``, ``wandb``, ``notebooks`` or ``gcloud`` directly. For more information, please refer to the ``pyproject.toml`` file at the root of the repository (``[tool.poetry.extras]`` section). If you desire to install all optional packages, you can use ``extras`` directly in the ``SINERGYM_EXTRAS`` argument.

Multi-stage architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Dockerfile`` in the repository root uses a `multi-stage build <https://docs.docker.com/build/building/multi-stage/>`__ with three stages. Knowing which one you are building helps you pick the right one for your use case:

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - **Stage**
     - **What it contains**
     - **When to use it**
   * - ``builder``
     - EnergyPlus, Python venv with Poetry and all deps. Intermediate stage, source of artifacts copied into the other two.
     - Internal --- not built directly.
   * - ``dev``
     - Everything in ``builder`` plus the development dependency groups (``format``, ``typing``, ``test``, ``doc``, ``drl``, ``gcloud``, ``plots``).
     - Local development and devcontainer. Build with ``--target dev``.
   * - ``runtime``
     - Minimal image: EnergyPlus + venv with only the packages from ``SINERGYM_EXTRAS``. No Poetry, no build tools, no ``dev`` dependencies.
     - Running simulations and tests in CI. This is the default target when no ``--target`` is specified.

.. code:: sh

    # Runtime image (default) — minimal, only SINERGYM_EXTRAS packages
    $ docker build -t sinergym:runtime .

    # Dev image — adds format/typing/test/doc/drl/gcloud/plots groups
    $ docker build --target dev -t sinergym:dev .

A fourth file, ``.devcontainer/Dockerfile_lite``, does not build from source but instead pulls a pre-built image from the `Docker Hub repository <https://hub.docker.com/repository/docker/sailugr/sinergym>`__. It is useful when you want a devcontainer without paying the cost of a local build.

.. note:: Our container can also be directly installed from the 
          `Docker Hub repository <https://hub.docker.com/repository/docker/sailugr/sinergym>`__. 
          It contains all the project's releases with secondary dependencies or lite versions.

Once the container image is ready, you can execute any command as follows:

.. code:: sh

    $ docker run -it --rm <tag_name> <command>

By default, the command executed is ``python scripts/try_env.py``, which is a minimal working example.

If you want to run a sample DRL experiment, you can do it as follows:

.. code:: sh

    $ docker build -t example/sinergym:latest --build-arg SINERGYM_EXTRAS="drl" .
    $ docker run -e WANDB_API_KEY=$WANDB_API_KEY -it --rm example/sinergym:latest python scripts/train/local_confs/train_agent_local_conf.py -conf scripts/train/local_confs/conf_examples/train_agent_PPO.yaml

.. important:: The ``WANDB_API_KEY`` is **not** baked into the image at build time
               (to avoid leaking your token in ``docker history``). You must pass it
               at runtime with ``-e WANDB_API_KEY=$WANDB_API_KEY`` (or ``--env-file``)
               on every ``docker run`` that executes code using Weights & Biases. The
               same applies to the devcontainer: ``.devcontainer/devcontainer.json``
               forwards the variable from your local environment via ``containerEnv``,
               so make sure it is exported in your shell before opening the container.

If the script requires a WandB account, remember to include the environment variable in the container with the token.

It is also possible to keep a session open in the image to copy and run your own scripts. For more information, please refer to the Docker documentation. This can be useful if you want to run your own scripts within the container.

.. code:: sh

    $ docker run -it <tag_name> /bin/bash

.. note:: For `Visual Studio Code <https://code.visualstudio.com/>`__ users, 
          simply open the root directory and click on the *Reopen in container* pop-up button. 
          This action will automatically install all dependencies and enable you to run *Sinergym* 
          in an isolated environment. For more details on this feature, 
          refer to the `VSCode Containers extension documentation <https://code.visualstudio.com/docs/remote/containers>`__.

*******************
Manual installation
*******************

If you prefer not to use containers and have everything installed natively on your system, we will explain how to do so.

First, make sure that you meet the compatibility matrix; otherwise, we cannot provide support or guarantees of functionality.

Configure Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by installing the desired version of Python and pip. It is recommended to set up a working environment for Python. Finally, install the necessary dependencies of Sinergym in that environment:

.. code:: sh

    $ pip install sinergym

You can also install the optional packages from here, just like in the Docker container:

.. code:: sh

    $ pip install sinergym[extras]

If you want to install the cloned repository directly, you can do so by running the following command located in its root directory:

.. code:: sh

    $ poetry install --no-interaction --extras <optional_extras>
    # or
    $ pip install .[<optional_extras>]

You now have the correct Python version and the necessary modules to run *Sinergym*. Let us proceed with the installation of the other programs needed to run the simulations, in addition to Python.

Install EnergyPlus
~~~~~~~~~~~~~~~~~~

In order to proceed, please install **EnergyPlus**. We have **tested and confirmed compatibility** with version ``24.1.0``. While the code may be compatible with other versions, we have not tested them.

To install it for Linux (only **Ubuntu** is tested and supported), please follow the instructions `here <https://energyplus.net/downloads>`__. You can choose any location for the installation. After installation, a folder named ``Energyplus-24-1-0`` should appear in the chosen location.

Include EnergyPlus Python API in Python path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Sinergym* uses the *EnergyPlus* Python API as its backend. The modules of this API are located in the *EnergyPlus* folder that was installed in the previous step. You must add this installation path to the ``PYTHONPATH`` environment variable so that the interpreter can access these modules.

*******************
Develop in Sinergym
*******************

Whether you have chosen to use Docker or a manual installation, we offer facilities for developing and extending Sinergym.

If you have chosen the Docker container installation, Visual Studio Code will set up a development environment with all the necessary packages automatically, including documentation, tests, DRL, etc integrated in the IDE. 

The devcontainer (``.devcontainer/devcontainer.json``) targets the ``dev`` stage of the root ``Dockerfile`` (see :ref:`Multi-stage architecture` above). This means the same Dockerfile serves both development and production — there is no longer a separate ``Dockerfile`` inside ``.devcontainer``. If you are building a development image manually (without VS Code), use:

.. code:: sh

    $ docker build --target dev -t sinergym:dev .

This installs the ``dev`` Poetry group, which includes ``format``, ``typing``, ``test``, ``doc``, ``drl``, ``gcloud`` and ``plots`` (see ``pyproject.toml``). To get only a subset, drop ``--with dev`` and use ``--extras`` instead:

.. code:: dockerfile

    RUN poetry install --no-interaction --extras "test drl"

If you have manually installed the project, you can install the development packages from Poetry in the same way. Once the repository is cloned, run the same command we explained earlier, adding the following:

.. code:: sh

    $ poetry install --no-interaction --with dev

If you prefer to use pip, please note that not all development packages will be installed, since pip is intended primarily for package distribution rather than a full development environment.

.. code:: sh

    $ pip install -e .[extras]

.. note:: For additional information about how Poetry dependencies work, visit its 
          `official documentation <https://python-poetry.org/docs/dependency-specification/>`__.

*******************
Verify installation
*******************

This project is automatically monitored using **tests**. To verify that *Sinergym* has been installed correctly, execute ``pytest tests/ -vv`` in the root directory. Remember to previously install the test extra requirements to use this.

Furthermore, each time the *Sinergym* repository is updated, the tests are automatically executed in a remote container built using the Dockerfile. This task is performed by `Github Action <https://docs.github.com/es/actions/>`__ (refer to the :ref:`Github Actions` section for additional details).

***************
Cloud computing
***************

We include the option to run your experiments using `Google Cloud <https://cloud.google.com/>`__. For more information on installation and preparing the Google Cloud SDK to run your experiments, please visit the :ref:`Google Cloud configuration` section.