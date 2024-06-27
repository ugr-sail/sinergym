############
Installation
############

*Sinergym* relies on several dependencies, the specifics of which vary by version. 
The table below provides a summary of the versions supported by *Sinergym* across its various releases:

+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **Sinergym version** | **Ubuntu version** | **Python version** | **EnergyPlus version** | **Building model file**   |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **0.0**              | 18.04 LTS          | 3.6                | 8.3.0                  | IDF                       |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **1.1.0**            | 18.04 LTS          | 3.6                | **9.5.0**              | IDF                       |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **1.7.0**            | 18.04 LTS          | **3.9**            | 9.5.0                  | IDF                       |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **1.9.5**            | **22.04 LTS**      | **3.10.6**         | 9.5.0                  | IDF                       |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **2.4.0**            | 22.04 LTS          | 3.10.6             | 9.5.0                  | **epJSON**                |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **2.5.0**            | 22.04 LTS          | 3.10.6             | **23.1.0**             | epJSON                    |
+----------------------+--------------------+--------------------+------------------------+---------------------------+
| **3.3.6**            | **24.04 LTS**      | **3.12.3**         | 23.1.0                 | epJSON                    |
+----------------------+--------------------+--------------------+------------------------+---------------------------+

We recommend always using the latest version of *Sinergym* that is supported by the container. 
This approach helps you avoid the complexities of the installation process. However, 
if you prefer to manually install *Sinergym* on your computer, we provide the necessary 
documentation in the subsequent sections.

****************
Docker container
****************

We provide a **Dockerfile** to install all dependencies and prepare the 
image for running *Sinergym*. 

By default, the *Dockerfile* executes ``pip install -e .[extras]``. If you wish 
to install a different setup, you need to execute the following command in the 
**root repository**:

.. code:: sh

    $ docker build -t <tag_name> --build-arg SINERGYM_EXTRAS=[<setup_tag(s)>] .

For instance, to create a container with only the documentation libraries 
and testing, use:

.. code:: sh

    $ docker build -t example1/sinergym:latest --build-arg SINERGYM_EXTRAS=[doc,test] .

If you do not require any extra libraries, specify an empty value as follows:

.. code:: sh

    $ docker build -t example1/sinergym:latest --build-arg SINERGYM_EXTRAS= .

.. note:: Our container can also be directly installed from the 
          `Docker Hub repository <https://hub.docker.com/repository/docker/sailugr/sinergym>`__. 
          It contains all the project's releases.

.. note:: For `Visual Studio Code <https://code.visualstudio.com/>`__ users, 
          simply open the root directory and click on the *Reopen in container* pop-up button. 
          This action will automatically install all dependencies and enable you to run *Sinergym* 
          in an isolated environment. For more details on this feature, 
          refer to the `VSCode Containers extension documentation <https://code.visualstudio.com/docs/remote/containers>`__.

*******************
Manual installation
*******************

To manually install *Sinergym* (although using the container is **recommended**), 
follow these steps:

Configure Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Begin by cloning this repository:

.. code:: sh

    $ git clone https://github.com/ugr-sail/sinergym.git
    $ cd sinergym

* Next, we recommend creating a **virtual environment** as follows:

.. code:: sh

    $ sudo apt-get install python-virtualenv virtualenv
    $ virtualenv env_sinergym --python=python3.10
    $ source env_sinergym/bin/activate
    $ pip install -e .[extras]

Alternatively, you can use **conda environments** (*recommended*). 
*Conda* is user-friendly and we provide a file for automatic configuration:

.. code:: sh
    
        $ cd sinergym
        $ conda env create -f python_environment.yml
        $ conda activate sinergym

With this, you have the correct Python version and the necessary modules to run 
*Sinergym*. Let's proceed with the installation of other required programs 
outside of Python to run the simulations:

Install EnergyPlus 23.1.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to install *EnergyPlus*. We have tested and confirmed compatibility 
with version ``23.1.0``. The code might work with other versions, but we 
have not tested them.

Follow the instructions `here <https://energyplus.net/downloads>`__ to install 
it for Linux (we only support **Ubuntu**). You can choose any location for the 
installation. After installation, a folder named ``Energyplus-23-1-0`` should 
appear in the chosen location.

Include Energyplus Python API in Python Path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Sinergym* uses the *Energyplus* Python API as its backend. The modules of this 
API are located in the *Energyplus* folder that you installed in the previous 
step. You must add this installation path to the ``PYTHONPATH`` environment 
variable so that the interpreter can access these modules.

***********************
About Sinergym package
***********************

As we have told you in section :ref:`Manual installation`, Python environment 
can be set up using ``python_environment.yml`` with *conda*. This will install 
the virtual environment with Python version required and all packages used 
*all-in-one*.
However, we can make an installation using the Github repository in a python 
environment directly, like we have shown with *virtualenv*:

.. code:: sh

    $ source env_sinergym/bin/activate
    $ cd sinergym
    $ pip install -e .

Extra libraries can be installed by typing ``pip install -e .[extras]``.
*extras* include all optional libraries which have been considered in this project such as 
testing, visualization, Deep Reinforcement Learning, monitoring , etc.
It's possible to select a subset of these libraries instead of 'extras' tag in which 
we select all optional libraries, for example:

.. code:: sh

    $ cd sinergym
    $ pip install -e .[test,doc]

In order to check all our tag list, visit `setup.py <https://github.com/ugr-sail/sinergym/blob/main/setup.py>`__ 
in *Sinergym* root repository. In any case, they are not a requirement of the package.

You can also install from `official PyPi repository <https://pypi.org/project/sinergym/>`__ 
with last stable version by default:

.. code:: sh

    $ pip install sinergym[extras]

*******************
Verify Installation
*******************

This project is automatically monitored using **tests** specifically developed for it. 
To verify that *Sinergym* has been installed correctly, execute ``pytest tests/ -vv`` 
in the **repository root**.

Furthermore, each time the *Sinergym* repository is updated, the tests are automatically executed in a remote container 
built using the Dockerfile. This task is performed by `Github Action <https://docs.github.com/es/actions/>`__ 
(refer to the :ref:`Github Actions` section for more details).

****************
Cloud Computing
****************

You also have the option to run your experiments in the Cloud. We utilize `Google Cloud <https://cloud.google.com/>`__ 
for this purpose. Our team is working on setting up an account to run our *Sinergym* container 
with **remote storage** and **Weights&Biases tracking**.
For more information on installation and preparing the Google Cloud SDK to run your experiments, 
please visit our :ref:`Preparing Google Cloud` section.