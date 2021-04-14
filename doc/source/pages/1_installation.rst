############
Installation
############

To install *energym*, follow these steps:

* First, it is recommended to create a virtual environment. You can do so by:

.. code:: sh

    $ sudo apt-get install python-virtualenv virtualenv
    $ virtualenv env_energym --python=python3.7
    $ source env_energym/bin/activate

* Then, clone this repository using this command:

.. code:: sh

    $ git clone https://github.com/jajimer/energym.git

****************
Docker container
****************

We include a **Dockerfile** for installing all dependencies and setting
up the image for running *energym*. If you use `Visual Studio
Code <https://code.visualstudio.com/>`__, by simply opening the root
directory and clicking on the pop-up button "*Reopen in container*\ ",
dependencies will be installed automatically and you will be able to run
*energym* in an isolated environment.

*******************
Manual installation
*******************

If you prefer installing *energym* manually, follow the steps below:

1. Install EnergyPlus 8.6.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, install EnergyPlus. Currently only version 8.6.0 has
been tested, but code may also work with other versions.

Follow the instructions `here <https://energyplus.net/downloads>`__ and
install it for Linux (only Ubuntu is supported). Choose any location
to install the software. Once installed, a folder called
``Energyplus-8-6-0`` should appear in the selected location.

2. Install BCVTB software
~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the instructions
`here <https://simulationresearch.lbl.gov/bcvtb/Download>`__ for
installing BCVTB software. Another option is to copy the ``bcvtb``
folder from `this
repository <https://github.com/zhangzhizza/Gym-Eplus/tree/master/eplus_env/envs>`__.

3. Set environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two environment variables must be set: ``EPLUS_PATH`` and
``BCVTB_PATH``, with the locations where EnergyPlus and BCVTB are
installed respectively.

4. Install the package
~~~~~~~~~~~~~~~~~~~~~~

Finally, *energym* can be installed by running this command at the repo
root folder:

.. code:: sh

    $ pip install -e .

Extra libraries can be installed by typing ``pip install -e .[extras]``.
They are intended for running and analysing DRL algorithms over *energym*,
but they are not a requirement of the package.

And that's all!
