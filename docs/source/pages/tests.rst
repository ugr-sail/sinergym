#####
Tests
#####

This project is automatically tested using `pytest <https://docs.pytest.org/en/6.2.x/>`__.

To check if *Sinergym* is installed successfully, run the following command:

.. code:: sh

  $ pytest tests/ -vv

Whenever the *Sinergym* repository is updated, the tests will automatically run in a remote container using the Dockerfile. This is done through `Github actions <https://docs.github.com/en/actions/>`__.

.. note:: Refer to the `.github/workflows YML files <https://github.com/ugr-sail/sinergym/tree/develop/.github/workflows>`__ 
          and the :ref:`Github actions` section for further details.

**************
Install pytest
**************

The **pytest** package is included by default in the project requirements. Alternatively, it can be installed manually as follows:

.. code:: sh

  $ pip install pytest


*************
Running tests
*************

To run all the tests within the ``tests/`` directory, use the following command:

.. code:: sh

  $ pytest tests/ -vv


To run tests for a specific module, such as ``test_common.py``, use the following command:

.. code:: sh

  $ pytest tests/test_common.py -vv

***************
Available tests
***************

Tests are available in the ``sinergym/tests`` directory. They are organized by different modules:

- ``test_common.py``. Tests the ``sinergym/sinergym/utils/common.py`` module. These tests check the functionalities of *Sinergym* common utils.

- ``test_reward.py``. Tests for ``sinergym/sinergym/utils/rewards.py``. These tests check the implementation of the defined reward functions.

- ``test_wrapper.py``. Test the wrappers defined in ``sinergym/sinergym/utils/wrappers.py``.

- ``test_simulator.py``. Tests for ``sinergym/sinergym/simulators/``. These tests check the low-level communication interface with the simulators.

- ``test_env.py``. Tests for ``sinergym/sinergym/envs/``. These tests check *Sinergym*'s' Gymnasium environments.

- ``test_controller.py``. Tests the controllers defined in ``sinergym/sinergym/utils/controllers.py``.

- ``test_modeling.py``. Tests for ``sinergym/sinergym/config/modeling.py``. These tests check the simulator configuration, including epJSON and EPW Python modules.

- ``test_stable_baselines.py``. Tests related to Stable Baselines 3 integration. These tests check the compatibility of 
  *Sinergym* simulation environments with Stable Baselines 3 algorithms. If Stable Baselines 3 package is not installed, these tests will be ignored by *Sinergym*.

.. warning:: Tests about :ref:`WandBLogger` are not included in the test suite, since an account is required to run them.

.. note:: To **create new tests**, you can add them to the existing modules or create a new one if required.
