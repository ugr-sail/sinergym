############
Tests
############

This project is automatically supervised using specific tests. To check if *Sinergym* is installed successfully, 
run the following command:

.. code:: sh

  $ pytest tests/ -vv

Whenever the *Sinergym* repository is updated, the tests will automatically run in a remote container 
using the Dockerfile. This is done through `Github Actions <https://docs.github.com/en/actions/>`__.

.. note:: Refer to the `.github/workflows YML files <https://github.com/ugr-sail/sinergym/tree/develop/.github/workflows>`__ 
          and the :ref:`Github Actions` section for more details.

These tests are executed using the `pytest <https://docs.pytest.org/en/6.2.x/>`__ framework, which allows 
for writing small tests that can scale to support complex functional testing for applications and libraries.

****************
Install Pytest
****************

To install Pytest independently, use the following command, although it is already included in the project requirements:

.. code:: sh

  $ pip install pytest


****************
Running tests
****************

To run all tests within the ``tests/`` directory, use the following command:

.. code:: sh

  $ pytest tests/ -vv


To run tests for a specific module, such as ``test_common.py``, use the following command:

.. code:: sh

  $ pytest tests/test_common.py -vv


****************
Create new tests
****************

New tests have been created in the `sinergym/tests` directory. They are organized by different modules:

- **test_common.py**: Tests for `sinergym/sinergym/utils/common.py`. These tests check the functionalities 
  of *Sinergym* common utils.

- **test_reward.py**: Tests for `sinergym/sinergym/utils/rewards.py`. These tests check the implementation 
  of reward functions applicable to sinergym environments.

- **test_wrapper.py**: Tests for `sinergym/sinergym/utils/wrappers.py`. These tests check the wrappers 
  used to normalize *Sinergym* default environment observations.

- **test_simulator.py**: Tests for `sinergym/sinergym/simulators/*`. These tests check the low-level 
  *Sinergym* simulator and communication interface.

- **test_env.py**: Tests for `sinergym/sinergym/envs/*`. These tests check the *Sinergym* simulation 
  environments based on Gymnasium.

- **test_controller.py**: Tests for `sinergym/sinergym/utils/controllers.py`. These tests check the 
  agent controller, such as the Rule-Based-Controller.

- **test_modeling.py**: Tests for `sinergym/sinergym/config/modeling.py`. These tests check the simulator 
  configuration, including epJSON and EPW Python models.

- **test_stable_baselines.py**: Tests for `Stable Baselines 3`. These tests check the compatibility of 
  *Sinergym* simulation environments with Stable Baselines 3 algorithms. If Stable Baselines 3 package is not installed, these tests will be ignored by *Sinergym*.

If you want to create new tests, you can append them to these modules or create a new one if the conceptual 
context is different.

For more information about *Sinergym* tests, please refer to our repository README.
