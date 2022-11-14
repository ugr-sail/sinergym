############
Tests
############

This project is automatically supervised using tests developed specifically for it. If you want to check sinergym has been installed successfully, run next command:

.. code:: sh

    $ pytest tests/ -vv

Anyway, every time sinergym repository is updated, the tests will run automatically in a remote container using the Dockerfile to build it. `Github Action <https://docs.github.com/es/actions/>`__ will do that job.

.. note:: See `.github/workflows YML files <https://github.com/jajimer/sinergym/tree/develop/.github/workflows>`__ and :ref:`Github Actions` section for more information.

These tests running under `pytest <https://docs.pytest.org/en/6.2.x/>`__ framework which makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.

****************
Install Pytest
****************

This project has already established this dependency if you have installed *extras* libraries or *test* library specifically (see :ref:`4. Install the package`). However, to install it independently:

.. code:: sh

    $ pip install pytest


****************
Running tests
****************

In order to run current tests:

.. code:: sh

    $ pytest tests/ -vv


This runs all tests within tests/ directory. If you want verbose use `-v` or `-vv` option. To run an unique module tests, for example: 

.. code:: sh

    $ pytest tests/test_common.py -vv


****************
Create new tests
****************

These tests have been created in `sinergym/tests <https://github.com/jajimer/sinergym/tree/main/tests>`__ directory, they are organized by different modules:

- **test_common.py**: Tests for `sinergym/sinergym/utils/common.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/common.py>`__. Here will be all tests that check Sinergym common utils functionalities. 
- **test_reward.py**: Tests for `sinergym/sinergym/utils/rewards.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/rewards.py>`__. Here will be all tests that check implementation(s) of reward function(s) applicable to sinergym environments.
- **test_wrapper.py**: Tests for `sinergym/sinergym/utils/wrappers.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/wrappers.py>`__. Here will be all tests that check wrappers to normalize Sinergym default environment observations.
- **test_simulator.py**: Tests for `sinergym/sinergym/simulators/\* <https://github.com/jajimer/sinergym/tree/main/sinergym/simulators>`__. Here will be all tests that check low level Sinergym simulator and communication interface.
- **test_config.py**: Tests for `sinergym/sinergym/utils/config.py <https://github.com/jajimer/sinergym/tree/main/sinergym/utils/config.py>`__. Here will be all tests that check python building model, weather, directories tree for executions and extra configuration set up in simulator.
- **test_env.py**: Tests for `sinergym/sinergym/envs/\* <https://github.com/jajimer/sinergym/tree/main/sinergym/envs>`__. Here will be all tests that check Sinergym simulation environments based on OpenAI Gym.
- **test_controller.py**: Tests for `sinergym/sinergym/utils/controllers.py <https://github.com/jajimer/sinergym/blob/main/sinergym/utils/controllers.py>`__. Here will be all tests that check agent controller like Rule-Based-Controller for example.
- **test_stable_baselines.py**: Tests for `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`__. Here will be all tests that check Sinergym simulation environments can be used correctly with Stable Baselines 3 algorithms.

If you want to make new tests, you can append to this modules or create a new one if the conceptual context is different.

.. note:: For more information about Sinergym tests, visit our `repository README <https://github.com/jajimer/sinergym/blob/main/tests/README.md>`__.