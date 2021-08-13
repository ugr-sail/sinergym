############
Tests
############

This project is automatically supervised using tests developed specifically for it. If you want to check energym has been installed successfully, run next command:

.. code:: sh

    $ pytest tests/ -vv

Anyway, every time energym repository is updated, the tests will run automatically in a remote container using the Dockerfile to build it. `Travis-CI <https://docs.travis-ci.com/>`__ will do that job.

.. note:: See `.travis.yml <https://github.com/jajimer/energym/blob/main/.travis.yml>`__ for more information

These tests running under `pytest <https://docs.pytest.org/en/6.2.x/>`__ framework which makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.

****************
Install Pytest
****************

This project has already established this dependency if you have installed *extras* libreries or *test* librery specifically (see :ref:`4. Install the package`). However, to install it independently:

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

These tests have been created in `energym/tests <https://github.com/jajimer/energym/tree/main/tests>`__ directory, they are organized by different modules:

- **test_common.py**: Tests for `energym/energym/utils/common.py <https://github.com/jajimer/energym/blob/main/energym/utils/common.py>`__. Here will be all tests that check Energym common utils functionalities. 
- **test_reward.py**: Tests for `energym/energym/utils/rewards.py <https://github.com/jajimer/energym/blob/main/energym/utils/rewards.py>`__. Here will be all tests that check implementation(s) of reward function(s) applicable to energym environments.
- **test_wrapper.py**: Tests for `energym/energym/utils/wrappers.py <https://github.com/jajimer/energym/blob/main/energym/utils/wrappers.py>`__. Here will be all tests that check wrappers to normalize Energym default environment observations.
- **test_simulator.py**: Tests for `energym/energym/simulators/\* <https://github.com/jajimer/energym/tree/main/energym/simulators>`__. Here will be all tests that check low level Energym simulator and communication interface.
- **test_env.py**: Tests for `energym/energym/envs/\* <https://github.com/jajimer/energym/tree/main/energym/envs>`__. Here will be all tests that check Energym simulation environments based on OpenAI Gym.
- **test_controller.py**: Tests for `energym/energym/utils/controllers.py <https://github.com/jajimer/energym/blob/main/energym/utils/controllers.py>`__. Here will be all tests that check agent controller like Rule-Based-Controller for example.
- **test_stable_baselines.py**: Tests for `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`__. Here will be all tests that check Energym simulation environments can be used correctly with Stable Baselines 3 algorithms.

If you want to make new tests, you can append to this modules or create a new one if the conceptual context is different.

.. note:: For more information about Energym tests, visit our `repository README <https://github.com/jajimer/energym/blob/main/tests/README.md>`__.