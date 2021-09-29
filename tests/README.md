# Sinergym Tests

**Welcome to pytest!**

The [pytest](https://docs.pytest.org/en/6.2.x/) framework makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.

Main features are:

- Detailed info on failing assert statements (no need to remember self.assert* names).
- Auto-discovery of test modules and functions.
- Modular fixtures for managing small or parametrized long-lived test resources.
- Can run unittest (including trial) and nose test suites out of the box.
- Rich plugin architecture, with over 315+ external plugins and thriving community.
- Python 3.6+ and PyPy 3.


## Install Pytest

This project has already established this dependency. However, to install it independently:

```sh
$ pip install pytest
```

## Running tests

In order to run our current tests:

```sh
$ pytests tests/ -vv
```

This runs all tests within tests/ directory. If we want verbose use `-v` or `-vv` option. To run an unique module tests, for example we can do: 

```sh
$ pytests tests/test_common.py -vv
```

## Create new tests

These tests have been created in [sinergym/tests](https://github.com/jajimer/sinergym/tree/main/tests) directory, they are organized by different modules:

- **test_common.py**: Tests for [sinergym/sinergym/utils/common.py](https://github.com/jajimer/sinergym/blob/main/sinergym/utils/common.py). Here will be all tests that check Sinergym common utils functionalities. 
- **test_reward.py**: Tests for [sinergym/sinergym/utils/rewards.py](https://github.com/jajimer/sinergym/blob/main/sinergym/utils/rewards.py). Here will be all tests that check implementation(s) of reward function(s) applicable to sinergym environments. 
- **test_wrapper.py**: Tests for [sinergym/sinergym/utils/wrappers.py](https://github.com/jajimer/sinergym/blob/main/sinergym/utils/wrappers.py). Here will be all tests that check wrappers to normalize Sinergym default environment observations.
- **test_simulator.py**: Tests for [sinergym/sinergym/simulators/\*](https://github.com/jajimer/sinergym/tree/main/sinergym/simulators). Here will be all tests that check low level Sinergym simulator and communication interface.
- **test_env.py**: Tests for [sinergym/sinergym/envs/\*](https://github.com/jajimer/sinergym/tree/main/sinergym/envs). Here will be all tests that check Sinergym simulation Environments based on OpenAI Gym.

If you want to make new tests, you can append to this modules or create a new one if the conceptual context is different.

### Fixtures

In order to execute tests, it's necessary to define project instances to check specific functionalities. Thus, we need to create [Fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) which can be used in each testing module and they are stored in [conftest.py](https://github.com/jajimer/sinergym/blob/main/tests/conftest.py) file (centralized fixtures). Let's see two of them:

```python
# sinergym/tests/conftest.py

@pytest.fixture(scope="session")
def idf_path(pkg_data_path):
     return os.path.join(pkg_data_path, 'buildings', "5ZoneAutoDXVAV.idf")

# ...
@pytest.fixture(scope="session")
def epm(idf_path):
    return Epm.from_idf(idf_path)
# ...
```

Fixtures are functions, these functions return a specific value or instance. we can determine the fixture execution frequency using `Scope` argument. Possible values for scope are:

- *function*: Execution per individual test.
- *class*: Execution per class (set of tests).
- *module*: Execution per module (each .py file).
- *package*: Execution per package.
- *session*: An unique execution during testing. 

Then, these returned values can be used in any test at any time. If we define "session" scope, changes of the test to the value returned by fixture conditions the next test that uses this value. However, if we define "function" scope, fixture will be executed again each time test finish, overwriting returned value and not condition next test that use this fixture value.

At the same time, we can use a fixture within definition of other fixture, like we see in epm.

### Test format

In order to make a new test, the format is really simple:

```python
def test_get_eplus_run_info(simulator, idf_path):
    info = simulator._get_eplus_run_info(idf_path)
    assert info == (1, 1, 3, 31, 0, 4)
```

Test is only a function. It must start with the word "*test*" and we can specify fixtures which we have defined previously like arguments. We check specific states of execution using `assert` following of a conditional we want to check.

It is possible to check for exception that should occur in our tests using pytest like this:

```python
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

### Parametrize tests

Imagine you want to check the same tests with different inputs. How can you do that? You can execute the same test several times using parametrize mark:

```python
#We want check this function
def multiply(input_num1,input_num2):
    return input_num1*input_num2

@pytest.mark.parametrize(
    "input_num1,input_num2,expected",
    [
        # Input 1
        (
            1,   #input_num1
            1,   #input_num2
            1    #expected
        ),
        # Input 2
        (
            2, 
            3,
            6
        ),
        # Input 3
        (
            5,
            -2,
            -10
        ),
    ]
)
def test_multiply(input_num1,input_num2,expected):
    assert multiply(input_num1,input_num2) == expected
```

This code has three test in one. We define variables in parametrize mark and they will acquire the value of each input. Then, they are used like test arguments. You can combine parametrize marks and fixtures in test arguments. Also, You can concatenate several parametrize marks and pytest will combine them in all possible options.