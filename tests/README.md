# Sinergym tests

The [pytest](https://docs.pytest.org/en/6.2.x/) framework makes it easy to write small tests, yet scales to support complex functional testing for applications and libraries.

Main features are:

- Detailed info on failing assert statements (no need to remember self.assert* names).
- Auto-discovery of test modules and functions.
- Modular fixtures for managing small or parametrized long-lived test resources.
- Can run unittest (including trial) and nose test suites out of the box.
- Rich plugin architecture, with over 315+ external plugins and thriving community.
- Python 3.6+ and PyPy 3.


## Install pytest

This project has already established this dependency. However, to install it independently:

```sh
$ pip install pytest
```

## Running tests

In order to run our current tests:

```sh
$ pytest tests/ -vv
```

This runs all tests within tests/ directory. If we want verbose use `-v` or `-vv` option. To run an unique module tests, for example we can do: 

```sh
$ pytest tests/test_common.py -vv
```

## Create new tests

These tests have been created in [sinergym/tests](https://github.com/ugr-sail/sinergym/tree/main/tests) directory, they are organized by different modules:

- ``test_common.py``: includes tests for [sinergym/sinergym/utils/common.py](https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/common.py). Checks *Sinergym*'s common utils functionalities. 
- ``test_reward.py`` includes tests for [sinergym/sinergym/utils/rewards.py](https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/rewards.py). CheckS reward function(s).
- ``test_wrapper.py``: includes tests for [sinergym/sinergym/utils/wrappers.py](https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/wrappers.py). CheckS wrappers.
- ``test_simulator.py``: includes tests for [sinergym/sinergym/simulators/\*](https://github.com/ugr-sail/sinergym/tree/main/sinergym/simulators). Checks the low level simulator interaction and communication interface.
- ``test_env.py``: includes tests for [sinergym/sinergym/envs/\*](https://github.com/ugr-sail/sinergym/tree/main/sinergym/envs). Checks environments features.

If you want to add new tests, append them to this modules or create a new one if required.

### Fixtures

In order to execute tests, it is necessary to define project instances to check specific functionalities. Thus, we need to create [Fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) which can be used in each testing module, and which are stored in [conftest.py](https://github.com/ugr-sail/sinergym/blob/main/tests/conftest.py) (centralized fixtures). 

Let's see two of them:

```python
# sinergym/tests/conftest.py

@pytest.fixture(scope='session')
def json_path(pkg_data_path):
    return os.path.join(pkg_data_path, 'buildings', '5ZoneAutoDXVAV.epJSON')

# ...
@ pytest.fixture(scope='function')
def building(json_path):
    with open(json_path) as json_f:
        building_model = json.load(json_f)
    return building_model
# ...
```

Fixtures are functions which return a specific value or instance. We can determine the fixture execution frequency using `Scope` argument. Possible values for scope are:

- *function*: Execution per individual test.
- *class*: Execution per class (set of tests).
- *module*: Execution per module (each .py file).
- *package*: Execution per package.
- *session*: An unique execution during testing. 

Then, these returned values can be used in any test at any time. If we define a *session* scope, changes of the test to the value returned by fixture conditions the next test that uses this value. However, if we define a *function* scope, fixture will be executed again each time test finish, overwriting returned value and not condition next test that use this fixture value.

At the same time, we can use a fixture within definition of other fixture.

### Test format

In order to make a new test, follow the next format:

```python
def test_get_eplus_run_info(simulator, json_path):
    info = simulator._get_eplus_run_info(building_path)
    assert info == (1, 1, 3, 31, 0, 4)
```

A test is only a function that must start with the word *test*. We can specify previously-defined fixtures as arguments. We check specific states of execution using `assert` following of a conditional we want to check.

It is possible to check for exception that should occur in our tests using *pytest* like this:

```python
def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
```

### Parametrize tests

Suppose you want to check the same tests with different inputs. To do so, you can execute the same test several times using parametrize mark:

```python
# We want check this function
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

This code include three test in one. We define variables in parametrize mark and they will acquire the value of each input. Then, they are used like test arguments. You can combine parametrize marks and fixtures in test arguments. Also, You can concatenate several parametrize marks and *pytest* will combine them in all possible options.