import pytest

# We want tester this functions


def multiply(input_num1, input_num2):
    return input_num1 * input_num2


# We can implement several inputs and pytest try it one by one (in this
# function pytest execute 3 tests)
@pytest.mark.parametrize(
    'input_num1,input_num2,expected',
    [
        # Input 1
        (
            1,  # input_num1
            1,  # input_num2
            1  # expected
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
def test_multiply(input_num1, input_num2, expected):
    assert multiply(input_num1, input_num2) == expected

# If we have an object that we will use in several tests, fixtures can be
# really useful


@pytest.fixture
def example_value():
    # Normally we will use for complex items like instances, DB rows, etc.
    return 40

# And now, we can use this value whenever


def test_divisible5(example_value):
    assert ((example_value % 5) == 0)


def test_divisible10(example_value):
    # If we change the fixture state only is applied in that test function
    assert ((example_value % 10) == 0)
