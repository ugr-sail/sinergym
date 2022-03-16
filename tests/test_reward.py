import pytest

import sinergym.utils.rewards as R


@pytest.mark.parametrize(
    'power,temperatures,month,day,reward,reward_energy,reward_comfort',
    [
        # Input 1
        (
            186.5929171535975,
            [22.16742570092868],
            3,
            31,
            -0.009329645857679876,
            -0.018659291715359752,
            -0.0
        ),
        # Input 2
        (
            688.0477550424935,
            [26.7881162590194],
            3,
            30,
            -1.6784605172618248,
            -0.06880477550424935,
            -3.2881162590194
        ),
        # Input 3
        (
            23168.30752221127,
            [20.37505026953311],
            2,
            25,
            -1.1584153761105636,
            -2.316830752221127,
            -0.0
        ),
    ]
)
def test_calculate(
        simple_reward,
        power,
        temperatures,
        month,
        day,
        reward,
        reward_energy,
        reward_comfort):
    result = simple_reward.calculate(power, temperatures, month, day)
    assert result[0] == reward
    assert result[1]['reward_energy'] == reward_energy
    assert result[1]['reward_comfort'] == reward_comfort
