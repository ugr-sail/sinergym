#######
Rewards
#######

Defining a reward function is one of the most important things in reinforcement learning. Consequently, Sinergym allows you to define your own reward functions or use 
the ones we have already implemented (see code below).

-  ``LinearReward`` implements a linear reward function, where both energy consumption and thermal discomfort are normalized and add together with different weights. The 
   discomfort is calculated as the absolute difference between current temperature and comfort range (so if the temperature is inside that range, the discomfort would be 0).
   This is a typically used function where thermal satisfaction of people inside the controlled building has been taken into account.

-  ``ExpReward`` is very similar, but in this case the discomfort is calculated using the exponential difference between current temperature and comfort ranges. That means 
   that the penalty for the discomfort is higher is we are far from the target temperatures.

-  ``HourlyLinearReward`` is a slight modification of the linear function, but the weight given to the discomfort depends on the hour of the day. If the current hour of the 
   simulation is in working hours (by default, from 9 AM to 7 PM) both comfort and energy consumption weights equally, but outside those hours only energy is considered.


These rewards are always negative, meaning that perfect behavior has a cumulative reward of 0. Notice also that there are two temperature comfort ranges defined, one for the 
summer period and other for the winter period. The weights of each term in the reward allow to adjust the importance of each aspect when evaluating the environments.

By default, all environments use ``LinearReward`` with default parameters. But you can change this configuration using ``gym.make()`` as follows:

.. code:: python
    
    from sinergym.utils.rewards import ExpReward

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=ExpReward, reward_kwargs = {
                                                                            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                                                            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                                                            'range_comfort_winter': (20.0, 23.5),
                                                                            'range_comfort_summer': (23.0, 26.0),
                                                                            'energy_weight': 0.1})

.. warning:: When specifying a different reward with `gym.make` than the default environment ID, it is very important to set the `reward_kwargs` that are required and therefore do not have a default value. In the rewards we have defined it is required: **temperature_variable(s)**, **energy_variable**, **range_comfort_winter**, **range_comfort_summer**. The rest of them have default values and it is not necessary to specify.


It is also pretty simple to define your own classes. For example, imagine you want a reward signal which returns always -1 (however we do not recommend using it for training agents :)).
The only requirement is that the calculation is performed using ``__call__`` method, which returns the reward and a dictionary with extra information. The below code implements this.

.. code:: python

    from sinergym.utils.rewards import BaseReward

    class CustomReward(BaseReward):
        """Naive reward function."""
        def __init__(self, env):
            super(CustomReward, self).__init__(env)
        def __call__(self):
            return -1.0, {}

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=CustomReward)


More reward functions will be included in the future, so stay tuned!


.. literalinclude:: ../../../sinergym/utils/rewards.py
    :language: python
