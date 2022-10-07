#######
Rewards
#######

Defining a reward function is one of the most important things in reinforcement learning. 
Consequently, *Sinergym* allows you to define your own reward functions or use 
the ones we have already implemented (see section bellow).

-  ``LinearReward`` implements a **linear reward** function, where both energy consumption and 
   thermal discomfort are normalized and add together with different weights. 
   The discomfort is calculated as the absolute difference between current temperature and 
   comfort range (so if the temperature is inside that range, the discomfort would be 0).
   This is a typically used function where thermal satisfaction of people inside the 
   controlled building has been taken into account.

-  ``ExpReward`` is very similar, but in this case the discomfort is calculated 
   using the **exponential difference** between current temperature and comfort ranges. 
   That means that the increase penalty for the discomfort is higher if we are far from 
   the target temperatures.

-  ``HourlyLinearReward`` is a slight modification of the linear function, but 
   the weight given to the discomfort depends on the **hour of the day**. If the current 
   hour of the simulation is in working hours (by default, from 9 AM to 7 PM) both 
   comfort and energy consumption weights equally, but outside those hours only energy 
   is considered.


These rewards are **always negative**, meaning that perfect behavior has a cumulative 
reward of 0. Notice also that there are two temperature comfort ranges defined, 
one for the summer period and other for the winter period. The weights of each 
term in the reward allow to adjust the importance of each aspect when evaluating 
the environments.

The reward functions have a series of **parameters** in their constructor whose values 
may depend on the building we are using or other characteristics. For example, the 
internal temperature or energy variables used to calculate penalties may have a 
different name in different buildings.

The main parameters that it is considered in a function reward will be the next:

- **temperature_variable**: This field can be an *str* (only a unique zone temperature)
  or a *list* (with several zone temperatures).

- **energy_variable**: Name of the observation variable where energy consumption is 
  reflected.

- **range_comfort_winter**: Temperature comfort range for cold season. Depends on 
  environment you are using.

- **range_comfort_summer**: Temperature comfort range for hot season. Depends on 
  environment you are using.

- **energy_weight**: Weight given to the energy term. Defaults to 0.5. Comfort weight
  will have 1-*energy_weight*.

- **lambda_energy**: Constant for removing dimensions from power(1/W). Defaults to 1e-4.

- **lambda_temperature**: Constant for removing dimensions from temperature(1/C). 
  Defaults to 1.0.

.. note:: These parameters are usually common to any reward function. 
          However, they may have different parameters depending on the 
          one being used.

By default, all environments use ``LinearReward`` with default parameters. 
But you can change this configuration using ``gym.make()`` as follows:

.. code:: python
    
    from sinergym.utils.rewards import ExpReward

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=ExpReward, reward_kwargs = {
                                                                            'temperature_variable': 'Zone Air Temperature (SPACE1-1)',
                                                                            'energy_variable': 'Facility Total HVAC Electricity Demand Rate (Whole Building)',
                                                                            'range_comfort_winter': (20.0, 23.5),
                                                                            'range_comfort_summer': (23.0, 26.0),
                                                                            'energy_weight': 0.1})

.. warning:: When specifying a different reward with `gym.make` than the 
             default environment ID, it is very important to set the `reward_kwargs` 
             that are required and therefore do not have a default value. 
             In the rewards we have defined it is required: 
             **temperature_variable(s)**, **energy_variable**, 
             **range_comfort_winter**, **range_comfort_summer**. 
             The rest of them have default values and it is not necessary to specify.

***************
Custom Rewards
***************

It is also pretty simple to define your **own classes**. For example, imagine you want 
a reward signal which returns always -1 (however we do not recommend using it 
for training agents).
The only requirement is that the calculation is performed using ``__call__`` 
method, which returns the reward and a dictionary with extra information. 
The below code implements this.

.. code:: python

    from sinergym.utils.rewards import BaseReward

    class CustomReward(BaseReward):
        """Naive reward function."""
        def __init__(self, env):
            super(CustomReward, self).__init__(env)
        def __call__(self):
            return -1.0, {}

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=CustomReward)


*More reward functions will be included in the future, so stay tuned!*
