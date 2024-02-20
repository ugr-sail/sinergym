#######
Rewards
#######

Defining a reward function is one of the most important things in reinforcement learning. 
Consequently, *Sinergym* allows to use pre-implemented reward functions or to define your 
own reward functions (see section bellow).

*Sinergym*'s predefined reward functions are developed as **multi-objective**, where both *energy 
consumption* and *thermal discomfort* are normalized and added together with different weights.
These rewards are **always negative**, meaning that perfect behavior has a cumulative 
reward of 0. Notice also that there are two temperature comfort ranges defined, 
one for the summer period and other for the winter period. The weights of each 
term in the reward allow to adjust the importance of each aspect when environments are evaluated.

.. math:: r_t = - \omega \ \lambda_P \ P_t - (1 - \omega) \ \lambda_T \ (|T_t - T_{up}| + |T_t - T_{low}|)

Where :math:`P_t` represents power consumption; :math:`T_t` is the current indoor temperature; 
:math:`T_{up}` and :math:`T_{low}` are the imposed comfort range limits 
(penalty is :math:`0` if :math:`T_t` is within this range); :math:`\omega` is the weight 
assigned to power consumption (and consequently, :math:`1 - \omega`, the comfort weight), 
and :math:`\lambda_P` and :math:`\lambda_T` are scaling constants for consumption and comfort, 
respectively.

.. warning:: :math:`\lambda_P` and :math:`\lambda_T` are constants established in order to set up a 
             proportional concordance between energy and comfort penalties. If you are
             using other buildings, be careful with these constants and update them in order to
             have a similar magnitude of the reward components.

This is the main idea of reward system in *Sinergym*. However, depending some details,
different kinds of reward function is developed:

-  ``LinearReward`` implements a **linear reward** function where the discomfort is calculated 
   as the absolute difference between current temperature and comfort range (so if the 
   temperature is inside that range, the discomfort would be 0).
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

- ``NormalizedLinearReward`` is the same function than ``LinearReward``, but it does not use
  the :math:`\lambda_P` and :math:`\lambda_T` to equilibrate the value magnitudes in reward
  components. Instead, it applies a normalization using the maximum energy consumption and comfort
  out of range values. This reward function is adaptive, since these maximum values are updated
  along the process. It is possible to specify the initial maximum values, by default they are 0.

  .. warning:: This reward function is not very precise at the beginning of the simulation, be careful with that.

The reward functions have a series of **parameters** in their constructor whose values 
may depend on the building we are using or other characteristics. For example, the 
internal temperature or energy variables used to calculate penalties may have a 
different name in different buildings.

The main parameters that it is considered in a function reward will be the next:

- **temperature_variables**: This field can be an *str* (only a unique zone temperature)
  or a *list* (with several zone temperatures).

- **energy_variables**: Name of the observation variables where energy consumption is 
  reflected.

- **range_comfort_winter**: Temperature comfort range for cold season. Depends on 
  environment you are using.

- **range_comfort_summer**: Temperature comfort range for hot season. Depends on 
  environment you are using.

- **energy_weight**: Weight given to the energy term. Defaults to 0.5. Comfort weight
  will have 1-*energy_weight*.

.. note:: These parameters are usually common to any reward function. 
          However, they may have different parameters depending on the 
          one being used.

By default, all environments use ``LinearReward`` with default parameters. If you want to change this, see
an example in :ref:`Adding a new reward`.

.. note:: By default, reward class will return the reward value and the terms used in its calculation. Terms
          depends on the reward function used specifically.
          These terms will be added to info dict in environment automatically.

.. warning:: When specifying a different reward with `gym.make` than the 
             default environment ID, it is very important to set the `reward_kwargs` 
             that are required and therefore do not have a default value. 


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
