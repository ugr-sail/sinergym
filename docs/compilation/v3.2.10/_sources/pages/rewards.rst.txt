#######
Rewards
#######

.. |br| raw:: html

   <br />

Defining a reward function is crucial in reinforcement learning. Consequently, *Sinergym* offers 
the option to use pre-implemented reward functions or define custom ones (see section below).

*Sinergym*'s predefined reward functions are developed as **multi-objective**, incorporating both 
*energy consumption* and *thermal discomfort*, which are normalized and combined with different weights. 
These rewards are **always negative**, indicating that optimal behavior results in a cumulative reward of 0. 
There are separate temperature comfort ranges defined for summer and winter periods. The weights assigned 
to each term in the reward function allow for adjusting the importance of each aspect during environment evaluation.

The main idea behind the reward system in *Sinergym* is captured by the equation:

.. math:: r_t = - \omega \ \lambda_P \ P_t - (1 - \omega) \ \lambda_T \ (|T_t - T_{up}| + |T_t - T_{low}|)

Where: |br|
:math:`P_t` represents power consumption, |br|
:math:`T_t` is the current indoor temperature, |br|
:math:`T_{up}` and :math:`T_{low}` are the upper and lower comfort range limits, respectively, |br|
:math:`\omega` is the weight assigned to power consumption, and consequently, :math:`1 - \omega` represents the comfort weight, |br|
:math:`\lambda_P` and :math:`\lambda_T` are scaling constants for consumption and comfort penalties, respectively.

.. warning:: The constants :math:`\lambda_P` and :math:`\lambda_T` are set to establish a proportional 
             relationship between energy and comfort penalties, calibrating their magnitudes. If you're working with different buildings, 
             it's important to adjust these constants to ensure a similar magnitude of the reward components.

Different types of reward functions are developed based on specific details:

-  ``LinearReward`` implements a **linear reward** function where discomfort is calculated as the 
    absolute difference between the current temperature and the comfort range.

-  ``ExpReward`` is similar to linear reward, but discomfort is calculated using the **exponential 
   difference** between current temperature and comfort ranges, resulting in a higher penalty for 
   greater deviations from target temperatures.

-  ``HourlyLinearReward`` adjusts the weight given to discomfort based on the **hour of the day**, 
   focusing more on energy consumption outside working hours.

- ``NormalizedLinearReward`` normalizes the reward components based on the maximum energy penalty 
  and comfort penalty, allowing for adaptability during the simulation. IN this reward is not required
  the :math:`\lambda_P` and :math:`\lambda_T` constants to calibrate both magnitudes.


  .. warning:: This reward function is not very precise at the beginning of the simulation, be careful with that.

These reward functions have parameters in their constructors whose values may vary depending on the building used
or other characteristics. By default, all environments use ``LinearReward`` with default parameters for each building. 
If you want to change this, see an example in :ref:`Adding a new reward`.

.. warning:: When specifying a different reward with `gym.make` than the 
             default environment ID, it is very important to set the `reward_kwargs` 
             that are required and therefore do not have a default value. 

***************
Reward terms
***************

By default, reward functions will return the **reward scalar value** and the **terms** used in their calculation. 
The values of these terms depend on the specific reward function used. They will be automatically 
added to the info dictionary in the environment. Typically, the structure will be the same as depicted 
in the diagram below:

.. image:: /_static/reward_terms.png
  :scale: 70 %
  :alt: Reward terms
  :align: center


***************
Custom Rewards
***************

It's also straightforward to define custom reward functions. For example, a reward signal that always returns -1 
can be implemented as follows:

.. code:: python

    from sinergym.utils.rewards import BaseReward

    class CustomReward(BaseReward):
        """Naive reward function."""
        def __init__(self, env):
            super(CustomReward, self).__init__(env)
        def __call__(self, obs_dict):
            return -1.0, {}

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=CustomReward)

For advanced reward functions, we suggest inheriting from our main class, ``LinearReward``, and overriding relevant methods. 
Our reward functions streamline observation processing to derive consumption and comfort violation data, from which absolute 
penalty values are calculated. Subsequently, weighted reward terms are calculated from penalties and summed.

.. image:: /_static/reward_structure.png
  :scale: 70 %
  :alt: Reward steps structure
  :align: center

By modularizing each of these steps, you can swiftly and conveniently modify specific aspects of the reward to create a new one, 
as demonstrated with our *exponential function reward version*, for example.

*More reward functions will be included in the future, so stay tuned!*
