#######
Rewards
#######

.. |br| raw:: html

   <br />

The definition of a reward function is essential for reinforcement learning. For this reason, *Sinergym* allows you to use pre-implemented reward functions or to create custom ones.

The predefined reward functions in *Sinergym* are designed as multi-objective, incorporating both **energy consumption** and **thermal discomfort**. These are **normalised** and added with varying **weights**. The assigned weights for each term in the reward function enable the importance of each reward component to be adjusted.

It should be noted that pre -mplemented rewards are expressed in **negative** terms, signifying that optimal behaviour results in a cumulative reward of 0. Separate temperature comfort ranges are defined for summer and winter periods. 

The most basic definition of the reward signal in *Sinergym* consists of the following equation:

.. math:: r_t = - \omega \ \lambda_P \ P_t - (1 - \omega) \ \lambda_T \ (|T_t - T_{up}| + |T_t - T_{low}|)

Where: |br|

:math:`P_t` represents power consumption, |br|
:math:`T_t` is the current indoor temperature, |br|
:math:`T_{up}` and :math:`T_{low}` are the upper and lower comfort range limits, respectively, |br|
:math:`\omega` is the weight assigned to power consumption, and consequently, :math:`1 - \omega` represents the comfort weight, |br|
:math:`\lambda_P` and :math:`\lambda_T` are scaling constants for consumption and comfort penalties, respectively.

.. warning:: The constants :math:`\lambda_P` and :math:`\lambda_T` are configured to create a proportional 
             relationship between energy and comfort penalties, with the objective of calibrating their magnitudes.
             It is essential to adjust these constants when working with different buildings to ensure that the magnitude of both reward parts remains consistent.

Different types of reward functions are already pre-defined in *Sinergym*:

-  ``LinearReward``: implements a **linear reward** function, where discomfort is calculated as the absolute 
   difference between the current temperature and the comfort range.

-  ``ExpReward``: similar to the linear reward, but calculates discomfort using the **exponential difference** 
   between the current temperature and comfort ranges, resulting in a higher penalty for larger deviations 
   from target temperatures.

-  ``HourlyLinearReward``: adjusts the weight assigned to discomfort based on the **hour of the day**.

-  ``NormalizedLinearReward``: normalizes the reward components based on the maximum energy and comfort penalties. It is calculated using a moving average, and the :math:`\lambda_P` and :math:`\lambda_T` constants are not required to calibrate both magnitudes.

  .. warning:: This reward function improves in accuracy as the simulation progresses, but is less accurate in the early stages when it is not yet balanced.

- ``EnergyCostLinearReward``: is a linear reward function which includes an **energy cost** term:

   .. math:: r_t = - \omega_P \ \lambda_P \ P_t - \omega_T \ \lambda_T \ (|T_t - T_{up}| + |T_t - T_{low}|) - (1 - \omega_P - \omega_T) \ \lambda_{EC} \ EC_t

   .. warning:: This function is used internally by the :ref:`EnergyCostWrapper` and it is not intended to be used otherwise.

It should be noted that the reward functions have parameters in their constructors, the values of which may vary based on the building used or other factors. The default setting is the ``LinearReward`` function with the standard parameters for each building. Please refer to the example in :ref:`Adding a new reward` for further details on how to define custom rewards.

.. warning:: When specifying a reward other than the default reward for a given environment ID, it is necessary to specify the
             ``reward_kwargs`` when calling ``gym.make``.

************
Reward terms
************

By default, reward functions return a **scalar value** and the values of the **terms** involved in its calculation. The values of these terms depend on the specific reward function used and are automatically added to the environment's ``info`` dictionary. 

The reward structure generally matches the diagram below:

.. image:: /_static/reward_terms.png
  :scale: 70 %
  :alt: Reward terms
  :align: center

**************
Custom rewards
**************

Defining custom reward functions is straightforward. For instance, a reward signal that always returns -1 can be implemented as follows:

.. code:: python

    from sinergym.utils.rewards import BaseReward

    class CustomReward(BaseReward):
        """Naive reward function."""
        def __init__(self, env):
            super(CustomReward, self).__init__(env)
        def __call__(self, obs_dict):
            return -1.0, {}

    env = gym.make('Eplus-discrete-stochastic-mixed-v1', reward=CustomReward)

For advanced reward functions, we recommend inheriting from the main class, ``LinearReward``, and overriding the default methods. 

Pre-defined reward functions simplify observation processing to extract consumption and comfort violation data, from which  penalty values are calculated. Weighted reward terms are then computed from these penalties and subsequently added.

.. image:: /_static/reward_structure.png
  :scale: 70 %
  :alt: Reward steps structure
  :align: center

