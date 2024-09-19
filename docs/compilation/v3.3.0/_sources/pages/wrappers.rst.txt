############
Wrappers
############

*Sinergym* provides several **wrappers** to add functionality to the environment that isn't included by default. 
The code is available in 
`sinergym/sinergym/utils/wrappers.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/wrappers.py>`__. 
You can create your own wrappers by inheriting from *gym.Wrapper* or one of its variants, as seen in the 
`Gymnasium documentation <https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/>`__.

***********************
MultiObjectiveReward
***********************

The environment step will return a vector reward (selected elements in the wrapper constructor, 
one for each objective) instead of a traditional scalar value. Refer to 
`#301 <https://github.com/ugr-sail/sinergym/issues/301>`__ for more information.

***************************
PreviousObservationWrapper
***************************

This wrapper adds observation values from the previous timestep to the current environment 
observation. You can select the variables you want to track for their previous observation values.

***********************
DatetimeWrapper
***********************

This wrapper replaces the ``day_of_month`` value with the ``is_weekend`` flag, and the ``hour`` and ``month`` 
values with sin and cos values. The observation space is automatically updated.

***********************
NormalizeAction
***********************

This wrapper applies normalization in the action space. It's particularly useful in DRL algorithms, 
as some of them only work correctly with normalized values, making environments more generic in DRL solutions.

By default, normalization is applied in the range ``[-1,1]``. However, a different **range** can be specified 
when the wrapper is instantiated.

*Sinergym* **parses** these values to the real action space defined in the original environment internally 
before sending it to the *EnergyPlus* Simulator via the API middleware.

.. image:: /_static/normalize_action_wrapper.png
  :scale: 50 %
  :alt: Normalize action wrapper graph.
  :align: center

***********************
DiscretizeEnv
***********************

Wrapper to discretize the action space. The **Discrete space** should be defined according to the Gymnasium standard. 
This space should be either ``gym.spaces.Discrete``, ``gym.spaces.MultiDiscrete``, or ``gym.spaces.MultiBinary``. 
An **action mapping function** is also provided to map these values into ones that are compatible with the underlying 
continuous environment (before sending it to the simulator).

.. important:: The discrete space **must** discretize the original continuous space. Hence, 
               the discrete space should only reach values that are considered in the original 
               environment action space.

Users can define this action mapping function to specify the transition from discrete to continuous values. 
If the output of the action mapping function doesn't align with the original environment action space, 
an error will be raised. Refer to :ref:`Environment Discretization Wrapper` for a usage example.

.. image:: /_static/discretize_wrapper.png
  :scale: 50 %
  :alt: Discretize wrapper graph.
  :align: center

***************************
IncrementalWrapper
***************************

A wrapper is available to convert some of the continuous environment variables into actions that indicate an 
increase/decrease in their current value, rather than directly setting the value. A dictionary is specified 
as an argument to calculate the possible increments/decrements for each variable. This dictionary uses the name 
of each variable to be transformed as the key, and the value is a tuple of values called **delta** and **step**, 
which creates a set of possible increments for each desired variable.

- **delta**: The maximum range of increments and decrements.

- **step**: The interval of intermediate values within the ranges.

The following figure illustrates its operation. Essentially, the values are rounded to the nearest increment 
value and added to the current real values of the simulation:

.. image:: /_static/incremental_wrapper.png
  :scale: 50 %
  :alt: Incremental wrapper graph.
  :align: center

***************************
DiscreteIncrementalWrapper
***************************

A wrapper for an incremental setpoint action space environment is also available. This wrapper updates 
an environment, transforming it into a *discrete* environment with an action mapping function and action 
space based on the specified **delta** and **step**. The action will be added to the **current setpoint** 
values instead of overwriting the latest action. Therefore, the action is the current setpoint values with 
the increase, rather than the discrete value action, which is intended to define the increment/decrement itself.

.. warning:: This wrapper fully changes the action space from continuous to discrete, meaning that increments/decrements 
             apply to all variables. In essence, selecting variables individually as in IncrementalWrapper is not possible.

.. image:: /_static/discrete_incremental_wrapper.png
  :scale: 50 %
  :alt: Discrete incremental wrapper graph.
  :align: center

***********************
NormalizeObservation
***********************

This is used to transform observations received from the simulator into values between -1 and 1. 
It's based on the 
`dynamic normalization wrapper of Gymnasium <https://gymnasium.farama.org/_modules/gymnasium/wrappers/normalize/#NormalizeObservation>`__. 
Initially, it may not be precise and the values might often be out of range, so use this wrapper 
with caution.

***********************
LoggerWrapper
***********************

This is a wrapper for logging all interactions between the agent and the environment. The Logger class can 
be selected in the constructor if a different type of logging is required. For more information about the 
*Sinergym* Logger, refer to :ref:`Logger`.

***********************
MultiObsWrapper
***********************

This stacks observations received in a history queue (the size can be customized).


.. note:: For examples about how to use these wrappers, visit :ref:`Wrappers example`.

.. important:: You have to be careful if you are going to use several nested wrappers.
               A wrapper works on top of the previous one. The order is flexible since *Sinergym* v3.0.5.