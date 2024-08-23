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

However, *Sinergym* enhances its functionality with some additional features:

- It includes the last unnormalized observation as an attribute, which is very useful for logging.

- It provides access to the means and variations used for normalization calibration, addressing the low-level 
  issues found in the original wrapper.

- Similarly, these calibration values can be set via a method or in the constructor. 
  These values can be specified neither in list/numpy array format or writing the txt path 
  previously generated. Refer to the :ref:`API reference` for more information.

- The automatic calibration can be enabled or disabled as you interact with the environment, allowing the 
  calibration to remain static instead of adaptive.

In addition, this wrapper saves the values of **mean and var in txt files in the 
*Sinergym* output**. This should be used in case of evaluating the model later. 
An example of its use can be found in the use case :ref:`Loading a model`. It is
also important that normalization calibration update is deactivated during evaluation
processes.

These functionalities are crucial when evaluating models trained using this wrapper. 
For more details, visit `#407 <https://github.com/ugr-sail/sinergym/issues/407>`__.

*****************
Logger Wrappers
*****************

These wrappers use the *Sinergym* **LoggerStorage** class functionalities to store information during environment 
interactions. For more details, see :ref:`Logging System Overview`.

The diagram below illustrates the relationship between the wrappers and the logger, with explanations 
provided in the following subsections.

.. image:: /_static/logger_structure.png
  :scale: 50 %
  :alt: Logger wrappers graph.
  :align: center

LoggerWrapper
---------------

**BaseLoggerWrapper** is the abstract class for logger wrappers. It stores all information during 
environment interactions. The environment gains a new attribute, ``data_logger``, an instance of 
**LoggerStorage** containing all the information. You can create a custom *LoggerStorage* class by passing it to the 
constructor to change the logging backend, such as storing information in a different database.

Inherit from this class to create a new logger wrapper and implement abstract methods to define 
custom and episode summary metrics with the current data. Data is reset at the start of a new episode. 
*Sinergym* uses this base class to implement **LoggerWrapper**, the default logger, but custom loggers 
can be implemented easily following this abstract class (see :ref:`Logger Wrapper personalization/configuration`).

The current summary metrics for this default Sinergym wrapper are: *episode_num*,*mean_reward*,*std_reward*,
*mean_reward_comfort_term*,*std_reward_comfort_term*,*mean_reward_energy_term*,*std_reward_energy_term*,
*mean_abs_comfort_penalty*,*std_abs_comfort_penalty*,*mean_abs_energy_penalty*,*std_abs_energy_penalty*,
*mean_temperature_violation*,*std_temperature_violation*,*mean_power_demand*,*std_power_demand*,
*cumulative_power_demand*,*comfort_violation_time(%)*,*length(timesteps)*,*time_elapsed(hours)*,
*terminated*,*truncated*

Although data is reset with each new episode, this wrapper can be combined with others to save all data 
and summaries in different ways and platforms. *Sinergym* implements **CSVLogger** and **WandBLogger** by default.

CSVLogger
-----------

This wrapper works with the **LoggerWrapper** ``data_logger`` instance to parse and save data in CSV files during 
simulations. A **progress.csv** file is generated in the root output directory, containing general simulation results, 
updated per episode. The structure of this file is defined by the **LoggerWrapper** class.

Each episode directory includes a **monitor** folder with several CSV files for data such as observations, actions, 
rewards, infos, and custom metrics. For more details, see :ref:`Output Format`.

WandBLogger
-------------

This wrapper works with the **LoggerWrapper** ``data_logger`` instance to dump all information to the WandB platform in real-time. 
It is useful for real-time training process monitoring and is combinable with Stable Baselines 3 callbacks. 
The initialization allows definition of the project, entity, run groups, tags, and whether code or outputs are saved as platform 
artifacts, as well as dump frequency, excluded info keys, and excluded summary metric keys.

This wrapper can be used with a pre-existing WandB session, without the need to specify the entity or project 
(which, if provided, will be ignored), such as when using sweeps. It still allows specifying other parameters during construction, 
maintaining full functionality of the wrapper. If there is no pre-existing WandB session, the entity and project fields are required.

.. important:: A Weights and Biases account is required to use this wrapper, with an environment variable containing the API key for login. 
          For more information, visit `Weights and Biases <https://wandb.ai/site>`__.

**************************
ReduceObservationWrapper
**************************

This wrapper starts from the original observation space and reduces it by subtracting the variables 
specified in a string list parameter. These removed variables are returned in the info dictionary 
(under the key ``removed_variables``) and are not used in the agent optimization process.

If combined with the :ref:`LoggerWrapper` in subsequent layers, the removed variables will be saved 
in the output files, even if they are not "used". This makes it perfect for monitoring simulation 
values that are not part of the problem to be solved.

Similarly, any other wrapper applied in layers prior to this one will affect the removed variables, 
which can be observed in the info dictionary.

***********************
MultiObsWrapper
***********************

This stacks observations received in a history queue (the size can be customized).


.. note:: For examples about how to use these wrappers, visit :ref:`Wrappers example`.

.. important:: You have to be careful if you are going to use several nested wrappers.
               A wrapper works on top of the previous one. The order is flexible since *Sinergym* v3.0.5.