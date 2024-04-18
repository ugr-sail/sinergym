##############
Environments
##############

*Sinergym* generates multiple environments for each building, each defined by a unique 
configuration that outlines the control problem to be solved. To view the list of available 
environment IDs, it is recommended to use the provided method:

.. code-block:: python

  # This script is available in scripts/consult_environments.py
  import sinergym
  import gymnasium as gym
  from sinergym.utils.common import get_ids

  # Get the list of available environments
  sinergym_environment_ids = get_ids()
  print(sinergym_environment_ids)

  # Make and consult some of the environments
  env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
  print(env.info())

Environment names adhere to the pattern ``Eplus-<building-id>-<weather-id>-<control_type>-<stochastic (optional)>-v1``. 
These IDs offer a general overview of the environment. For more detailed information about each environment, 
utilize the ``info`` method as demonstrated in the example code.

.. important:: As of *Sinergym* v3.0.9, environments are automatically generated 
               using JSON configuration files for each building, eliminating the 
               need for manual registration of each environment ID with parameters 
               set directly in the environment constructor. Refer to 
               :ref:`Environments Configuration and Registration` for more details.

.. note:: Discrete environments are customizable. The default control for these 
          environments is quite basic. You can employ a continuous environment 
          and customize discretization using our dedicated wrapper. For more 
          information, see :ref:`DiscretizeEnv`.

.. note:: For additional details on the buildings (epJSON) and weather conditions (EPW) 
          used, please visit the :ref:`Buildings` and :ref:`Weathers` sections respectively.

*********************
Available Parameters
*********************

With the **environment constructor**, we can configure the complete **context** 
of our environment for experimentation, either starting from one predefined by 
*Sinergym* or creating a new one.

*Sinergym* initially provides **non-configured** buildings and weathers. 
Depending on these argument values, these files are updated to adapt to these 
new features, this will be done by Sinergym automatically. For example, using 
another weather file requires updating the building location and design days, 
using new observation variables requires updating the ``Output:Variable`` and 
``Output:Meter`` fields, the same occurs with extra configuration context 
concerned with simulation directly, if weather variability is set, then a weather 
with noise will be used. These new building and weather file versions are saved in 
the Sinergym output folder, leaving the original intact.

The next subsections will show which **parameters** are available and what 
their functions are:

building file 
==============

The parameter ``building_file`` is the *epJSON* file, a new 
`adaptation <https://energyplus.readthedocs.io/en/latest/schema.html>`__ of *IDF* 
(Intermediate Data Format) where *EnergyPlus* building model is defined. These 
files are not configured for a particular environment as we have mentioned. 
Sinergym does a previous building model preparation to the simulation, see the 
*Modeling* element in *Sinergym* backend diagram.

Weather files
==============

The parameter ``weather_file`` is the *EPW* (*EnergyPlus* Weather) file name where 
**climate conditions** during a year is defined.

This parameter can be either a weather file name (``str``) as mentioned, or a list 
of different weather files (``List[str]``). When a list of several files is defined, 
*Sinergym* will select an *EPW* file in each episode and re-adapt the building model 
randomly. This is done to increase the complexity in the environment if desired. 

The weather file used in each episode is stored in *Sinergym* episode output folder, 
if **variability** (section :ref:`Weather Variability` is defined), the *EPW* stored 
will have that noise included.

Weather Variability
====================

**Weather variability** can be integrated into an environment using the
``weather_variability`` parameter.

It implements the 
`Ornstein-Uhlenbeck process <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.4200&rep=rep1&type=pdf>`__ 
to introduce **noise** to the weather data episode to episode. Then, the parameter 
established is a Python tuple of three variables (*sigma*, *mu*, and *tau*) whose 
values define the nature of that noise.

.. image:: /_static/ornstein_noise.png
  :scale: 80 %
  :alt: Ornstein-Uhlenbeck process noise with different hyperparameters.
  :align: center


Reward
=======

The parameter called ``reward`` is used to define the **reward class** 
(see section :ref:`Rewards`) that the environment is going to use to 
calculate and return scalar reward values each timestep.

Reward Kwargs
==============

Depending on the reward class that is specified to the environment, it 
may have **different arguments** depending on its type. In addition, 
if a user creates a new custom reward, it can have new parameters as well.

Moreover, depending on the building being used for the environment, the 
values of these reward parameters may need to be different, such as the 
comfort range or the energy and temperature variables of the simulation 
that will be used to calculate the reward.

Then, the parameter called ``reward_kwargs`` is a Python dictionary where 
we can **specify all reward class arguments** that they are needed. For 
more information about rewards, visit section :ref:`Rewards`.

Maximum Episode Data Stored in Sinergym Output
===============================================

*Sinergym* stores all the output of an experiment in a folder organized in 
sub-folders for each episode (see section :ref:`Output format` for more 
information). Depending on the value of the parameter ``max_ep_data_store_num``, 
the experiment will store the output data of the **last n episodes** set, 
where **n** is the value of the parameter.

In any case, if *Sinergym Logger* (See :ref:`Logger` section) is activated, 
``progress.csv`` will be present with the summary data of each episode.

Time variables
===============

*EnergyPlus* Python API has several methods in order to extract information 
about simulation time in progress. The argument ``time_variables`` is a list 
in which we can specify the name of the 
`API methods <https://energyplus.readthedocs.io/en/latest/datatransfer.html#datatransfer.DataExchange>`__ 
whose values we want to include in our observation.

By default, *Sinergym* environments will have the time variables 
``month``, ``day_of_month`` and ``hour``.

Variables
==========

The argument called ``variables`` is a dictionary in which it is specified 
the ``Output:Variable``'s we want to include in the environment observation. 
The format of each element, in order for *Sinergym* to process it, is the next:

.. code-block:: python

  variables = {
    # <custom_variable_name> : (<"Output:Variable" original name>,<variable_key>),
    # ...
  }

.. note:: For more information about the available variables in an environment, execute a default simulation with
          *EnergyPlus* engine and see RDD file generated in the output.

Meters
==========

In a similar way, the argument ``meters`` is a dictionary in which we can specify 
the ``Output:Meter``'s we want to include in the environment observation. 
The format of each element must be the next:

.. code-block:: python

  meters = {
    # <custom_meter_name> : <"Output:Meter" original name>,
    # ...
  }

.. note:: For more information about the available meters in an environment, execute a default simulation with
          *EnergyPlus* engine and see MDD and MTD files generated in the output.

Actuators
==========

The argument called ``actuators`` is a dictionary in which we specify the actuators we 
want to control with gymnasium interface, the format must be the next:

.. code-block:: python

  actuators = {
    # <custom_actuator_name> : (<actuator_type>,<actuator_value>,<actuator_original_name>),
    # ...
  }

.. important:: Actuators that have not been specified will be controlled by the building's default schedulers.

.. note:: For more information about the available actuators in an environment, execute a default control with
          *Sinergym* directly (empty action space) and see ``data_available.txt`` generated.

Action space
===========================

In *Sinergym*, the environment's observation and action spaces are defined through the 
arguments ``time_variables``, ``variables``, ``meters``, and ``actuators``. The 
observation space, composed of ``time_variables``, ``variables``, and ``meters``, is 
automatically generated. The action space, defined by the ``actuators``, requires explicit 
definition to establish the range of values supported by the Gymnasium interface or the number 
of discrete values in a discrete environment.

.. image:: /_static/spaces_elements.png
  :scale: 35 %
  :alt: *EnergyPlus* API components that compose observation and action spaces in *Sinergym*.
  :align: center

The ``action_space`` argument adheres to the Gymnasium standard and must be a continuous 
space (``gym.spaces.Box``) due to the *EnergyPlus* simulator's continuous value 
requirement. It's crucial that this definition aligns with the previously defined actuators, 
with *Sinergym* highlighting any inconsistencies.

.. note:: To adapt an environment to Gymnasium's ``Discrete``, ``MultiDiscrete``, or ``MultiBinary`` spaces, 
          akin to our predefined discrete environments, consult the section :ref:`DiscretizeEnv` and the 
          example in :ref:`Environment Discretization Wrapper`.

.. important:: While *Sinergym*'s environments come with predefined observation and action variables (
               details available in `default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__), 
               users are encouraged to explore and experiment with these spaces. For guidance, refer to 
               :ref:`Changing observation and action spaces`.

*Sinergym* also provides the option to create **empty action interfaces**, allowing users 
to leverage its benefits without directly using the *EnergyPlus* simulator. Control in 
this scenario is managed by the **default building model schedulers**. For further details, 
refer to the usage example :ref:`Default building control setting up an empty action interface`.

Environment Name
================

The ``env_name`` parameter is utilized to generate the **working directory name**, 
facilitating differentiation between multiple experiments within the same environment.

Extra Configuration
===================

Parameters related to the building model and simulator, such as ``people occupant``, ``timesteps per simulation hour``, 
and ``runperiod``, can be set as extra configurations. These configurations, which may expand in the future, 
are specified in the ``config_params`` argument, a Python Dictionary. For additional information 
on extra configurations in *Sinergym*, refer to :ref:`Extra Configuration in Sinergym simulations`.

Adding New Weathers for Environments
====================================

*Sinergym* provides a variety of weather files for diverse global climates to enhance experimental diversity.

To incorporate a **new weather**:

1. Download an **EPW** and its corresponding **DDY** file from the `EnergyPlus page <https://energyplus.net/weather>`__. 
   The *DDY* file provides location and design day details.

2. Ensure both files share the same name, differing only in their extensions, and place them 
   in the `weathers <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/weather>`__ folder.

Upon addition, *Sinergym* will automatically modify the ``SizingPeriod:DesignDays`` and ``Site:Location`` 
fields in the building model file using the *DDY* file.

Adding New Buildings for Environments
=====================================

Users can either modify existing environments or create new ones, incorporating new climates, 
action, and observation spaces. They also have the option to use a different **building model** 
(epJSON file) than the ones currently supported.

To add new buildings for use with *Sinergym*, follow these steps:

1. **Add your building file** (*epJSON*) to the 
   `buildings <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/buildings>`__ 
   directory. Ensure it's compatible with the EnergyPlus version used in *Sinergym*. 
   If you're using an *IDF* file from an older version, update it with **IDFVersionUpdater** 
   and convert it to *epJSON* format using **ConvertInputFormat**. Both tools are available 
   in the EnergyPlus installation folder.

2. **Adjust building objects** like ``RunPeriod`` and ``SimulationControl`` to suit your needs 
   in Sinergym. We recommend setting ``run_simulation_for_sizing_periods`` to ``No`` in 
   ``SimulationControl``. ``RunPeriod`` sets the episode length, which can be configured 
   in the building file or Sinergym settings (see :ref:`runperiod`). Make these modifications 
   in the *IDF* before step 1 or directly in the *epJSON* file.

3. **Identify the components** of the building that you want to observe and control. This is 
   the most challenging part of the process. Typically, users are already familiar with the 
   building and know the *name* and *key* of the elements in advance. If not, follow the process below:

   a. Run a preliminary simulation with EnergyPlus directly, without any control flow, to view the 
      different ``OutputVariables`` and ``Meters``. Consult the output files, specifically the *RDD* 
      extension file, to identify possible observable variables.

   b. The challenge is knowing the names but not the possible *Keys* (EnergyPlus doesn't initially 
      provide this information). Use these names to define the environment (see step 4). If the *Key* 
      is incorrect, *Sinergym* will notify you of the error and provide a **data_available.txt** 
      file in the output, as it has already connected with the EnergyPlus API. This file contains 
      all the **controllable schedulers** for the actions and all the **observable variables**, now 
      with their respective *Keys*, enabling the correct definition of the environment.

4. With this information, the next step is **defining the environment** using the building model. 
   You have several options:

  a. Use the *Sinergym* environment constructor directly. The arguments for building observation 
     and control are explained within the class and should be specified in the same format as the 
     EnergyPlus API.

  b. Set up the configuration to register environment IDs directly. For more information, refer to 
     the documentation :ref:`Environments Configuration and Registration`. *Sinergym* will verify 
     that the established configuration is correct and notify you of any potential errors.

5. If you've used *Sinergym*'s registry, you'll have access to environment IDs associated with your building. 
   Use them with ``gym.make(<environment_id>)`` as usual. If you've created an environment instance directly, 
   use that instance to start interacting with the building.

.. note:: To obtain information about the environment instance with the new building model, refer to 
          :ref:`Getting information about Sinergym environments`.

