############
Environments
############

As mentioned in introduction, *Sinergym* follows the next structure:

.. image:: /_static/sinergym_diagram.png
  :width: 800
  :alt: *Sinergym* backend
  :align: center

|

*Sinergym* is composed of three main components: *agent*,
*communication* interface and *simulation*. The agent sends actions and receives observations from the environment
through the Gymnasium interface. At the same time, the gym interface communicates with the simulator engine
via *EnergyPlus* Python API, which provide the functionality to manage handlers such as actuators, meters and variables,
so their current values have a direct influence on the course of the simulation. 

The next image shows this process more detailed:

.. image:: /_static/backend.png
  :width: 1600
  :alt: *Sinergym* backend
  :align: center

|

The *Modeling* module works at the same level as the API and allows to adapt the building models before the start of each 
episode. This allows that the API can work correctly with the user's definitions in the environment. 

This scheme is very abstract, since these components do some additional tasks such as handling the folder structure 
of the output, preparing the handlers before using them, initiating callbacks for data collection during simulation, 
and much more.

***********************************
Additional observation information
***********************************

In addition to the observations returned in the step and reset methods as you can see in the images above, 
both return a Python dictionary with additional information:

- **Reset info:** This dictionary has the next keys:

.. code-block:: python

  info = {
            'time_elapsed(hours)': # <Simulation time elapsed in hours>,
            'month': # <Month in which the episode starts.>,
            'day': # <Day in which the episode starts.>,
            'hour': # <Hour in which the episode starts.>,
            'is_raining': # <True if it is raining in the simulation.>,
            'timestep': # <Timesteps count.>,
        }

- **step info:** This dictionary has the same keys than reset info, but it is added the action sent (action sent to the
  simulation, not the action sent to the environment), the reward and reward terms. The reward terms depend on
  the reward function used.  

**************************
Environments List
**************************

Sinergym creates multiple environments for each building, each with a specific configuration that defines the control 
problem to be solved. To access the list of available environment IDs, it is recommended to consult it using the provided 
method:

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

The environment names follow the pattern ``Eplus-<building-id>-<weather-id>-<control_type>-<stochastic (optional)>-v1``. These IDs 
provide information about the environment, but to get specific information about each environment, use the ``info`` method of each 
environment as shown in the example code.

.. important:: Since *Sinergym* v3.0.9, these environments are generated automatically using JSON configuration files
               for each building instead of register manually each environment id with parameters directly set in 
               environment constructor. See :ref:`Environments Configuration and Registration`.

.. warning:: Discrete environments can be customized. In fact, the default control of discrete environments is very simple.
             You can use a continuous environment and custom discretization using our dedicated wrapper directly, for more 
             information see :ref:`DiscretizeEnv`.

.. note:: For more information about buildings (epJSON) and weathers (EPW) used,
          please, visit sections :ref:`Buildings` and :ref:`Weathers` respectively.

*********************
Available Parameters
*********************

With the **environment constructor** we can configure the complete **context** of our environment 
for experimentation, either starting from one predefined by *Sinergym* shown in the 
table above or creating a new one.

*Sinergym* initially provides **non-configured** buildings and weathers. Depending of these argument values, 
these files are updated in order to adapt it to this new features, this will be made by Sinergym automatically.
For example, using another weather file requires building location and design days update, using new observation 
variables requires to update the ``Output:Variable`` and ``Output:Meter`` fields, the same occurs with extra 
configuration context concerned with simulation directly, if weather variability is set, then a weather with noise 
will be used. These new building and weather file versions, is saved in the Sinergym output folder, leaving the original intact.

The next subsections will show which **parameters** are available and what their function are:

building file 
==============

The parameter ``building_file`` is the *epJSON* file, a new `adaptation <https://energyplus.readthedocs.io/en/latest/schema.html>`__ 
of *IDF* (Intermediate Data Format) where *EnergyPlus* building model is defined. These files are not configured for a particular
environment as we have mentioned. Sinergym does a previous building model preparation to the simulation, see the *Modeling* element
in *Sinergym* backend diagram.

Weather files
==============

The parameter ``weather_file`` is the *EPW* (*EnergyPlus* Weather) file name where **climate conditions** during 
a year is defined.

This parameter can be either a weather file name (``str``) as mentioned, or a list of different weather files (``List[str]``).
When a list of several files is defined, *Sinergym* will select an *EPW* file in each episode and re-adapt building 
model randomly. This is done in order to increase the complexity in the environment whether is desired. 

The weather file used in each episode is stored in *Sinergym* episode output folder, if **variability** 
(section :ref:`Weather Variability` is defined), the *EPW* stored will have that noise included.

Weather Variability
====================

**Weather variability** can be integrated into an environment using ``weather_variability`` parameter.

It implements the `Ornstein-Uhlenbeck process <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.4200&rep=rep1&type=pdf>`__
in order to introduce **noise** to the weather data episode to episode. Then, parameter established is a Python tuple of three variables
(*sigma*, *mu* and *tau*) whose values define the nature of that noise.

.. image:: /_static/ornstein_noise.png
  :scale: 80 %
  :alt: Ornstein-Uhlenbeck process noise with different hyperparameters.
  :align: center


Reward
=======

The parameter called ``reward`` is used to define the **reward class** (see section :ref:`Rewards`)
that the environment is going to use to calculate and return reward values each timestep.

Reward Kwargs
==============

Depending on the reward class that is specified to the environment, it may have **different arguments** 
depending on its type. In addition, if a user creates a new custom reward, it can have new parameters as well.

Moreover, depending on the building being used for the environment, the values of these reward parameters may 
need to be different, such as the comfort range or the energy and temperature variables of the simulation that 
will be used to calculate the reward.

Then, the parameter called ``reward_kwargs`` is a Python dictionary where we can **specify all reward class arguments** 
that they are needed. For more information about rewards, visit section :ref:`Rewards`.

Maximum Episode Data Stored in Sinergym Output
===============================================

*Sinergym* stores all the output of an experiment in a folder organized in sub-folders for each episode 
(see section :ref:`Output format` for more information). Depending on the value of the parameter ``max_ep_data_store_num``, 
the experiment will store the output data of the **last n episodes** set, where **n** is the value of the parameter.

In any case, if *Sinergym Logger* (See :ref:`Logger` section) is activate, ``progress.csv`` will be present with 
the summary data of each episode.

Time variables
===============

*EnergyPlus* Python API has several methods in order to extract information about simulation time in progress. The
argument ``time_variables`` is a list in which we can specify the name of the 
`API methods <https://energyplus.readthedocs.io/en/latest/datatransfer.html#datatransfer.DataExchange>`__ 
whose values we want to include in our observation.

By default, *Sinergym* environments will have the time variables ``month``, ``day_of_month`` and ``hour``.

Variables
==========

The argument called ``variables`` is a dictionary in which it is specified the ``Output:Variable``'s we want to include in
the environment observation. The format of each element, in order to *Sinergym* can process it, is the next:

.. code-block:: python

  variables = {
    # <custom_variable_name> : (<"Output:Variable" original name>,<variable_key>),
    # ...
  }

.. note:: For more information about the available variables in an environment, execute a default simulation with
          *EnergyPlus* engine and see RDD file generated in the output.

Meters
==========

In a similar way, the argument ``meters`` is a dictionary in which we can specify the ``Output:Meter``'s we want to include in
the environment observation. The format of each element must be the next:

.. code-block:: python

  meters = {
    # <custom_meter_name> : <"Output:Meter" original name>,
    # ...
  }

.. note:: For more information about the available meters in an environment, execute a default simulation with
          *EnergyPlus* engine and see MDD and MTD files generated in the output.

Actuators
==========

The argument called ``actuators`` is a dictionary in which we specify the actuators we want to control with gymnasium interface, the format
must be the next:

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

As you have been able to observe, by defining the previous arguments, a definition of the observation and action 
space of the environment is being made. ``time_variables``, ``variables`` and ``meters`` make up our environment 
*observation*, while the ``actuators`` alone make up the environment *action*:

.. image:: /_static/spaces_elements.png
  :scale: 35 %
  :alt: *EnergyPlus* API components that compose observation and action spaces in *Sinergym*.
  :align: center

This allows us to do a **dynamic definition** of spaces, *Sinergym* will adapt the building model.
Observation space is created automatically, but action space must be defined in order to set up
the range values supported by the Gymnasium interface in the actuators, or the number of discrete values if
it is a discrete environment (using the wrapper for discretization).
                
Then, the argument called ``action_space`` defines this action space following the **gymnasium standard**.
*EnergyPlus* simulator works only with continuous values, so *Sinergym* action space defined must be continuous
too (``gym.spaces.Box``). This definition must be consistent with the previously defined actuators (*Sinergym* 
will show possible inconsistencies).

.. note:: If you want to adapt a environment to a gym ``Discrete``, ``MultiDiscrete`` or ``MultiBinary`` spaces,
          like our predefined discrete environments, see section :ref:`DiscretizeEnv` and an example in :ref:`Environment Discretization Wrapper`

.. important:: *Sinergym*'s listed environments have a default observation and action variables defined, 
               all information is available in `default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__.
               However, the users can experiment with this spaces, see :ref:`Changing observation and action spaces`.

*Sinergym* offers the possibility to create **empty action interfaces** too, so that you can take advantage 
of all its benefits instead of using the *EnergyPlus* simulator directly, meanwhile the control is 
managed by **default building model schedulers** as mentioned. For more information, see the example of use 
:ref:`Default building control setting up an empty action interface`.

Environment name
================

The parameter ``env_name`` is used to define the **name of working directory** generation. It is very useful to
difference several experiments in the same environment, for example.

Extra configuration
====================

Some parameters directly associated with the building model and simulator can be set as extra configuration 
as well, such as ``people occupant``, ``timesteps per simulation hour``, ``runperiod``, etc.

Like this **extra configuration context** can grow up in the future, this is specified in ``config_params`` argument.
It is a Python Dictionary where this values are specified. For more information about extra configuration
available for *Sinergym* visit section :ref:`Extra Configuration in Sinergym simulations`.

**************************************
Adding new weathers for environments
**************************************

*Sinergym* includes diverse weather files covering various climates worldwide to maximize experiment diversity.

To add a **new weather**:

1. Download an **EPW** and a corresponding **DDY** file from the `EnergyPlus page <https://energyplus.net/weather>`__.
   The *DDY* file specifies location and design day information.

2. Ensure both files have identical names, differing only in their extensions, and place them in 
   the `weathers <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/weather>`__ folder.

*Sinergym* will automatically adjust the ``SizingPeriod:DesignDays`` and ``Site:Location`` fields in 
the building model file using the *DDY* file for the added weather.

**************************************
Adding new buildings for environments
**************************************

Users can modify existing environments or create new environment definitions, incorporating new climates, 
action and observation spaces. Additionally, they have the option to use a different **building model** (epJSON file) 
than the ones currently supported.

To add new buildings for use with *Sinergym*, follow these steps:

1. **Add your building file** (*epJSON*) to the `buildings <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/buildings>`__.
   Ensure compatibility with EnergyPlus version used in *Sinergym*.
   If you are using an *IDF* file with an older version, it is advisable to update it with **IDFVersionUpdater** and then convert 
   it to *epJSON* format using **ConvertInputFormat**. Both tools are accessible in the EnergyPlus installation folder.

2. **Adjust building objects** such as ``RunPeriod`` and ``SimulationControl`` to suit user needs in Sinergym. 
   We suggest setting ``run_simulation_for_sizing_periods`` to ``No`` in ``SimulationControl``. ``RunPeriod`` 
   sets the episode length, which can be configured in the building file or Sinergym settings (see :ref:`runperiod`). 
   These modifications can be made in the *IDF* prior to step 1 or directly in the *epJSON* file.

3. We need to **identify the components** of the building that we want to observe and control, respectively. This is the most 
   challenging part of the process. Typically, the user is already familiar with the building and therefore knows the *name* 
   and *key* of the elements in advance. If not, the following process can be followed.

   To view the different ``OutputVariables`` and ``Meters``, a preliminary simulation with EnergyPlus can be conducted directly 
   without establishing any control flow. The output files, specifically the file with the *RDD* extension, can be consulted 
   to identify the possible observable variables.

   The challenge lies in knowing the names but not the possible *Keys* (EnergyPlus does not initially provide this information). 
   These names can be used to define the environment (see step 4). If the *Key* is incorrect, *Sinergym* will notify of the 
   error and provide a file called **data_available.txt** in the output, since it has already connected with the EnergyPlus API. This file will 
   contain all the **controllable schedulers** for the actions and all the **observable variables**, this time with their respective *Keys*, 
   enabling the correct definition of the environment.

4. Once this information is obtained, the next step is **defining the environment** using the building model. 
   We have several options:

  a. Use the *Sinergym* environment constructor directly. The arguments for building observation and 
     control are explained within the class and should be specified in the same format as the EnergyPlus API.

  b. Set up the configuration to register environment IDs directly. For detailed information on this, refer to 
     the documentation :ref:`Environments Configuration and Registration`. *Sinergym* will verify that the 
     established configuration is entirely correct and notify of any potential errors.

5. If you've used *Sinergym*'s registry, you'll have access to environment IDs paired with your building. Use them 
   with ``gym.make(<environment_id>)`` as usual. If you've created an environment instance directly, simply use 
   that instance to start interacting with the building.

.. note:: For obtain information about the environment instance with the new building model, see reference :ref:`Getting information about Sinergym environments`.

