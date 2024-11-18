############
Environments
############

**************************
Environments List
**************************

The **list of available environments** is the following:

+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Env. name                                          | Location        | IDF file                              | Weather type (*)           | Action space | Simulation period |
+====================================================+=================+=======================================+============================+==============+===================+
| Eplus-demo-v1                                      | Pittsburgh, USA | 5ZoneAutoDXVAV.idf                    |            \-              | Discrete(10) |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-discrete-v1                        | Arizona, USA    | 5ZoneAutoDXVAV.idf                    |        Hot dry (2B)        | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-discrete-v1                      | New York, USA   | 5ZoneAutoDXVAV.idf                    |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-discrete-v1                       | Washington, USA | 5ZoneAutoDXVAV.idf                    |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-continuous-v1                      | Arizona, USA    | 5ZoneAutoDXVAV.idf                    |        Hot dry (2B)        | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-continuous-v1                    | New York, USA   | 5ZoneAutoDXVAV.idf                    |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-continuous-v1                     | Washington, USA | 5ZoneAutoDXVAV.idf                    |      Cool marine (5C)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-discrete-stochastic-v1             | Arizona, USA    | 5ZoneAutoDXVAV.idf                    |        Hot dry (2B) (**)   | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-discrete-stochastic-v1           | New York, USA   | 5ZoneAutoDXVAV.idf                    |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-discrete-stochastic-v1            | Washington, USA | 5ZoneAutoDXVAV.idf                    |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-continuous-stochastic-v1           | Arizona, USA    | 5ZoneAutoDXVAV.idf                    |        Hot dry (2B) (**)   | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-continuous-stochastic-v1         | New York, USA   | 5ZoneAutoDXVAV.idf                    |      Mixed humid (4A) (**) | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-continuous-stochastic-v1          | Washington, USA | 5ZoneAutoDXVAV.idf                    |      Cool marine (5C) (**) | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-v1                   | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Hot dry (2B)          | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-v1                 | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Hot dry (2B)          | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-stochastic-v1        | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Hot dry (2B) (**)     | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-stochastic-v1      | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Hot dry (2B) (**)     | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-stochastic-v1      | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-v1               | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Mixed humid (4A)      | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-v1                 | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-stochastic-v1    | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Mixed humid (4A) (**) | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-stochastic-v1       | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-v1                | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Cool marine (5C)      | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-v1                  | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-stochastic-v1     | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf   |      Cool marine (5C) (**) | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-hot-discrete-v1                    | Arizona, USA    | ASHRAE9012016_Warehouse_Denver.idf    |      Hot dry (2B)          | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-hot-continuous-v1                  | Arizona, USA    | ASHRAE9012016_Warehouse_Denver.idf    |      Hot dry (2B)          | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-hot-discrete-stochastic-v1         | Arizona, USA    | ASHRAE9012016_Warehouse_Denver.idf    |      Hot dry (2B) (**)     | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-hot-continuous-stochastic-v1       | Arizona, USA    | ASHRAE9012016_Warehouse_Denver.idf    |      Hot dry (2B) (**)     | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-mixed-discrete-stochastic-v1       | New York, USA   | ASHRAE9012016_Warehouse_Denver.idf    |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-mixed-continuous-v1                | New York, USA   | ASHRAE9012016_Warehouse_Denver.idf    |      Mixed humid (4A)      | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-mixed-discrete-v1                  | New York, USA   | ASHRAE9012016_Warehouse_Denver.idf    |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-mixed-continuous-stochastic-v1     | New York, USA   | ASHRAE9012016_Warehouse_Denver.idf    |      Mixed humid (4A) (**) | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-cool-discrete-stochastic-v1        | Washington, USA | ASHRAE9012016_Warehouse_Denver.idf    |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-cool-continuous-v1                 | Washington, USA | ASHRAE9012016_Warehouse_Denver.idf    |      Cool marine (5C)      | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-cool-discrete-v1                   | Washington, USA | ASHRAE9012016_Warehouse_Denver.idf    |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-warehouse-cool-continuous-stochastic-v1      | Washington, USA | ASHRAE9012016_Warehouse_Denver.idf    |     Cool marine (5C) (**)  | Box(5)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-hot-discrete-v1                       | Arizona, USA    | ASHRAE9012016_OfficeMedium_Denver.idf |      Hot dry (2B)          | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-hot-continuous-v1                     | Arizona, USA    | ASHRAE9012016_OfficeMedium_Denver.idf |      Hot dry (2B)          | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-hot-discrete-stochastic-v1            | Arizona, USA    | ASHRAE9012016_OfficeMedium_Denver.idf |      Hot dry (2B) (**)     | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-hot-continuous-stochastic-v1          | Arizona, USA    | ASHRAE9012016_OfficeMedium_Denver.idf |      Hot dry (2B) (**)     | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-mixed-discrete-stochastic-v1          | New York, USA   | ASHRAE9012016_OfficeMedium_Denver.idf |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-mixed-continuous-v1                   | New York, USA   | ASHRAE9012016_OfficeMedium_Denver.idf |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-mixed-discrete-v1                     | New York, USA   | ASHRAE9012016_OfficeMedium_Denver.idf |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-mixed-continuous-stochastic-v1        | New York, USA   | ASHRAE9012016_OfficeMedium_Denver.idf |      Mixed humid (4A) (**) | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-cool-discrete-stochastic-v1           | Washington, USA | ASHRAE9012016_OfficeMedium_Denver.idf |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-cool-continuous-v1                    | Washington, USA | ASHRAE9012016_OfficeMedium_Denver.idf |      Cool marine (5C)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-cool-discrete-v1                      | Washington, USA | ASHRAE9012016_OfficeMedium_Denver.idf |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+
| Eplus-office-cool-continuous-stochastic-v1         | Washington, USA | ASHRAE9012016_OfficeMedium_Denver.idf |     Cool marine (5C) (**)  | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+---------------------------------------+----------------------------+--------------+-------------------+


(\*) Weather types according to `DOE's
classification <https://www.energycodes.gov/development/commercial/prototype_models#TMY3>`__.

(\*\*) In these environments, weather series change from episode to
episode. Gaussian noise with 0 mean and 2.5 std is added to the original
values in order to add stochastic.

*********************
Available Parameters
*********************

With the **environment constructor** we can configure the complete **context** of our environment 
for experimentation, either starting from one predefined by *Sinergym* shown in the 
table above or creating a new one.

.. literalinclude:: ../../../sinergym/envs/eplus_env.py
    :language: python
    :pyobject: EplusEnv.__init__

We will show which **parameters** are available and what their function is.

IDF file 
=========

The parameter *idf_file* is the path to *IDF* (Intermediate Data Format) 
file where *Energyplus* building model is defined.

*Sinergym* initially provides **"free" buildings**. This means that the *IDF* does not have the external 
interface defined and default components, such as the ``timesteps``, the ``runperiod``, the 
``location`` or ``DesignDays``. 

Depending on the rest of the parameters that make up the environment, the building model is **updated** 
by *Sinergym* automatically, changing those components that are necessary, such as the external interface that we 
have mentioned.

Once the building is configured, it is **copied** to the output folder of that particular experimentation 
and used by the simulator of that execution.

EPW file
=========

The parameter *weather_file* is the path to *EPW* (EnergyPlus Weather) file where **climate conditions** during 
a year is defined.

Initially, this file will not be copied to the specific output folder of the experiment, since the original 
file present in *Sinergym* can be used directly. However, the user can set a year-to-year **variability** in 
the climate (see section :ref:`Weather Variability`). In that case, the weather updated with such variability 
will be copied and used in the output folder. 

Depending on the climate that is set for the environment, some of building model components need to be **modified** 
in such a way that it is **compatible** with that weather. Therefore, *Sinergym* updates the ``DesignDays`` and ``Location`` 
fields automatically using the weather data, without the need for user intervention. 

Weather Variability
====================

**Weather variability** can be integrated into an environment using *weather_variability* parameter.

It implements the `Ornstein-Uhlenbeck process <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.4200&rep=rep1&type=pdf>`__
in order to introduce **noise** to the weather data episode to episode. Then, parameter established is a Python tuple of three variables
(*sigma*, *mu* and *tau*) whose values define the nature of that noise.

.. image:: /_static/weather_variability.png
  :scale: 120 %
  :alt: Ornstein-Uhlenbeck process noise with different hyperparameters.
  :align: center


Reward
=======

The parameter called *reward* is used to define the **reward class** (see section :ref:`Rewards`)
that the environment is going to use to calculate and return reward values each timestep.

Reward Kwargs
==============

Depending on the reward class that is specified to the environment, it may have **different parameters** 
depending on its type. In addition, if a user creates a new custom reward, it can have new parameters as well.

Moreover, depending on the building being used (*IDF* file) for the environment, the values of these reward parameters may 
need to be different, such as the comfort range or the energy and temperature variables of the simulation that 
will be used to calculate the reward.

Then, the parameter called *reward_kwargs* is a Python dictionary where we can **specify all reward class parameters** 
that they are needed. For more information about rewards, visit section :ref:`Rewards`.

Action Repeat
==============

The parameter called *act_repeat* is the number of timesteps that an **action is repeated** in the simulator, 
regardless of the actions it receives during that repetition interval. Default value is 1.

Maximum Episode Data Stored in Sinergym Output
===============================================

*Sinergym* stores all the output of an experiment in a folder organized in sub-folders for each episode 
(see section :ref:`Output format` for more information). Depending on the value of the parameter *max_ep_data_store_num*, 
the experiment will store the output data of the **last n episodes** set, where **n** is the value of the parameter.

In any case, if *Sinergym Logger* (See :ref:`Logger` section) is activate, ``progress.csv`` will be present with 
the summary data of each episode.

Observation/action spaces
===========================

Structure of observation and action space is defined in Environment constructor directly. 
This allows for a **dynamic definition** of these spaces. Let's see the fields required to do it:

- **observation_variables**: List of observation variables that simulator is going to process like an observation. 
  These variables names must follow the structure ``<variable_name>(<zone_name>)`` in order to register 
  them correctly. *Sinergym* will check for you that the variable names are correct with respect to 
  the building you are trying to simulate (*IDF* file). 
  To do this, it will look at the list found in the 
  `variables <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/variables>`__ 
  folder of the project (*RDD* file). 

- **observation_space**: Definition of the observation space following the **OpenAI gym standard**. 
  This space is used to represent all the observations variables that we have previously 
  defined. Remember that the **year, month, day and hour** are added by *Sinergym* later, 
  so space must be reserved for these fields in the definition. If an inconsistency is 
  found, *Sinergym* will notify you so that it can be fixed easily. 

- **action_variables**: List of the action variables that simulator is going to process 
  like schedule control actuator in the building model. These variables
  must be defined in the building model (*IDF* file) correctly before simulation. You can 
  modify **manually** in the *IDF* file or using our **action definition** field 
  in which you set what you want to control and *Sinergym* 
  takes care of modifying this file for you automatically. For more information about 
  this automatic adaptation in section :ref:`Action definition`.
                
- **action_space**: Definition of the action space following the **OpenAI gym standard**. 
  This definition can be discrete or continuous and must be consistent with 
  the previously defined action variables (*Sinergym* will show inconsistency as usual).

.. note:: In order to make environments more generic in DRL solutions. We have updated 
          action space for **continuous problems**. Gym action space is defined always 
          between [-1,1] and Sinergym **parse** this values to action space defined in 
          environment internally before to send it to EnergyPlus Simulator. 
          The method in charge of parse this values from [-1,1] to real action space is 
          called ``_setpoints_transform(action)`` in *sinergym/sinergym/envs/eplus_env.py*

- **action_mapping**: It is only necessary to specify it in **discrete** action spaces. 
  It is a dictionary that links an **index** to a specific configuration of values for 
  each action variable. 

As we have told, observation and action spaces are defined **dynamically** in *Sinergym* 
Environment constructor. Environment ID's registered in *Sinergym* use a **default** definition
set up in `constants.py <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/constants.py>`__.

As can be seen in environments observations, the **year, month, day and hour** are included in, 
but is not configured in default observation variables definition. 
This is because they are not variables recognizable by the simulator (*Energyplus*) as such 
and *Sinergym* does the calculations and adds them in the states returned as 
output by the environment. This feature is **common to all environments** available in 
*Sinergym* and all supported building designs. In other words, you don't need to 
add this variables (**year, month, day and hour**) to observation variables, but yes to 
the observation space.

As we told before, all environments ID's registered in *Sinergym* use its respectively 
**default action and observation spaces, variables and action definition**. 
However, you can **change** this values giving you the possibility of playing with different 
observation/action spaces in discrete and continuous environments in order to study how this 
affects the resolution of a building problem.

*Sinergym* has several **checkers** to ensure that there are no inconsistencies in the alternative 
specifications made to the default ones. In case the specification offered is wrong, 
*Sinergym* will launch messages indicating where the error or inconsistency is located.

.. note:: ``variables.cfg`` is a requirement in order to establish a connection between gym environment and Simulator 
           with a external interface (using *BCVTB*). Since *Sinergym* ``1.9.0`` version, it is created automatically using 
           action and observation space definition in environment construction.

Environment name
================

The parameter *env_name* is used to define the **name of working directory** generation.

Action definition
==================

Creating a **new external interface** to control different parts of a building is not a trivial task, 
it requires certain changes in the building model (*IDF*), configuration files for the external 
interface (``variables.cfg``), etc in order to control it.

The **changes in the building model** are **complex** due to depending on the building model we will have
available different zones and actuators. 

Thus, there is the possibility to add an **action definition** in environment instead of modify *IDF* 
directly about components or actuators changes required to control by external variables specified 
in :ref:`Observation/action spaces`.

For this purpose,  we have available *action_definition* parameter in environments. Its value is a 
dictionary with the next structure:

.. code:: python

    action_definition_example={  
      <controller_type>:[<controller1_definition>,<controller2_definition>, ...],
      <other_type>: ...
    }

The ``<controller_definition>`` will depend on the specific type of controller that we are 
going to create, we have the next support:

~~~~~~~~~~~~~~~~~~~~~~~~
Thermostat:DualSetpoint
~~~~~~~~~~~~~~~~~~~~~~~~

This controller has the next values in its definition:

- *name*: DualSetpoint resource name (str).

- *heating_name*: Heating setpoint name. This name should be an action variable defined 
  in your environment (str).

- *cooling_name*: Cooling setpoint name. This name should be an action variable defined
  in your environment (str).

- *heating_initial_value*: Initial value the heating thermostat initialize the simulation with (float).

- *cooling_initial_value*: Initial value the cooling thermostat initialize the simulation with (float).

- *zones*: An thermostat can manage several building zones at the same time. Then, you 
  can specify one or more zones (List(str)). If the zone name specified is not 
  exist in building, Sinergym will report the error.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ThermostatSetpoint:SingleHeating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This controller has the next values in its definition:

- *name*: SingleHeating Setpoint resource name (str).

- *heating_name*: Heating setpoint name. This name should be an action variable defined 
  in your environment (str).

- *heating_initial_value*: Initial value the heating thermostat initialize the simulation with.

- *zones*: An thermostat can manage several building zones at the same time. Then, you 
  can specify one or more zones (List(str)). If the zone name specified is not 
  exist in building, Sinergym will report the error.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ThermostatSetpoint:SingleCooling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This controller has the next values in its definition:

- *name*: SingleCooling Setpoint resource name (str).

- *cooling_name*: Cooling setpoint name. This name should be an action variable defined
  in your environment (str).

- *cooling_initial_value*: Initial value the cooling thermostat initialize the simulation with.

- *zones*: An thermostat can manage several building zones at the same time. Then, you 
  can specify one or more zones (List(str)). If the zone name specified is not 
  exist in building, Sinergym will report the error.

For an example about how to use it, see :ref:`Adding a new action definition`.

.. note:: If you want to create your own controller type compatibilities, 
          please see the method ``adapt_idf_to_action_definition`` from 
          `Config class <https://github.com/ugr-sail/sinergym/tree/main/sinergym/utils/config.py>`__.

Extra configuration
===================

Some parameters directly associated with the simulator can be set as extra configuration 
as well, such as ``people occupant``, ``timesteps per simulation hour``, ``runperiod``, etc.

Like this **extra configuration context** can grow up in the future, this is specified in *config_params* field.
It is a Python Dictionary where this values are specified. For more information about extra configuration
available for *Sinergym* visit section :ref:`Extra Configuration in Sinergym simulations`.

**************************************
Adding new buildings for environments
**************************************

As we have already mentioned, a user can change the already available environments or even create new environment 
definitions including new climates, action and observation spaces, etc. However, perhaps **the most complex thing 
to incorporate** into the project are **new building models** (*IDF* files) than the ones we support.

This section is intended to provide information if someone decides to add new buildings for use with *Sinergym*. 
The main steps you have to follow are the next:

1. Add your building (*IDF* file) to `buildings <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/buildings>`__. 
   That building model must be "free" as far as external interface is concerned if you plan to use Sinergym's **action 
   definition** which will modify the model for you before starting the 
   simulation automatically (see section :ref:`Action definition`).
   If you are going to control a component which is not supported by *Sinergym* currently, 
   you will have to update *IDF* manually before starting
   simulation. **Be sure that new IDF model version is compatible with EnergyPlus version**.

2. Add your own *EPW* file for weather conditions or use ours in environment constructor. 
   *Sinergym* will adapt ``DesignDays`` and ``Location`` in *IDF* file using *EPW* automatically.
   It is important to add the *DDY* file too, with the same name than *EPW* in order to
   read the ``DesignDay`` correctly.

3. *Sinergym* will check that observation variables specified in environments constructor are 
   available in the simulation before starting. In order to be able to do these checks, 
   you need to copy **RDD file** with the same name than *IDF* file (except extension) 
   to `variables <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/variables>`__. 
   To obtain this **RDD file** you have to run a simulation with *Energyplus* directly 
   and extract from output folder. 
   Make sure that **Output:VariableDictionary** object in *IDF* has the value *Regular* 
   in order to *RDD* file has the correct format for *Sinergym*.

4. Register your own environment ID `here <https://github.com/ugr-sail/sinergym/blob/main/sinergym/__init__.py>`__ 
   following the same structure than the rest.

5. Now, you can use your own environment ID with ``gym.make()`` like our documentation examples.

