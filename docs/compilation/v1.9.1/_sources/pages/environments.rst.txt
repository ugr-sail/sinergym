############
Environments
############

**************************
Environments List
**************************

The list of available environments is the following:

+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Env. name                                          | Location        | IDF file                            | Weather type (*)           | Action space | Simulation period |
+====================================================+=================+=====================================+============================+==============+===================+
| Eplus-demo-v1                                      | Pittsburgh, USA | 5ZoneAutoDXVAV.idf                  |            \-              | Discrete(10) |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-discrete-v1                        | Arizona, USA    | 5ZoneAutoDXVAV.idf                  |        Hot dry (2B)        | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-discrete-v1                      | New York, USA   | 5ZoneAutoDXVAV.idf                  |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-discrete-v1                       | Washington, USA | 5ZoneAutoDXVAV.idf                  |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-continuous-v1                      | Arizona, USA    | 5ZoneAutoDXVAV.idf                  |        Hot dry (2B)        | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-continuous-v1                    | New York, USA   | 5ZoneAutoDXVAV.idf                  |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-continuous-v1                     | Washington, USA | 5ZoneAutoDXVAV.idf                  |      Cool marine (5C)      | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-discrete-stochastic-v1             | Arizona, USA    | 5ZoneAutoDXVAV.idf                  |        Hot dry (2B) (**)   | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-discrete-stochastic-v1           | New York, USA   | 5ZoneAutoDXVAV.idf                  |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-discrete-stochastic-v1            | Washington, USA | 5ZoneAutoDXVAV.idf                  |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-hot-continuous-stochastic-v1           | Arizona, USA    | 5ZoneAutoDXVAV.idf                  |        Hot dry (2B) (**)   | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-mixed-continuous-stochastic-v1         | New York, USA   | 5ZoneAutoDXVAV.idf                  |      Mixed humid (4A) (**) | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-5Zone-cool-continuous-stochastic-v1          | Washington, USA | 5ZoneAutoDXVAV.idf                  |      Cool marine (5C) (**) | Box(2)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-v1                   | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf |      Hot dry (2B)          | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-v1                 | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf |      Hot dry (2B)          | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-discrete-stochastic-v1        | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf |      Hot dry (2B) (**)     | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-hot-continuous-stochastic-v1      | Arizona, USA    | 2ZoneDataCenterHVAC_wEconomizer.idf |      Hot dry (2B) (**)     | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-stochastic-v1      | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-v1               | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A)      | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-discrete-v1                 | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-mixed-continuous-stochastic-v1    | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A) (**) | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-stochastic-v1       | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-v1                | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A)      | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-discrete-v1                  | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-datacenter-cool-continuous-stochastic-v1     | Washington, USA | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A) (**) | Box(4)       |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-mixed-discrete-v1                  | New York, USA   | IWMullion.idf                       |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-mixed-discrete-stochastic-v1       | New York, USA   | IWMullion.idf                       |      Mixed humid (4A) (**) | Discrete(10) |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-mixed-continuous-v1                | New York, USA   | IWMullion.idf                       |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-cool-discrete-v1                   | Washington, USA | IWMullion.idf                       |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-mixed-continuous-stochastic-v1     | New York, USA   | IWMullion.idf                       |      Mixed humid (4A) (**) | Box(2)       |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-cool-discrete-stochastic-v1        | Washington, USA | IWMullion.idf                       |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-cool-continuous-v1                 | Washington, USA | IWMullion.idf                       |      Cool marine (5C)      | Box(2)       |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+
| Eplus-IWMullion-cool-continuous-stochastic-v1      | Washington, USA | IWMullion.idf                       |      Cool marine (5C) (**) | Box(2)       |   01/01 - 31/03   |
+----------------------------------------------------+-----------------+-------------------------------------+----------------------------+--------------+-------------------+

(\*) Weather types according to `DOE's
classification <https://www.energycodes.gov/development/commercial/prototype_models#TMY3>`__.

(\*\*) In these environments, weather series change from episode to
episode. Gaussian noise with 0 mean and 2.5 std is added to the original
values in order to add stochastic.

**************************
Observation/action spaces
**************************

Structure of observation and action space is defined in Environment constructor directly. This allows for a **dynamic definition** of these spaces. Let's see the fields required to do it:

.. literalinclude:: ../../../sinergym/envs/eplus_env.py
    :language: python
    :pyobject: EplusEnv.__init__


- **observation_variables**: List of observation variables that simulator is going to process like an observation. These variables names must 
                             follow the structure `<variable_name>(<zone_name>)` in order to register them correctly. Sinergym will check for
                             you that the variable names are correct with respect to the building you are trying to simulate (IDF file). 
                             To do this, it will look at the list found in the `variables <https://github.com/jajimer/sinergym/tree/main/sinergym/data/variables>`__ folder of the project (**RDD** file). 

- **observation_space**: Definition of the observation space following the **gym standard**. This space is used to represent all the observations 
                         variables that we have previously defined. Remember that the **year, month, day and hour** are added by Sinergym later, 
                         so space must be reserved for these fields in the definition. If an inconsistency is found, Sinergym will notify you 
                         so that it can be fixed. 

- **action_variables**: List of the action variables that simulator is going to process like schedule control actuator in the building model. These variables
                        must be defined in the building model (IDF file) correctly before simulation. You can modify **manually** in the IDF file or using
                        our **action definition** extra configuration field in which you set what you want to control and Sinergym takes care of modifying 
                        this file for you automatically. For more information about this automatic adaptation in section HIPERENLACE.
                
- **action_space**: Definition of the action space following the **gym standard**. This definition can be discrete or continuous and must be consistent with 
                    the previously defined action variables (Sinergym will show inconsistency as usual).

.. note:: In order to make environments more generic in DRL solutions. We have updated action space for **continuous problems**. Gym action space is defined always between [-1,1] 
          and Sinergym **parse** this values to action space defined in environment internally before to send it to EnergyPlus Simulator. The method in charge of parse this values 
          from [-1,1] to real action space is called ``_setpoints_transform(action)`` in *sinergym/sinergym/envs/eplus_env.py*

- **action_mapping**: It is only necessary to specify it in discrete action spaces. It is a dictionary that links an index to a specific configuration of values for each action variable. 

Specification
~~~~~~~~~~~~~~

As we have told, Observation and action spaces are defined **dinamically** in Sinergym Environment constructor. Environmet ID's registered in Sinergym use a **default** definition
set up in `constants.py <https://github.com/jajimer/sinergym/tree/main/sinergym/utils/constants.py>`__.

As can be seen in environments observations, the **year, month, day and hour** are included in, but is not configured in default observation variables definition. 
This is because they are not variables recognizable by the simulator as such (Energyplus) and Sinergym does the calculations and adds them in the states returned as 
output by the environment. This feature is common to all environments available in Sinergym and all supported building designs. In other words, you don't need to 
add this variables (**year, month, day and hour**) to observation variables but yes to the observation space.

As we told before, all environments ID's registered in Sinergym use its respectively **default action and observation spaces, variables and definition**. 
However, you can **change** this values giving you the possibility of playing with different observation/action spaces in discrete and continuous environments in order to study how this affects the resolution of a building problem.

Sinergym has several checkers to ensure that there are no inconsistencies in the alternative specifications made to the default ones. In case the specification offered is wrong, Sinergym will launch messages indicating where the error or inconsistency is located.

.. note:: `variables.cfg` is a requirement in order to stablish a connection between gym environment and Simulator 
           with a external interface (using BCVTB). Since Sinergym `1.9.0` version, it is created automatically using 
           action and observation space definition in environment construction.

**************************************
Adding new buildings for environments
**************************************

This section is intended to provide information if someone decides to add new buildings for use with Sinergym. The main steps you have to follow are the next:

1. Add your building (IDF file) to `buildings <https://github.com/jajimer/sinergym/tree/main/sinergym/data/buildings>`__. 
   As mentioned in section :ref:`Observation/action spaces`, the IDF must be previously adapted to your action space or use our **action definition** (section HIPERENLACE) 
   to be automatically adapted to the action space and variables you have designed for it, as long as we provide support for the specific component you want to control..

2. Add your own EPW file for weather conditions or use ours in environment constructor. Sinergym will adapt `DesignDays` in IDF file using EPW automatically.

3. Sinergym will check that observation variables specified are available in the simulation before starting. In order to be able to do these checks, 
   you need to copy **RDD file** with the same name than IDF file (except extension) to `variables <https://github.com/jajimer/sinergym/tree/main/sinergym/data/variables>`__. 
   To obtain this **RDD file** you have to run a simulation with *Energyplus* directly and extract from output folder.

4. Register your own environment ID `here <https://github.com/jajimer/sinergym/blob/main/sinergym/__init__.py>`__ following the same structure than the rest.

5. Now, you can use your own environment ID with `gym.make()` like our documentation examples.

