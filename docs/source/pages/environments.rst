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

- `observation_variables`: List of observation variables

Specification
~~~~~~~~~~~~~~

As we have told, Observation and action spaces are defined **dinamically** in Sinergym Environment constructor. Environmet ID's registered in Sinergym use a **default** definition
set up in `constants.py <https://github.com/jajimer/sinergym/tree/main/sinergym/utils/constants.py>`__.

As can be seen in environments observations, the **year, month, day and hour** are included in, but is not configured in default observation variables definition. 
This is because they are not variables recognizable by the simulator as such (Energyplus) and Sinergym does the calculations and adds them in the states returned as 
output by the environment. This feature is common to all environments available in Sinergym and all supported building designs. In other words, you don't need to 
add this variables (**year, month, day and hour**) to observation variables.

As we told before, all environments ID's registered in Sinergym use its respectively **default action and observation spaces, variables and definition**. 
However, you can **change** this values giving you the possibility of playing with different observation/action spaces in discrete and continuous environments in order to study how this affects the resolution of a building problem.

.. note:: If you are interested in modifying default action and observation specification of our environments, please visit section HIPERENLACE.

Sinergym has several checkers to ensure that there are no inconsistencies in the alternative specifications made to the default ones. In case the specification offered is wrong, Sinergym will launch messages indicating where the error or inconsistency is located.

Gym Action Space Note
~~~~~~~~~~~~~~~~~~~~~~

In order to make environments more generic in DRL solutions. We have updated action space for continuous problem. Gym action space is defined always between [-1,1] and Sinergym **parse** this values to action space defined in environment internally before to send it to EnergyPlus Simulator.

The method in charge of parse this values from [-1,1] to real action space is called ``_setpoints_transform(action)`` in *sinergym/sinergym/envs/eplus_env.py*

**************************************
Adding new buildings for environments
**************************************

Explain RDD information

