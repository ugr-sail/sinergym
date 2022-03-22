############
Environments
############

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
| Eplus-datacenter-mixed-continuous-stochastic-v1    | New York, USA   | 2ZoneDataCenterHVAC_wEconomizer.idf |      Mixed humid (4A) (**) | Box(4)       |   01/01 - 31/12   |
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

Observation and action spaces are defined in `sinergym/sinergym/data/variables <https://github.com/jajimer/sinergym/tree/main/sinergym/data/variables>`__ specifically for each IDF file. Therefore, external interface with simulation is defined in the same way.

This is a definition example for 5ZoneAutoDXVAV.idf and its variants:

.. literalinclude:: ../../../sinergym/data/variables/5ZoneAutoDXVAV_spaces.cfg
    :language: xml

This gives you the possibility of playing with different observation/action spaces in discrete and continuous environments in order to study how this affects the resolution of a building problem.
Inner each environment it is known what configuration file must read and spaces will be defined automatically, so you should not worry about anything.

In order to make environments more generic in DRL solutions. We have updated action space for continuous problem. Gym action space is defined always between [-1,1] and Sinergym parse this values to action space defined in configuration file internally.

The function in charge of reading those configuration files is ``parse_observation_action_space(space_file)`` in *sinergym/sinergym/utils/common.py*

.. note:: *Set up observation and action spaces in environment constructor dynamically could be upgrade in the future. Stay tuned for upcoming releases!*
