############
Environments
############

+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Env. name                            | Location        | IDF file           | Weather type (*)           | Action space | Simulation period |
+======================================+=================+====================+============================+==============+===================+
| Eplus-demo-v1                        | Pittsburgh, USA | 5ZoneAutoDXVAV.idf |            \-              | Discrete(10) |   01/01 - 31/03   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-discrete-hot-v1                | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B)        | Discrete(10) |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-discrete-mixed-v1              | New York, USA   | 5ZoneAutoDXVAV.idf |      Mixed humid (4A)      | Discrete(10) |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-discrete-cool-v1               | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C)      | Discrete(10) |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-continuous-hot-v1              | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B)        | Box(2)       |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-continuous-mixed-v1            | New York, USA   | 5ZoneAutoDXVAV.idf |      Mixed humid (4A)      | Box(2)       |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-continuous-cool-v1             | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C)      | Box(2)       |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-discrete-stochastic-cool-v1    | Washington, USA | 5ZoneAutoDXVAV.idf |      Cool marine (5C) (**) | Discrete(10) |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+
| Eplus-continuous-stochastic-hot-v1   | Arizona, USA    | 5ZoneAutoDXVAV.idf |        Hot dry (2B) (**)   | Box(2)       |   01/01 - 31/12   |
+--------------------------------------+-----------------+--------------------+----------------------------+--------------+-------------------+


(\*) Weather types according to `DOE's
classification <https://www.energycodes.gov/development/commercial/prototype_models#TMY3>`__.

(\*\*) In these environments, weather series change from episode to
episode. Gaussian noise with 0 mean and 2.5 std is added to the original
values in order to add stochasticity.