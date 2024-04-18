############################################
Environments Configuration and Registration
############################################

When defining a new environment, we can use the *Sinergym* environment constructor and fill the parameters 
explained in section :ref:`Available Parameters`.

Many environments can be created based on the same building, depending on its configurations. Therefore, 
creating (or registering Gymnasium IDs) for all of them can be tedious.

*Sinergym* has a system that automates this process. From a JSON file hosted in 
`sinergym/data/default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__, 
a set of parameters for each of the possible configurations will be built, along with an associated ID, and will be 
registered in Gymnasium automatically.

This section will explain the structure of these JSON configuration definitions. Additionally, this structure 
facilitates the definition of observation variables (``time_variables``, ``variables`` and ``meters``) and 
action variables (``actuators``). Instead of defining in the EnergyPlus Python API format like the environment 
constructor, *Sinergym* will read this simpler structure and parse it to the EnergyPlus Python API format 
automatically.

+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|        **Key**        | **Optional** |                                                                              **Description**                                                                              |
+=======================+==============+===========================================================================================================================================================================+
| id_base               | No           | Base name to refer to the ID's with this building model.                                                                                                                  |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| building_file         | No           | Building model file allocated in ``sinergym/data/buildings/``.                                                                                                            |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| weather_specification | No           | Set of weather to generate an environment with each one, allocated in ``sinergym/data/weather/``.                                                                         |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| config_params         | Yes          | Extra parameters for the environments; it is optional.                                                                                                                    |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| variation             | Yes          | Create additionally a version of all environments with stochasticity in weather.                                                                                          |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| max_ep_data_store_num | Yes          | Max storage in *Sinergym* episodes, by default 10.                                                                                                                        |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| time_variables        | No           | ``time_variables`` list definition.                                                                                                                                       |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| variables             | No           | ``variables`` dict definition.                                                                                                                                            |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| meters                | No           | ``meters`` dict definition.                                                                                                                                               |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| actuators             | No           | ``actuators`` dict definition.                                                                                                                                            |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| action_space          | No           | Gymnasium action space definition.                                                                                                                                        |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| action_space_discrete | Yes          | If you want that *Sinergym* auto-generate a discrete version of environments, you should write this space too.                                                            |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| only_discrete         | Yes          | If you have specified action_space and a discrete_space based on action_space (required, see :ref:`DiscretizeEnv`), you can only register discrete version with this flag |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| reward                | No           | Reward class name to use.                                                                                                                                                 |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| reward_kwargs         | No           | Reward kwargs for Reward class constructor in dict format.                                                                                                                |
+-----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. important:: Sinergym requires a continuous action space, but not a discrete one. 
               It begins with continuous environments and discretizes them using a wrapper. 
               For more details, refer to :ref:`DiscretizeEnv`.


These are the keys of the JSON. An example might make it clearer:

.. literalinclude:: ../../../sinergym/data/default_configuration/5ZoneAutoDXVAV.json
    :language: json

With this JSON configuration for the building ``5ZoneAutoDXVAV.epJSON``, the following 
environments will be automatically configured ::

    ['Eplus-5zone-hot-continuous-v1', 
     'Eplus-5zone-hot-discrete-v1', 
     'Eplus-5zone-hot-continuous-stochastic-v1', 
     'Eplus-5zone-hot-discrete-stochastic-v1', 
     'Eplus-5zone-mixed-continuous-v1', 
     'Eplus-5zone-mixed-discrete-v1', 
     'Eplus-5zone-mixed-continuous-stochastic-v1', 
     'Eplus-5zone-mixed-discrete-stochastic-v1', 
     'Eplus-5zone-cool-continuous-v1', 
     'Eplus-5zone-cool-discrete-v1', 
     'Eplus-5zone-cool-continuous-stochastic-v1', 
     'Eplus-5zone-cool-discrete-stochastic-v1'
    ]

For example, if you don't define discrete space or variation, the discrete and stochastic versions will 
not appear in the list.

.. warning:: For discrete environments, an action mapping must be defined in 
             `constants.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/constants.py>`__ 
             named as DEFAULT_<id_base in upper case>_DISCRETE_FUNCTION for correct registration.


****************************
Weather Specification field
****************************

The weather specification is set up to generate multiple environments based on the defined weathers. 
Weather files must be in the correct folder, and the keys are used to define the final name 
in the environment's IDs.


*****************
Variables field
*****************

The ``variables`` field uses a specific format to quickly define all the observed variables 
in the environment. The variable names and keys can be either an individual string or a list of 
strings. The following graph explains its functionality:

.. image:: /_static/json_variables_conf.png
  :scale: 70 %
  :alt: Configuration for *variables* in json configuration for *Sinergym* environments.
  :align: center

*Sinergym* will parse this information into the variables parameter for the environment constructor 
(similar to the EnergyPlus Python API) during registration. The same process is applied to the 
``meters`` and ``actuators`` fields, although they are simpler.





