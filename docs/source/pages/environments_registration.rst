############################################
Environments Configuration and Registration
############################################

When defining a new environment, we can use the *Sinergym* environment constructor and 
fill the parameters that we explained in section :ref:`Available Parameters`.

Many environments can be made based on the same building, depending on its configurations. 
Therefore, this can be tedious to create (or register Gymnasium ID's) of all of them.

*Sinergym* has a system that automates this process. From a JSON file hosted in 
`sinergym/data/default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__, 
a set of parameters for each of the possible configurations will be built, along with an associated 
ID, and will be registered in gymnasium automatically.

The structure of these JSON configuration definitions will be explained in this section. Additionally, this structure facilitates 
the definition of observation variables (``time_variables``, ``variables`` and ``meters``) and action variables (``actuators``).
Instead of defining in EnergyPlus Python API format like environment constructor, *Sinergym* will read this simpler structure and parse
to EnergyPlus Python API format automatically.

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

.. important:: As you can see, the continuous action space is mandatory, while the discrete one is not. 
               This is because Sinergym starts from continuous environments and then discretizes them 
               through a wrapper. For more information see :ref:`DiscretizeEnv`.


These are the keys of the JSON, an example could be more intuitive:

.. literalinclude:: ../../../sinergym/data/default_configuration/5ZoneAutoDXVAV.json
    :language: json

With this JSON configuration for the building ``5ZoneAutoDXVAV.epJSON``, we will have the next environment automatically configured ::

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

For example, if you don't define discrete space or variation, the discrete and stochastic versions will not appear in the list.

.. warning:: For Discrete environments, it must be defined an action mapping in `constants.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/constants.py>`__
          with the name DEFAULT_<id_base in upper case>_DISCRETE_FUNCTION to register correctly.


****************************
Weather Specification field
****************************

Weather specification is configured to generate several environments depending on the weathers defined. Weather files must be in
the correct folder and the keys are used in order to define the final name in environment's ID's.

*****************
Variables field
*****************

``variables`` field have a specific format in order to define all the variables observed in the environment faster. The variable names and keys
can be an individual str or a list of str. The next graph explain how its functionality is:

.. image:: /_static/json_variables_conf.png
  :scale: 70 %
  :alt: Configuration for *variables* in json configuration for *Sinergym* environments.
  :align: center

*Sinergym* will parse this information to variables parameter to env constructor (same that EnergyPlus Python API) in the registration.
The same is done with ``meters`` and ``actuators`` fields, but they are simpler.





