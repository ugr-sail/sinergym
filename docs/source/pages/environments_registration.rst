###########################################
Environments configuration and registration
###########################################

New *Sinergym* environments can be created using the constructor of the `EplusEnv <https://github.com/ugr-sail/sinergym/blob/main/sinergym/envs/eplus_env.py>`__ class and the parameters listed in section :ref:`Available Parameters`.

Since multiple environments can be generated from the same building, *Sinergym* provides an automated way for creating and registering them. Using a JSON file located in `sinergym/data/default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__, a predefined set of parameters for each possible configuration will be applied to generate the environments. Each environment will be assigned a unique ID and automatically registered in Gymnasium.

This section outlines the structure of these JSON configuration files. Moreover, this structure allows for the definition of observation variables (``time_variables``, ``variables`` and ``meters``) and action variables (``actuators``). Instead of manually specifying these in the EnergyPlus Python API format as part of the environment constructor, Sinergym will read this simplified structure and automatically convert it into the appropriate EnergyPlus Python API format.

+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Key**              | **Optional** | **Description**                                                                                                                                                           |
+======================+==============+===========================================================================================================================================================================+
| id_base              | No           | Building base name / ID model.                                                                                                                                            |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| building_file        | No           | Building model file located in ``sinergym/data/buildings/``.                                                                                                              |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| weather_specification| No           | Set of weather files used to generate different environments. Located in ``sinergym/data/weather/``.                                                                      |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| config_params        | Yes          | Extra environments parameters (optional).                                                                                                                                 |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| weather_variability  | Yes          | Indicates if additional versions of all environments adding episodic weather noise will be created.                                                                       |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| max_ep_data_store_num| Yes          | Maximum number of episodes for which data will be stored. Default: 10.                                                                                                    |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| time_variables       | No           | ``time_variables`` list definition.                                                                                                                                       |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| variables            | No           | ``variables`` dict definition.                                                                                                                                            |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| meters               | No           | ``meters`` dict definition.                                                                                                                                               |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| actuators            | No           | ``actuators`` dict definition.                                                                                                                                            |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| action_space         | No           | Gymnasium action space definition.                                                                                                                                        |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| action_space_discrete| Yes          | Allows the creation of discrete action-space versions of the environments.                                                                                                |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| only_discrete        | Yes          | Used in composite action space problems where continuous values                                                                                                           |
|                      |              | must be discretized. If an action space and a discrete space based on it                                                                                                  |
|                      |              | have been defined (required; see :ref:`DiscretizeEnv`), you can only                                                                                                      |
|                      |              | register the discrete version using this flag.                                                                                                                            |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| reward               | No           | Reward class name.                                                                                                                                                        |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| reward_kwargs        | No           | Reward kwargs for the Reward class constructor.                                                                                                                           |
+----------------------+--------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


.. important:: Sinergym uses continuous action spaces by default, rather than discrete ones.
               If needed, environments are discretized afterwards using a wrapper. 
               For additional details, refer to :ref:`DiscretizeEnv`.

Having presented the available parameters, the following is a JSON example of how to use them: 

.. literalinclude:: ../../../sinergym/data/default_configuration/5ZoneAutoDXVAV.json
    :language: json

Based on this JSON configuration for the building ``5ZoneAutoDXVAV.epJSON``, the following 
environments will be automatically created: ::

    [
     'Eplus-5zone-hot-continuous-v1', 
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

Note how the weather specification enables the creation of multiple environments based on different weather files. These files must be located in the appropriate folder, and their keys are used to construct the environment IDs.

For example, if the discrete space or weather variability are not defined, the discrete and stochastic versions of the environment will not be created. 

.. warning:: For discrete environments, an action mapping must be defined in 
             `constants.py <https://github.com/ugr-sail/sinergym/blob/main/sinergym/utils/constants.py>`__ 
             named as ``DEFAULT_<upper case id_base>_DISCRETE_FUNCTION``.

***********************
Variables specification
***********************

The ``variables`` field follows a specific format to efficiently define all observed variables in the environment. Variable names and keys can be provided as either individual strings or lists of strings. The functionality is illustrated in the following graph:

.. image:: /_static/json_variables_conf.png
  :scale: 70 %
  :alt: *variables* field format in the JSON configuration of *Sinergym* environments
  :align: center


|

During registration, *Sinergym* parses the information introduced in the ``variables`` parameter to the environment constructor, previously adapting it to the format used by the EnergyPlus Python API. A similar process is followed with the ``meters`` and ``actuators`` fields.





