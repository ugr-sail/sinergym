############
Environments
############

*Sinergym* generates multiple environments for each building, each defined by a unique configuration that specifies the control problem to be addressed. To view the list of available environment IDs, it is recommended to use the provided method:

.. code-block:: python

  # This script is available in scripts/consult_environments.py
  import sinergym
  import gymnasium as gym
  
  print(sinergym.__version__)
  print(sinergym.__ids__)

  # Make and consult environment
  env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
  print(env.info())

Environment names follow the format ``Eplus-<building-id>-<weather-id>-<control_type>-<stochastic (optional)>-v1``.  
These identifiers provide a general summary of the environment's characteristics. For more detailed information about a specific environment, use the `info` method as shown in the example code.

.. important:: Environments are automatically generated using JSON configuration files
               for each building. This eliminates the need to manually register each 
               environment ID or set parameters directly in the environment constructor.
               For more information, see :ref:`Environments Configuration and Registration`.

.. note:: Discrete environments are fully customizable. By default, these environments use a basic control scheme.
          However, you can opt for a continuous environment and apply custom discretization using our dedicated wrapper. For further details, refer to :ref:`DiscretizeEnv`.

.. note:: For additional details on buildings (epJSON) and weather (EPW) configuration, see :ref:`Buildings` and :ref:`Weathers` sections, respectively.

********************
Available parameters
********************

The **environment constructor** allows you to fully configure the **context** of an environment for experimentation. You can either start with a predefined setup provided by *Sinergym* or create a completely new one.

*Sinergym* initially supplies **non-configured** buildings and weather files. Based on the arguments provided, these files are automatically updated by *Sinergym* to accommodate the specified features. For example: 

- Selecting a different weather file updates the building's location and simulated days.  
- Adding new observation variables modifies the ``Output:Variable`` and ``Output:Meter`` fields.  
- If weather variability is enabled, a weather file with episodic random noise will be used.  

These updated versions of the building and weather files are saved in the *Sinergym* output folder, while the original files remain untouched. 

The following subsections will detail the **parameters** available and their respective functions.

Building file 
=============

The ``building_file`` parameter refers to the *epJSON* file, an `adaptation <https://energyplus.readthedocs.io/en/latest/schema.html>`__ of the *IDF* (Intermediate Data Format) used to define *EnergyPlus* building models.

Before starting the simulation, *Sinergym* performs a preparatory step to adapt the building model. For more details, refer to the *Modeling* component in the *Sinergym* backend diagram.

Weather files
=============

The ``weather_file`` parameter specifies the *EPW* (*EnergyPlus* Weather) file, which defines the **climate conditions** for a full year. 

This parameter can be provided as a single weather file name (``str``) or as a list of multiple weather files (``List[str]``). When multiple files are specified, *Sinergym* will randomly select one *EPW* file for each episode and automatically adapt the building model accordingly. This feature adds complexity to the environment, if desired.

The weather file used in each episode is saved in the *Sinergym* episode output folder. If **variability** (see section :ref:`Weather Variability`) is enabled, the stored *EPW* file will include the corresponding noise adjustments.

Weather variability
~~~~~~~~~~~~~~~~~~~

**Weather variability** can be added to an environment using the ``weather_variability`` parameter. 

This feature utilizes an `Ornstein-Uhlenbeck process <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.710.4200&rep=rep1&type=pdf>`__  to introduce **random noise** into the weather data on an episode-by-episode basis. This noise is specified as a Python dictionary, where each key is the name of an EPW column, and the corresponding value is a tuple of three variables (*sigma*, *mu*, and *tau*) that define the characteristics of the noise. This enables to apply different noise configurations to different variables of the weather data.

Starting with *Sinergym* v3.6.2, the weather data column names (or variable names) are generated using the ``Weather`` class from the `epw module <https://pypi.org/project/epw/>`__. The list of available variable names is as follows:

- ``Year``, ``Month``, ``Day``, ``Hour``, ``Minute``,
  ``Data Source and Uncertainty Flags``, ``Dry Bulb Temperature``,
  ``Dew Point Temperature``, ``Relative Humidity``,
  ``Atmospheric Station Pressure``, ``Extraterrestrial Horizontal Radiation``,
  ``Extraterrestrial Direct Normal Radiation``,
  ``Horizontal Infrared Radiation Intensity``,
  ``Global Horizontal Radiation``, ``Direct Normal Radiation``,
  ``Diffuse Horizontal Radiation``, ``Global Horizontal Illuminance``,
  ``Direct Normal Illuminance``, ``Diffuse Horizontal Illuminance``,
  ``Zenith Luminance``, ``Wind Direction``, ``Wind Speed``, ``Total Sky Cover``,
  ``Opaque Sky Cover (used if Horizontal IR Intensity missing)``,
  ``Visibility``, ``Ceiling Height``, ``Present Weather Observation``,
  ``Present Weather Codes``, ``Precipitable Water``, ``Aerosol Optical Depth``,
  ``Snow Depth``, ``Days Since Last Snowfall``, ``Albedo``,
  ``Liquid Precipitation Depth``, ``Liquid Precipitation Quantity``

.. note:: If you are using an older version of Sinergym, the weather data columns or variables names is
          generated with the *opyplus* ``WeatherData`` class, for more  information about the available variable
          names with *opyplus*, visit `opyplus documentation <https://opyplus.readthedocs.io/en/2.0.7/quickstart/index.html#weather-data-epw-file>`__.

.. image:: /_static/ornstein_noise.png
  :scale: 80 %
  :alt: Ornstein-Uhlenbeck process noise with different hyperparameters.
  :align: center

Reward
======

The `reward` parameter specifies the **reward class** (refer to section :ref:`Rewards`) that the environment will use to compute and return scalar reward values at each timestep.

Reward kwargs
~~~~~~~~~~~~~

The ``reward_kwargs`` parameter is a Python dictionary used to define **all the arguments required by the reward class** specified for the environment. 

The arguments may vary depending on the type of reward class chosen. Additionally, if a user creates a custom reward class, this parameter can include any new arguments needed for that implementation. 

Furthermore, these arguments may need to be adjusted based on the building used in the environment. For instance, parameters like the comfort range or the energy and temperature variables used to compute the reward might differ between buildings.

For more details about rewards, refer to section :ref:`Rewards`.

Maximum episode data stored in Sinergym output
==============================================

*Sinergym* stores all experiment outputs in a folder, which is organized into sub-folders for each episode (see section :ref:`Sinergym output` for further details). The ``env_name`` parameter is utilized to generate the **working directory name**, facilitating differentiation between multiple experiments within the same environment.

The parameter ``max_ep_data_store_num`` controls the number of episodes' output data that will be retained. Specifically, the experiment will store the output of the last ``n`` episodes, where ``n`` is defined by this parameter.

If *Sinergym*'s CSV storage feature is enabled (refer to section :ref:`CSVLogger`), a ``progress.csv`` file will be generated. This file contains summary data for each episode.

Time variables
==============

The *EnergyPlus* Python API offers several methods to extract information about the ongoing simulation time. The ``time_variables`` argument is a list where you can specify the names of the 
`API methods <https://energyplus.readthedocs.io/en/latest/datatransfer.html#datatransfer.DataExchange>`__  with the values to be included in the observations.

By default, *Sinergym* environments include the time variables ``month``, ``day_of_month`` and ``hour``.

Variables
=========

The ``variables`` argument is a dictionary in which it is specified the ``Output:Variable`` entries to be included in the environment's observation. The format for each element, so that *Sinergym* can process it correctly, is as follows:

.. code-block:: python

  variables = {
    # <custom_variable_name> : (<"Output:Variable" original name>,<variable_key>),
    # ...
  }

.. note:: For more information about the available variables in an environment, execute a default simulation with 
          *EnergyPlus* and check the RDD file generated in the output.

Meters
======

In a similar way, the argument ``meters`` is a dictionary in which we can specify the ``Output:Meter``'s we want to include in the environment observation. 
The format of each element must be the following:

.. code-block:: python

  meters = {
    # <custom_meter_name> : <"Output:Meter" original name>,
    # ...
  }

.. note:: For more information about the available meters in an environment, execute a default simulation with
          *EnergyPlus* and see the MDD and MTD files generated in the output.

Actuators
=========

The argument called ``actuators`` is a dictionary in which we specify the actuators to be controlled. The format must be the following:

.. code-block:: python

  actuators = {
    # <custom_actuator_name> : (<actuator_type>,<actuator_value>,<actuator_original_name>),
    # ...
  }

.. important:: Actuators that have not been specified will be controlled by the building's default schedulers.

.. note:: For more information about the available actuators in an environment, execute a default control with
          *Sinergym* directly (i.e., with an empty action space) and check the file ``data_available.txt`` generated.

Action space
============

In *Sinergym*, the environment's observation and action spaces are defined through the arguments ``time_variables``, ``variables``, ``meters``, and ``actuators``. While the observation space (composed of ``time_variables``, ``variables``, and ``meters``) is automatically generated, the action space (defined by the ``actuators``) requires explicit definition to establish the range of values supported by the Gymnasium interface or the number of discrete values in a discrete environment.

.. image:: /_static/spaces_elements.png
  :scale: 50 %
  :alt: *EnergyPlus* API components that compose observation and action spaces in *Sinergym*.
  :align: center

|

The ``action_space`` argument adheres to the Gymnasium standard and must be a continuous space (``gym.spaces.Box``) due to the *EnergyPlus* simulator's continuous values requirements. It's crucial that this definition aligns with the previously defined actuators. In any case, *Sinergym* will highlight any inconsistencies.

.. note:: To adapt an environment to Gymnasium's ``Discrete``, ``MultiDiscrete``, or ``MultiBinary`` spaces, 
          similar to our predefined discrete environments, see section :ref:`DiscretizeEnv` and the 
          example in :ref:`Action discretization wrapper`.

.. important:: While *Sinergym*'s environments come with predefined observation and action variables (
               details available in `default_configuration <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/default_configuration>`__), 
               users are encouraged to explore and experiment with these spaces. For guidance, refer to 
               :ref:`Changing observation and action spaces`.

*Sinergym* also offers the option to create **empty action interfaces**. In this case, control is managed by the **default building model schedulers**. For more information, see the usage example in :ref:`Default building control using an empty action space`.

Extra configuration
===================

Parameters related to the building model and simulation, such as ``people occupant``, ``timesteps per simulation hour``, and ``runperiod``, can be set as extra configurations. These parameters are specified in the ``config_params`` argument, a Python Dictionary. For additional information on extra configurations in *Sinergym*, refer to :ref:`Extra Configuration in Sinergym simulations`.

*******************
Adding new weathers
*******************

*Sinergym* provides a variety of weather files of diverse global climates to enhance experimental diversity.

To incorporate a **new weather**:

1. Download an **EPW** and its corresponding **DDY** file from the `EnergyPlus page <https://energyplus.net/weather>`__.  The *DDY* file provides location and design day details.

2. Ensure both files share the same name, differing only in their extensions, and place them in the `weathers <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/weather>`__ folder.

*Sinergym* will automatically modify the ``SizingPeriod:DesignDays`` and ``Site:Location`` fields in the building model file using the *DDY* file.

********************
Adding new buildings
********************

Users can either modify existing environments or create new ones, incorporating new climates, actions, and observation spaces. It is also possible to incorporate new **building models** (epJSON file) apart from those currently supported.

To add new buildings to *Sinergym*, follow these steps:

  1. **Add your building file** (*epJSON*) to the `buildings <https://github.com/ugr-sail/sinergym/tree/main/sinergym/data/buildings>`__ directory. Ensure it's compatible with the EnergyPlus version used by *Sinergym*. If you're using an *IDF* file from an older version, update it with **IDFVersionUpdater** and convert it to *epJSON* format using **ConvertInputFormat**. Both tools are available in the EnergyPlus installation folder.

  2. **Adjust building objects** like ``RunPeriod`` and ``SimulationControl`` to suit your needs in Sinergym. We recommend setting ``run_simulation_for_sizing_periods`` to ``No`` in ``SimulationControl``. ``RunPeriod`` sets the episode length, which can be configured in the building file or Sinergym settings (see :ref:`runperiod`). Make these modifications in the *IDF* before step 1 or directly in the *epJSON* file.

  3. **Identify the components** of the building that you want to observe and control. This is the most challenging part of the process. Typically, users are already familiar with the building and know the *name* and *key* of the elements in advance. If not, follow the process below:

    a. Run a preliminary simulation with EnergyPlus directly, without any control, to check the different ``OutputVariables`` and ``Meters``. Consult the output files, specifically the *RDD* extension file, to identify possible observable variables.

    b. The challenge is knowing the names but not the possible *Keys* (EnergyPlus doesn't initially provide this information). Use these names to define the environment (see step 4). If the *Key* is incorrect, *Sinergym* will notify you of the error and provide a ``data_available.txt`` file in the output, as it has already connected with the EnergyPlus API. This file contains all the **controllable schedulers** for the actions and all the **observable variables**, now with their respective *Keys*, enabling the correct definition of the environment.

  4. With this information, the next step is **defining the environment** using the building model. You can:

    a. Use the *Sinergym* environment constructor directly. The arguments for building observation and control are explained within the class and should be specified in the same format as the EnergyPlus API.

    b. Set up the configuration to register environment IDs directly. For more information, refer to :ref:`Environments Configuration and Registration`. *Sinergym* will verify that the established configuration is correct and notify about any potential errors.

  5. If you used *Sinergym*'s registry, you will have access to environment IDs associated with your building. Use them with ``gym.make(<environment_id>)`` as usual. Besides, if you created an environment instance directly, use that instance to start interacting with the building.

.. note:: To obtain information about the environment instance with the new building model, refer to 
          :ref:`Getting information about Sinergym environments`.

