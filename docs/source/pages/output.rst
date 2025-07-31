###############
Sinergym output
###############

When a simulation is running, it creates a directory named ``<env_name>-res<num_simulation>``. The ``modeling`` module is responsible for managing the directory structure generated during a simulation.

The contents of this root output directory include the results of the simulation, and consists of the following: 

.. image:: /_static/output_structure.png
  :scale: 50 %
  :alt: Output directories and files.
  :align: center

|

.. important:: Optional files and directories are not shown in the image above. They are explained in the text below.

- ``episode-<num_episode>`` directories record the results of each simulation episode. The number of directories retained depends on the value specified by the ``max_ep_store`` parameter (see :ref:`Maximum episode data stored in Sinergym output`). Within these directories, the structure is consistent and follows the same format, including:

    - A copy of the ``environment.epJSON`` used during the simulation episode. This does not need to match the original one, as the simulation can be modified to accommodate specific user-defined settings when defining the Gymnasium environment.

    - A copy of the ``weather.epw`` used during the simulation episode. This file will not be identical to the original when using weather variability.

    - ``monitor/``: this directory contains information about the agent-environment interactions during the episode, timestep by timestep. It is only created when the environment has been wrapped with ``LoggerWrapper`` and ``CSVLogger`` (see :ref:`Logger Wrappers`). Several CSV files are included:
        
        - ``observations.csv``. This file contains the observations of the environment at each timestep. The header 
          includes the observation variables names. 
        
        - ``agent_actions.csv``. Contains the actions taken by the agent at each timestep. The header includes the
          action variables names.

        - ``simulated_actions.csv``. Contains the actions executed during the simulation at each timestep. This values
          do not have to be the same as the agent actions. For example, if actions are normalized. The header includes the action variables names.

        - ``rewards.csv``. This file contains the rewards obtained by the agent at each timestep.

        - ``infos.csv``. This file contains the ``info`` dictionary returned by the environment at each timestep. The header corresponds the dictionary keys. Some of these keys can be ignored depending on the ``CSVLogger`` configuration.

        - ``normalized_observations.csv``. It is only generated when the environment is wrapped with ``NormalizeObservation``
          (see :ref:`NormalizeObservation`). The structure is similar to ``observations.csv``, but the values are normalized.

        - ``custom_metrics.csv``. This file is created when the logger wrapper includes custom metrics. The header           includes the custom metric names.

    - ``output/``. This directory contains **EnergyPlus simulation output data**. To learn more about these files, 
      visit the `EnergyPlus Output File List <https://bigladdersoftware.com/epx/docs/24-1/output-details-and-examples/output-file-list.html#output-file-list>`__.

- ``progress.csv``. This file contains information about general simulation results. Each row contains episode information registering relevant data such as mean power consumption, rewards or comfort penalties. This file is only available when the environment has been wrapped with a ``LoggerWrapper`` and ``CSVLogger`` (see :ref:`Logger Wrappers` for more information). The structure of this file is defined by the ``LoggerWrapper`` class.

- ``weather_variability_config.csv``. This file contains the configuration of the weather variability for each episode. It is only created when the environment has been wrapped with ``LoggerWrapper`` and ``CSVLogger``. It is very useful when you are using ranges in weather variability paramters (more information in :ref:`Weather variability`)

- ``data_available.txt``. It is generated when the *EnergyPlus* API initializes all callbacks and handlers for the simulation. In this file, you can find all the available components of the building model, such as actuators, schedulers, meters, variables,  internal variables, etc.

- ``mean.txt``, ``var.txt`` and ``count.txt``. These files contain the count, mean and variation values for calibration of normalization in observation space if wrapper ``NormalizeObservation`` is used (see :ref:`NormalizeObservation`).

- ``env_config.pyyaml``. This file contains the environment configuration in YAML format. It is automatically generated when the environment is created and includes all the parameters used to instantiate the environment, such as the building model, weather file, simulation settings, and any user-defined configurations. For more information, see the :ref:`Environment Configuration Serialization` section.

- ``wrappers_config.pyyaml``. This file contains the configuration of the wrappers applied to the environment, formatted in YAML. It is generated by calling get_wrappers_info(env) after the wrappers have been applied. The file includes all parameters used to instantiate the wrappers, such as normalization settings, action scaling, and any user-defined configurations. For more information, see the :ref:`Wrapper Serialization and Restoration` section.

- ``evaluation/``. This directory contains the best model obtained during the training process. It is only created when the environment has been used with Stable Baselines 3 and wrapped with ``LoggerEvalCallback`` (see :ref:`LoggerEvalCallback`). The structure of this directory is as follows:

    - ``best_model.zip``. This file contains the best model obtained during the training process. It is saved in a compressed format.

    - ``mean.txt``, ``var.txt`` and ``count.txt``. Same as the files in the root directory, these files contain the count, mean and variation values for calibration of normalization in observation space, but for best model evaluation moments.

    - ``evaluation_summary.csv``. This file contains the evaluation summaries in CSV format. The structure of this file is defined by the ``LoggerEvalCallback`` class.

.. warning:: Some component lists, such as ``Output:Variable``, may not fully appear in `data_available.txt` because they need to
            be declared in the building model first. To view all variables or meters specifically, you should check the *EnergyPlus* output file. If you specify a valid variable in the environment, *Sinergym* will automatically add the ``Output:Variable`` element to the building model before the simulation begins.